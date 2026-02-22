"""Core training loop for VAE audio models.

Orchestrates forward passes, loss computation, gradient updates,
checkpoint saving, preview generation, and metrics emission.  The
``train`` function is the main entry point; it creates the model,
optimiser, scheduler, data loaders, and runs epochs until completion
or cancellation.

Key features:
- NaN detection skips bad gradient updates (MPS stability, pitfall #4).
- Overfitting gap and KL divergence monitored with log warnings.
- Cancel event triggers immediate checkpoint save.
- All file operations wrapped in try/except (project error isolation).
- Lazy imports for heavy dependencies (project pattern).

Design notes:
- ``train_epoch`` and ``validate_epoch`` are standalone for testability.
- ``train`` is the top-level orchestrator called by ``TrainingRunner``.
- Device memory reporting supports cuda, mps, and cpu.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

    import torch
    from torch.utils.data import DataLoader

    from distill.audio.spectrogram import AudioSpectrogram, SpectrogramConfig
    from distill.training.config import TrainingConfig, VQVAEConfig
    from distill.training.metrics import MetricsCallback, VQMetricsHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train Epoch
# ---------------------------------------------------------------------------


def train_epoch(
    model: "torch.nn.Module",
    train_loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    spectrogram: "AudioSpectrogram",
    kl_weight: float,
    device: "torch.device",
    gradient_clip_norm: float,
    free_bits: float = 0.1,
    epoch: int = 0,
    callback: "MetricsCallback | None" = None,
    cancel_event: "threading.Event | None" = None,
) -> dict:
    """Run one training epoch.

    Parameters
    ----------
    model:
        The VAE model (must be in train mode).
    train_loader:
        DataLoader yielding waveform batches ``[B, 1, samples]``.
    optimizer:
        Optimiser (AdamW).
    spectrogram:
        Spectrogram converter for waveform-to-mel.
    kl_weight:
        Current KL annealing weight.
    device:
        Target device for computation.
    gradient_clip_norm:
        Maximum gradient L2 norm for clipping.
    free_bits:
        Minimum KL per latent dimension.
    epoch:
        Current epoch number (for metrics).
    callback:
        Optional metrics event subscriber.
    cancel_event:
        If set, returns early with partial results.

    Returns
    -------
    dict
        ``{train_loss, recon_loss, kl_loss}`` averaged over all steps.
    """
    import torch  # noqa: WPS433

    from distill.models.losses import vae_loss
    from distill.training.metrics import StepMetrics

    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    valid_steps = 0
    total_steps = len(train_loader)
    lr = optimizer.param_groups[0]["lr"]

    for step, batch in enumerate(train_loader):
        # Check cancellation
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel detected during training epoch %d, step %d", epoch, step)
            break

        step_start = time.time()

        # Move waveform batch to device
        batch = batch.to(device)

        # Convert to mel spectrogram
        mel = spectrogram.waveform_to_mel(batch)

        # Forward pass
        recon, mu, logvar = model(mel)

        # Compute loss
        total, recon_loss, kl_loss = vae_loss(
            recon, mel, mu, logvar, kl_weight=kl_weight, free_bits=free_bits,
        )

        # NaN detection: skip bad gradient updates (MPS stability)
        if total.isnan():
            logger.warning(
                "NaN loss at epoch %d step %d -- skipping gradient update",
                epoch, step,
            )
            step_time = time.time() - step_start
            if callback is not None:
                callback(StepMetrics(
                    epoch=epoch, step=step, total_steps=total_steps,
                    train_loss=float("nan"), recon_loss=float("nan"),
                    kl_loss=float("nan"), kl_weight=kl_weight,
                    learning_rate=lr, step_time_s=step_time,
                ))
            continue

        # Backward pass
        optimizer.zero_grad()
        total.backward()

        # Gradient clipping
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        optimizer.step()

        # Accumulate
        total_loss_sum += total.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        valid_steps += 1

        step_time = time.time() - step_start

        # Log progress every 10 steps (or every step in first epoch)
        if step % 10 == 0 or epoch == 0:
            print(
                f"[TRAIN] Epoch {epoch} step {step + 1}/{total_steps}  "
                f"loss={total.item():.4f}  recon={recon_loss.item():.4f}  "
                f"kl={kl_loss.item():.4f}  kl_w={kl_weight:.3f}  "
                f"step_time={step_time:.2f}s",
                flush=True,
            )

        # Emit step metrics
        if callback is not None:
            callback(StepMetrics(
                epoch=epoch, step=step, total_steps=total_steps,
                train_loss=total.item(), recon_loss=recon_loss.item(),
                kl_loss=kl_loss.item(), kl_weight=kl_weight,
                learning_rate=lr, step_time_s=step_time,
            ))

    # Average over valid steps
    if valid_steps > 0:
        avg_loss = total_loss_sum / valid_steps
        avg_recon = recon_loss_sum / valid_steps
        avg_kl = kl_loss_sum / valid_steps
    else:
        avg_loss = float("nan")
        avg_recon = float("nan")
        avg_kl = float("nan")

    return {
        "train_loss": avg_loss,
        "recon_loss": avg_recon,
        "kl_loss": avg_kl,
    }


# ---------------------------------------------------------------------------
# Validate Epoch
# ---------------------------------------------------------------------------


def validate_epoch(
    model: "torch.nn.Module",
    val_loader: "DataLoader",
    spectrogram: "AudioSpectrogram",
    kl_weight: float,
    device: "torch.device",
    free_bits: float = 0.1,
) -> dict:
    """Run one validation epoch.

    Parameters
    ----------
    model:
        The VAE model (switched to eval mode internally).
    val_loader:
        DataLoader yielding waveform batches ``[B, 1, samples]``.
    spectrogram:
        Spectrogram converter for waveform-to-mel.
    kl_weight:
        Current KL annealing weight.
    device:
        Target device for computation.
    free_bits:
        Minimum KL per latent dimension.

    Returns
    -------
    dict
        ``{val_loss, val_recon_loss, val_kl_loss, kl_divergence}``.
    """
    import torch  # noqa: WPS433

    from distill.models.losses import compute_kl_divergence, vae_loss

    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    kl_div_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            mel = spectrogram.waveform_to_mel(batch)

            recon, mu, logvar = model(mel)
            total, recon_loss, kl_loss = vae_loss(
                recon, mel, mu, logvar, kl_weight=kl_weight, free_bits=free_bits,
            )

            # Raw KL divergence for posterior collapse monitoring
            raw_kl = compute_kl_divergence(mu, logvar)

            total_loss_sum += total.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            kl_div_sum += raw_kl
            num_batches += 1

    model.train()

    if num_batches > 0:
        return {
            "val_loss": total_loss_sum / num_batches,
            "val_recon_loss": recon_loss_sum / num_batches,
            "val_kl_loss": kl_loss_sum / num_batches,
            "kl_divergence": kl_div_sum / num_batches,
        }

    return {
        "val_loss": float("nan"),
        "val_recon_loss": float("nan"),
        "val_kl_loss": float("nan"),
        "kl_divergence": 0.0,
    }


# ---------------------------------------------------------------------------
# Main Training Orchestrator
# ---------------------------------------------------------------------------


def train(
    config: "TrainingConfig",
    file_paths: list[Path],
    output_dir: Path,
    device: "torch.device",
    callback: "MetricsCallback | None" = None,
    cancel_event: "threading.Event | None" = None,
    resume_checkpoint: "Path | None" = None,
    models_dir: "Path | None" = None,
    dataset_name: str = "",
    model_name: str = "",
) -> dict:
    """Full training orchestrator.

    Creates the model, optimiser, scheduler, data loaders, and runs
    the training loop from start (or resume point) to completion or
    cancellation.

    Parameters
    ----------
    config:
        Training configuration.
    file_paths:
        Paths to audio files for the dataset.
    output_dir:
        Root output directory (checkpoints/ and previews/ created inside).
    device:
        Target device (e.g. ``torch.device("cpu")``).
    callback:
        Optional metrics event subscriber.
    cancel_event:
        If set, training stops and saves a checkpoint.
    resume_checkpoint:
        Path to a ``.pt`` checkpoint file to resume from.

    Returns
    -------
    dict
        ``{model, metrics_history, output_dir, best_checkpoint_path}``.
    """
    import torch  # noqa: WPS433

    from distill.audio.spectrogram import AudioSpectrogram, SpectrogramConfig
    from distill.models.losses import get_kl_weight
    from distill.models.vae import ConvVAE
    from distill.training.checkpoint import (
        get_best_checkpoint,
        load_checkpoint,
        manage_checkpoints,
        save_checkpoint,
    )
    from distill.training.config import get_effective_preview_interval
    from distill.training.dataset import create_data_loaders
    from distill.training.metrics import (
        EpochMetrics,
        MetricsHistory,
        PreviewEvent,
        TrainingCompleteEvent,
    )
    from distill.training.preview import generate_preview

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    print(f"[TRAIN] train() called: {len(file_paths)} files, device={device}", flush=True)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Create spectrogram converter
    spec_config = SpectrogramConfig()
    spectrogram = AudioSpectrogram(spec_config)

    # Create model
    model = ConvVAE(
        latent_dim=config.latent_dim,
        dropout=config.regularization.dropout,
    )
    model = model.to(device)
    spectrogram.to(device)

    # Initialize lazy linear layers before any load_state_dict call
    # (required for checkpoint resume -- lazy layers must be materialized first)
    n_mels = spec_config.n_mels
    time_frames = spec_config.sample_rate // spec_config.hop_length + 1
    pad_h = (16 - n_mels % 16) % 16
    pad_w = (16 - time_frames % 16) % 16
    padded_h = n_mels + pad_h
    padded_w = time_frames + pad_w
    spatial = (padded_h // 16, padded_w // 16)
    model.decoder._init_linear(spatial)
    flatten_dim = 256 * spatial[0] * spatial[1]
    model.encoder._init_linear(flatten_dim)

    # Create optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.regularization.weight_decay,
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs,
    )

    # Metrics history
    metrics_history = MetricsHistory()

    # Resume state
    start_epoch = 0
    kl_weight_override: float | None = None

    if resume_checkpoint is not None:
        try:
            ckpt = load_checkpoint(
                resume_checkpoint, model, optimizer, scheduler,
                device=str(device),
            )
            start_epoch = ckpt.get("epoch", 0) + 1
            kl_weight_override = ckpt.get("kl_weight")

            # Restore metrics history
            if "metrics_history" in ckpt:
                metrics_history = MetricsHistory.from_dict(ckpt["metrics_history"])

            logger.info(
                "Resumed training from epoch %d (val_loss=%.4f)",
                start_epoch, ckpt.get("val_loss", float("inf")),
            )
        except Exception:
            logger.warning(
                "Failed to load resume checkpoint %s, starting fresh",
                resume_checkpoint, exc_info=True,
            )

    # -----------------------------------------------------------------
    # Create data loaders
    # -----------------------------------------------------------------
    augmentation_pipeline = None
    if config.regularization.augmentation_expansion > 0:
        try:
            from distill.audio.augmentation import (
                AugmentationConfig,
                AugmentationPipeline,
            )
            aug_config = AugmentationConfig(
                expansion_ratio=config.regularization.augmentation_expansion,
            )
            # Disable PitchShift -- it runs on CPU during data loading
            # (even with CUDA training) and is unusably slow at 48kHz
            # (minutes per chunk via STFT).
            aug_config.pitch_shift_probability = 0.0
            print("[TRAIN] PitchShift disabled (too slow for real-time data loading)", flush=True)
            print(f"[TRAIN] Creating augmentation pipeline...", flush=True)
            augmentation_pipeline = AugmentationPipeline(config=aug_config)
            print(f"[TRAIN] Augmentation pipeline ready", flush=True)
        except Exception as exc:
            print(f"[TRAIN] Augmentation failed: {exc}", flush=True)
            logger.warning("Failed to create augmentation pipeline", exc_info=True)

    train_loader, val_loader = create_data_loaders(
        file_paths, config, augmentation_pipeline,
    )

    train_chunks = len(train_loader.dataset)
    val_chunks = len(val_loader.dataset)
    num_batches = len(train_loader)
    print(
        f"[TRAIN] {train_chunks} train chunks, {val_chunks} val chunks, "
        f"batch_size={config.batch_size}, {num_batches} batches/epoch",
        flush=True,
    )

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    effective_preview_interval = get_effective_preview_interval(config)
    train_start_time = time.time()

    for epoch in range(start_epoch, config.max_epochs):
        epoch_start = time.time()
        print(f"[TRAIN] Starting epoch {epoch + 1}/{config.max_epochs}", flush=True)

        # KL weight (annealing)
        kl_weight = get_kl_weight(
            epoch, config.max_epochs, config.kl_warmup_fraction,
            kl_weight_max=config.kl_weight_max,
        )

        # Train epoch
        train_results = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            spectrogram=spectrogram,
            kl_weight=kl_weight,
            device=device,
            gradient_clip_norm=config.regularization.gradient_clip_norm,
            free_bits=config.free_bits,
            epoch=epoch,
            callback=callback,
            cancel_event=cancel_event,
        )

        # Check cancellation after train epoch
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel detected after train epoch %d, saving checkpoint", epoch)
            _save_checkpoint_safe(
                checkpoint_dir, model, optimizer, scheduler, epoch, 0,
                train_results["train_loss"], float("inf"), kl_weight,
                config, spec_config, metrics_history,
            )
            if callback is not None:
                elapsed = time.time() - train_start_time
                callback(TrainingCompleteEvent(
                    total_epochs=epoch + 1,
                    total_time_s=elapsed,
                    final_train_loss=train_results["train_loss"],
                    final_val_loss=float("inf"),
                    best_val_loss=_get_best_val_loss(metrics_history),
                    best_epoch=metrics_history.get_best_epoch(),
                ))
            return {
                "model": model,
                "metrics_history": metrics_history,
                "output_dir": output_dir,
                "best_checkpoint_path": get_best_checkpoint(checkpoint_dir),
            }

        # Validate epoch
        val_results = validate_epoch(
            model=model,
            val_loader=val_loader,
            spectrogram=spectrogram,
            kl_weight=kl_weight,
            device=device,
            free_bits=config.free_bits,
        )

        # Step scheduler
        scheduler.step()

        # Compute metrics
        t_loss = train_results["train_loss"]
        v_loss = val_results["val_loss"]
        kl_divergence = val_results["kl_divergence"]

        # Overfitting gap
        if t_loss > 0 and not (t_loss != t_loss):  # not NaN
            overfitting_gap = (v_loss - t_loss) / t_loss
        else:
            overfitting_gap = 0.0

        elapsed = time.time() - train_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # ETA
        eta = metrics_history.compute_eta(epoch - start_epoch, config.max_epochs - start_epoch)

        # Record epoch metrics
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            total_epochs=config.max_epochs,
            train_loss=t_loss,
            val_loss=v_loss,
            val_recon_loss=val_results["val_recon_loss"],
            val_kl_loss=val_results["val_kl_loss"],
            kl_divergence=kl_divergence,
            overfitting_gap=overfitting_gap,
            learning_rate=current_lr,
            eta_seconds=eta,
            elapsed_seconds=elapsed,
        )
        metrics_history.add_epoch(epoch_metrics)

        if callback is not None:
            callback(epoch_metrics)

        # Overfitting warning
        if overfitting_gap > 0.2:
            logger.warning(
                "Overfitting detected at epoch %d: gap=%.2f%% "
                "(train=%.4f, val=%.4f). Training continues -- "
                "user decides when to stop.",
                epoch, overfitting_gap * 100, t_loss, v_loss,
            )

        # Posterior collapse warning
        if kl_divergence < 0.5:
            logger.warning(
                "Low KL divergence at epoch %d: %.4f (< 0.5 threshold). "
                "Possible posterior collapse.",
                epoch, kl_divergence,
            )

        # Preview generation
        if effective_preview_interval > 0 and epoch > 0 and epoch % effective_preview_interval == 0:
            try:
                preview_paths = generate_preview(
                    model=model,
                    spectrogram=spectrogram,
                    output_dir=preview_dir,
                    epoch=epoch,
                    device=device,
                )
                for p in preview_paths:
                    if callback is not None:
                        callback(PreviewEvent(
                            epoch=epoch,
                            audio_path=str(p),
                            sample_rate=spec_config.sample_rate,
                        ))
            except Exception:
                logger.warning("Preview generation failed at epoch %d", epoch, exc_info=True)

        # Checkpoint saving
        if config.checkpoint_interval > 0 and epoch > 0 and epoch % config.checkpoint_interval == 0:
            _save_checkpoint_safe(
                checkpoint_dir, model, optimizer, scheduler, epoch, 0,
                t_loss, v_loss, kl_weight, config, spec_config, metrics_history,
            )
            try:
                manage_checkpoints(checkpoint_dir, max_recent=config.max_checkpoints)
            except Exception:
                logger.warning("Checkpoint management failed at epoch %d", epoch, exc_info=True)

        # Cancel check at end of epoch
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel detected at end of epoch %d, saving checkpoint", epoch)
            _save_checkpoint_safe(
                checkpoint_dir, model, optimizer, scheduler, epoch, 0,
                t_loss, v_loss, kl_weight, config, spec_config, metrics_history,
            )
            if callback is not None:
                callback(TrainingCompleteEvent(
                    total_epochs=epoch + 1,
                    total_time_s=elapsed,
                    final_train_loss=t_loss,
                    final_val_loss=v_loss,
                    best_val_loss=_get_best_val_loss(metrics_history),
                    best_epoch=metrics_history.get_best_epoch(),
                ))
            return {
                "model": model,
                "metrics_history": metrics_history,
                "output_dir": output_dir,
                "best_checkpoint_path": get_best_checkpoint(checkpoint_dir),
            }

    # -----------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------
    elapsed = time.time() - train_start_time

    # Save final checkpoint
    final_epoch = config.max_epochs - 1
    final_train = metrics_history.epoch_metrics[-1].train_loss if metrics_history.epoch_metrics else 0.0
    final_val = metrics_history.epoch_metrics[-1].val_loss if metrics_history.epoch_metrics else 0.0
    final_kl = get_kl_weight(
        final_epoch, config.max_epochs, config.kl_warmup_fraction,
        kl_weight_max=config.kl_weight_max,
    )

    _save_checkpoint_safe(
        checkpoint_dir, model, optimizer, scheduler, final_epoch, 0,
        final_train, final_val, final_kl, config, spec_config, metrics_history,
    )
    try:
        manage_checkpoints(checkpoint_dir, max_recent=config.max_checkpoints)
    except Exception:
        logger.warning("Final checkpoint management failed", exc_info=True)

    # Run latent space analysis
    analysis_result = None
    try:
        from distill.controls.analyzer import LatentSpaceAnalyzer
        from distill.controls.serialization import analysis_to_dict
        from distill.training.dataset import AudioTrainingDataset
        from torch.utils.data import DataLoader as TorchDataLoader

        logger.info("Running latent space analysis...")
        analyzer = LatentSpaceAnalyzer()

        # Use ALL training files (not split) for maximum PCA coverage
        analysis_dataset = AudioTrainingDataset(
            file_paths=file_paths,
            chunk_samples=int(1.0 * 48_000),
            augmentation_pipeline=None,
        )
        analysis_loader = TorchDataLoader(
            analysis_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        analysis_result = analyzer.analyze(
            model=model,
            dataloader=analysis_loader,
            spectrogram=spectrogram,
            device=device,
        )
        logger.info(
            "Analysis complete: %d active components, %.1f%% variance explained",
            analysis_result.n_active_components,
            sum(analysis_result.explained_variance_ratio) * 100,
        )

        # Re-save the final checkpoint WITH analysis included
        analysis_dict = analysis_to_dict(analysis_result)
        _save_checkpoint_safe(
            checkpoint_dir, model, optimizer, scheduler, final_epoch, 0,
            final_train, final_val, final_kl, config, spec_config, metrics_history,
            latent_analysis=analysis_dict,
        )
    except Exception:
        logger.warning("Latent space analysis failed -- model saved without analysis", exc_info=True)

    # Save as .distill model to the library
    saved_model_path = None
    if models_dir is not None:
        try:
            from distill.models.persistence import ModelMetadata, save_model

            metadata = ModelMetadata(
                name=model_name or dataset_name or "Untitled Model",
                dataset_name=dataset_name,
                dataset_file_count=len(file_paths),
                training_epochs=config.max_epochs,
                final_train_loss=final_train,
                final_val_loss=final_val,
                has_analysis=analysis_result is not None,
                n_active_components=(
                    analysis_result.n_active_components
                    if analysis_result is not None else 0
                ),
            )
            saved_model_path = save_model(
                model=model,
                spectrogram_config=asdict(spec_config),
                training_config=asdict(config),
                metadata=metadata,
                models_dir=models_dir,
                analysis=analysis_result,
            )
            print(f"[TRAIN] Model saved to library: {saved_model_path}", flush=True)
        except Exception as exc:
            print(f"[TRAIN] Failed to save model to library: {exc}", flush=True)
            logger.warning("Failed to save model to library", exc_info=True)

    # Emit completion event
    if callback is not None:
        callback(TrainingCompleteEvent(
            total_epochs=config.max_epochs,
            total_time_s=elapsed,
            final_train_loss=final_train,
            final_val_loss=final_val,
            best_val_loss=_get_best_val_loss(metrics_history),
            best_epoch=metrics_history.get_best_epoch(),
        ))

    best_ckpt = get_best_checkpoint(checkpoint_dir)
    logger.info(
        "Training complete: %d epochs in %.1fs (best val_loss=%.4f at epoch %d)",
        config.max_epochs, elapsed,
        _get_best_val_loss(metrics_history),
        metrics_history.get_best_epoch(),
    )

    return {
        "model": model,
        "metrics_history": metrics_history,
        "output_dir": output_dir,
        "best_checkpoint_path": best_ckpt,
        "analysis": analysis_result,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_checkpoint_safe(
    checkpoint_dir: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler",
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    kl_weight: float,
    config: "TrainingConfig",
    spec_config: "SpectrogramConfig",
    metrics_history: "MetricsHistory",
    latent_analysis: dict | None = None,
) -> None:
    """Save a checkpoint, catching and logging any errors."""
    from dataclasses import asdict as _asdict

    from distill.training.checkpoint import save_checkpoint

    try:
        path = checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            kl_weight=kl_weight,
            training_config=_asdict(config),
            spectrogram_config=_asdict(spec_config),
            metrics_history_dict=metrics_history.to_dict(),
            latent_analysis=latent_analysis,
        )
    except Exception:
        logger.warning("Failed to save checkpoint at epoch %d", epoch, exc_info=True)


def _get_best_val_loss(metrics_history: "MetricsHistory") -> float:
    """Return the best validation loss from history."""
    if not metrics_history.epoch_metrics:
        return float("inf")
    return min(m.val_loss for m in metrics_history.epoch_metrics)


def _get_best_vq_val_loss(metrics_history: "VQMetricsHistory") -> float:
    """Return the best validation loss from VQ metrics history."""
    if not metrics_history.epoch_metrics:
        return float("inf")
    return min(m.val_loss for m in metrics_history.epoch_metrics)


# ---------------------------------------------------------------------------
# VQ-VAE Training Functions (v1.1)
# ---------------------------------------------------------------------------

# Codebook health check interval (every N steps during training)
_VQ_HEALTH_INTERVAL = 10


def train_vqvae_epoch(
    model: "torch.nn.Module",
    train_loader: "DataLoader",
    optimizer: "torch.optim.Optimizer",
    spectrogram: "AudioSpectrogram",
    device: "torch.device",
    gradient_clip_norm: float,
    commitment_weight: float = 0.25,
    epoch: int = 0,
    callback: "MetricsCallback | None" = None,
    cancel_event: "threading.Event | None" = None,
) -> dict:
    """Run one VQ-VAE training epoch.

    Mirrors :func:`train_epoch` but uses ``ConvVQVAE`` + ``vqvae_loss``
    instead of ``ConvVAE`` + ``vae_loss``.  There is no KL divergence,
    no KL annealing, and no free bits.

    Parameters
    ----------
    model:
        The VQ-VAE model (must be in train mode).
    train_loader:
        DataLoader yielding waveform batches ``[B, 1, samples]``.
    optimizer:
        Optimiser (AdamW).
    spectrogram:
        Spectrogram converter for waveform-to-mel.
    device:
        Target device for computation.
    gradient_clip_norm:
        Maximum gradient L2 norm for clipping.
    commitment_weight:
        Weight applied to commitment loss (default 0.25).
    epoch:
        Current epoch number (for metrics).
    callback:
        Optional metrics event subscriber.
    cancel_event:
        If set, returns early with partial results.

    Returns
    -------
    dict
        ``{train_loss, recon_loss, commit_loss}`` averaged over all steps.
    """
    import torch  # noqa: WPS433

    from distill.models.losses import vqvae_loss
    from distill.training.metrics import VQStepMetrics

    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    commit_loss_sum = 0.0
    valid_steps = 0
    total_steps = len(train_loader)
    lr = optimizer.param_groups[0]["lr"]

    for step, batch in enumerate(train_loader):
        # Check cancellation
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel detected during VQ-VAE training epoch %d, step %d", epoch, step)
            break

        step_start = time.time()

        # Move waveform batch to device
        batch = batch.to(device)

        # Convert to mel spectrogram
        mel = spectrogram.waveform_to_mel(batch)

        # Forward pass (VQ-VAE returns recon, indices, commit_loss)
        recon, indices, commit_loss = model(mel)

        # Compute loss
        total, recon_loss, weighted_commit = vqvae_loss(
            recon, mel, commit_loss, commitment_weight,
        )

        # NaN detection: skip bad gradient updates (MPS stability)
        if total.isnan():
            logger.warning(
                "NaN loss at epoch %d step %d -- skipping gradient update",
                epoch, step,
            )
            step_time = time.time() - step_start
            if callback is not None:
                callback(VQStepMetrics(
                    epoch=epoch, step=step, total_steps=total_steps,
                    train_loss=float("nan"), recon_loss=float("nan"),
                    commit_loss=float("nan"), commitment_weight=commitment_weight,
                    learning_rate=lr, step_time_s=step_time,
                ))
            continue

        # Backward pass
        optimizer.zero_grad()
        total.backward()

        # Gradient clipping
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        optimizer.step()

        # Accumulate
        total_loss_sum += total.item()
        recon_loss_sum += recon_loss.item()
        commit_loss_sum += weighted_commit.item()
        valid_steps += 1

        step_time = time.time() - step_start

        # Codebook health: compute every _VQ_HEALTH_INTERVAL steps
        # Skip step 0 of epoch 0 (k-means not yet initialized, misleading 0%)
        codebook_health = None
        if step % _VQ_HEALTH_INTERVAL == 0 and not (epoch == 0 and step == 0):
            try:
                codebook_health = model.quantizer.get_codebook_utilization(indices)
            except Exception:
                logger.warning("Codebook health check failed at epoch %d step %d", epoch, step)

        # Log progress every 10 steps (or every step in first epoch)
        if step % 10 == 0 or epoch == 0:
            print(
                f"[VQ-TRAIN] Epoch {epoch} step {step + 1}/{total_steps}  "
                f"loss={total.item():.4f}  recon={recon_loss.item():.4f}  "
                f"commit={weighted_commit.item():.4f}  "
                f"step_time={step_time:.2f}s",
                flush=True,
            )

        # Emit step metrics
        if callback is not None:
            callback(VQStepMetrics(
                epoch=epoch, step=step, total_steps=total_steps,
                train_loss=total.item(), recon_loss=recon_loss.item(),
                commit_loss=weighted_commit.item(),
                commitment_weight=commitment_weight,
                learning_rate=lr, step_time_s=step_time,
                codebook_health=codebook_health,
            ))

    # Average over valid steps
    if valid_steps > 0:
        avg_loss = total_loss_sum / valid_steps
        avg_recon = recon_loss_sum / valid_steps
        avg_commit = commit_loss_sum / valid_steps
    else:
        avg_loss = float("nan")
        avg_recon = float("nan")
        avg_commit = float("nan")

    return {
        "train_loss": avg_loss,
        "recon_loss": avg_recon,
        "commit_loss": avg_commit,
    }


# ---------------------------------------------------------------------------
# VQ-VAE Validate Epoch
# ---------------------------------------------------------------------------


def validate_vqvae_epoch(
    model: "torch.nn.Module",
    val_loader: "DataLoader",
    spectrogram: "AudioSpectrogram",
    device: "torch.device",
    commitment_weight: float = 0.25,
) -> dict:
    """Run one VQ-VAE validation epoch.

    Mirrors :func:`validate_epoch` but uses ``ConvVQVAE`` + ``vqvae_loss``.
    Computes codebook health on the full validation set by accumulating
    indices across all batches and calling ``get_codebook_utilization()``
    once at the end.

    Parameters
    ----------
    model:
        The VQ-VAE model (switched to eval mode internally).
    val_loader:
        DataLoader yielding waveform batches ``[B, 1, samples]``.
    spectrogram:
        Spectrogram converter for waveform-to-mel.
    device:
        Target device for computation.
    commitment_weight:
        Weight applied to commitment loss (default 0.25).

    Returns
    -------
    dict
        ``{val_loss, val_recon_loss, val_commit_loss, codebook_health}``.
    """
    import torch  # noqa: WPS433

    from distill.models.losses import vqvae_loss

    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    commit_loss_sum = 0.0
    num_batches = 0
    all_indices: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            mel = spectrogram.waveform_to_mel(batch)

            recon, indices, commit_loss = model(mel)
            total, recon_loss, weighted_commit = vqvae_loss(
                recon, mel, commit_loss, commitment_weight,
            )

            total_loss_sum += total.item()
            recon_loss_sum += recon_loss.item()
            commit_loss_sum += weighted_commit.item()
            num_batches += 1

            # Accumulate indices for codebook health computation
            all_indices.append(indices)

    model.train()

    # Compute codebook health on accumulated validation indices
    codebook_health = None
    if all_indices:
        try:
            combined_indices = torch.cat(all_indices, dim=0)
            codebook_health = model.quantizer.get_codebook_utilization(combined_indices)
        except Exception:
            logger.warning("Codebook health computation failed during validation")

    if num_batches > 0:
        return {
            "val_loss": total_loss_sum / num_batches,
            "val_recon_loss": recon_loss_sum / num_batches,
            "val_commit_loss": commit_loss_sum / num_batches,
            "codebook_health": codebook_health,
        }

    return {
        "val_loss": float("nan"),
        "val_recon_loss": float("nan"),
        "val_commit_loss": float("nan"),
        "codebook_health": None,
    }


# ---------------------------------------------------------------------------
# VQ-VAE Main Training Orchestrator
# ---------------------------------------------------------------------------


def train_vqvae(
    config: "VQVAEConfig",
    file_paths: list[Path],
    output_dir: Path,
    device: "torch.device",
    callback: "MetricsCallback | None" = None,
    cancel_event: "threading.Event | None" = None,
    models_dir: "Path | None" = None,
    dataset_name: str = "",
    model_name: str = "",
) -> dict:
    """Full VQ-VAE training orchestrator.

    Creates the VQ-VAE model, optimiser, scheduler, data loaders, and runs
    the training loop from start to completion or cancellation.  Mirrors
    :func:`train` but uses ``ConvVQVAE``, ``vqvae_loss``, and emits
    VQ-specific metrics with codebook health.

    Parameters
    ----------
    config:
        VQ-VAE training configuration.
    file_paths:
        Paths to audio files for the dataset.
    output_dir:
        Root output directory (checkpoints/ and previews/ created inside).
    device:
        Target device (e.g. ``torch.device("cpu")``).
    callback:
        Optional metrics event subscriber.
    cancel_event:
        If set, training stops and saves a checkpoint.
    models_dir:
        Directory to save the .distill model file into.
    dataset_name:
        Name of the dataset (used for model metadata).
    model_name:
        User-specified model name (used for saved model name).

    Returns
    -------
    dict
        ``{model, metrics_history, output_dir, best_checkpoint_path,
        final_codebook_health}``.
    """
    import torch  # noqa: WPS433

    from distill.audio.spectrogram import AudioSpectrogram, SpectrogramConfig
    from distill.models.vqvae import ConvVQVAE
    from distill.training.checkpoint import (
        get_best_checkpoint,
        manage_checkpoints,
    )
    from distill.training.dataset import create_data_loaders
    from distill.training.metrics import (
        PreviewEvent,
        TrainingCompleteEvent,
        VQEpochMetrics,
        VQMetricsHistory,
    )
    from distill.training.preview import generate_vqvae_reconstruction_preview

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    print(f"[VQ-TRAIN] train_vqvae() called: {len(file_paths)} files, device={device}", flush=True)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Create spectrogram converter
    spec_config = SpectrogramConfig()
    spectrogram = AudioSpectrogram(spec_config)

    # Create VQ-VAE model
    model = ConvVQVAE(
        codebook_dim=config.codebook_dim,
        codebook_size=config.codebook_size,
        num_quantizers=config.num_quantizers,
        decay=config.decay,
        commitment_weight=config.commitment_weight,
        threshold_ema_dead_code=config.threshold_ema_dead_code,
        dropout=config.dropout,
    )
    model = model.to(device)
    spectrogram.to(device)

    # Create optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs,
    )

    # Metrics history
    metrics_history = VQMetricsHistory()

    # -----------------------------------------------------------------
    # Create data loaders (reuse from v1.0 dataset module)
    # -----------------------------------------------------------------
    # Build a minimal TrainingConfig-like object for create_data_loaders
    # which expects val_fraction, chunk_duration_s, batch_size, num_workers
    from distill.training.config import RegularizationConfig, TrainingConfig

    loader_config = TrainingConfig(
        batch_size=config.batch_size,
        val_fraction=config.val_fraction,
        chunk_duration_s=config.chunk_duration_s,
        num_workers=config.num_workers,
        regularization=RegularizationConfig(
            augmentation_expansion=config.augmentation_expansion,
        ),
    )

    augmentation_pipeline = None
    if config.augmentation_expansion > 0:
        try:
            from distill.audio.augmentation import (
                AugmentationConfig,
                AugmentationPipeline,
            )
            aug_config = AugmentationConfig(
                expansion_ratio=config.augmentation_expansion,
            )
            aug_config.pitch_shift_probability = 0.0
            print("[VQ-TRAIN] PitchShift disabled (too slow for real-time data loading)", flush=True)
            augmentation_pipeline = AugmentationPipeline(config=aug_config)
            print("[VQ-TRAIN] Augmentation pipeline ready", flush=True)
        except Exception as exc:
            print(f"[VQ-TRAIN] Augmentation failed: {exc}", flush=True)
            logger.warning("Failed to create augmentation pipeline", exc_info=True)

    train_loader, val_loader = create_data_loaders(
        file_paths, loader_config, augmentation_pipeline,
    )

    train_chunks = len(train_loader.dataset)
    val_chunks = len(val_loader.dataset)
    num_batches = len(train_loader)
    print(
        f"[VQ-TRAIN] {train_chunks} train chunks, {val_chunks} val chunks, "
        f"batch_size={config.batch_size}, {num_batches} batches/epoch",
        flush=True,
    )

    # -----------------------------------------------------------------
    # Preview interval
    # -----------------------------------------------------------------
    if config.max_epochs < 50:
        effective_preview_interval = max(2, config.preview_interval)
    else:
        effective_preview_interval = config.preview_interval

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    train_start_time = time.time()

    for epoch in range(config.max_epochs):
        print(f"[VQ-TRAIN] Starting epoch {epoch + 1}/{config.max_epochs}", flush=True)

        # Train epoch
        train_results = train_vqvae_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            spectrogram=spectrogram,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
            commitment_weight=config.commitment_weight,
            epoch=epoch,
            callback=callback,
            cancel_event=cancel_event,
        )

        # Check cancellation after train epoch
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel detected after VQ-VAE train epoch %d, saving checkpoint", epoch)
            _save_vqvae_checkpoint_safe(
                checkpoint_dir, model, optimizer, scheduler, epoch, 0,
                train_results["train_loss"], float("inf"),
                config, spec_config, metrics_history,
            )
            if callback is not None:
                elapsed = time.time() - train_start_time
                callback(TrainingCompleteEvent(
                    total_epochs=epoch + 1,
                    total_time_s=elapsed,
                    final_train_loss=train_results["train_loss"],
                    final_val_loss=float("inf"),
                    best_val_loss=_get_best_vq_val_loss(metrics_history),
                    best_epoch=metrics_history.get_best_epoch(),
                ))
            return {
                "model": model,
                "metrics_history": metrics_history,
                "output_dir": output_dir,
                "best_checkpoint_path": get_best_checkpoint(checkpoint_dir),
                "final_codebook_health": None,
            }

        # Validate epoch
        val_results = validate_vqvae_epoch(
            model=model,
            val_loader=val_loader,
            spectrogram=spectrogram,
            device=device,
            commitment_weight=config.commitment_weight,
        )

        # Step scheduler
        scheduler.step()

        # Compute metrics
        t_loss = train_results["train_loss"]
        v_loss = val_results["val_loss"]
        codebook_health = val_results["codebook_health"]

        # Overfitting gap
        if t_loss > 0 and not (t_loss != t_loss):  # not NaN
            overfitting_gap = (v_loss - t_loss) / t_loss
        else:
            overfitting_gap = 0.0

        elapsed = time.time() - train_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # ETA
        eta = metrics_history.compute_eta(epoch, config.max_epochs)

        # Low utilization warnings (VQVAE-07)
        utilization_warnings: list[str] | None = None
        if codebook_health is not None and epoch > 0:
            warnings_list: list[str] = []
            for level_name, level_data in sorted(codebook_health.items()):
                util = level_data.get("utilization", 1.0)
                if util < 0.30:
                    level_num = level_name.replace("level_", "")
                    warning_msg = f"Level {level_num}: utilization {util:.0%} (below 30% threshold)"
                    warnings_list.append(warning_msg)
                    logger.warning(
                        "Low codebook utilization at epoch %d: %s",
                        epoch, warning_msg,
                    )
            if warnings_list:
                utilization_warnings = warnings_list

        # Record epoch metrics
        epoch_metrics = VQEpochMetrics(
            epoch=epoch,
            total_epochs=config.max_epochs,
            train_loss=t_loss,
            val_loss=v_loss,
            val_recon_loss=val_results["val_recon_loss"],
            val_commit_loss=val_results["val_commit_loss"],
            overfitting_gap=overfitting_gap,
            learning_rate=current_lr,
            eta_seconds=eta,
            elapsed_seconds=elapsed,
            codebook_health=codebook_health,
            utilization_warnings=utilization_warnings,
        )
        metrics_history.add_epoch(epoch_metrics)

        if callback is not None:
            callback(epoch_metrics)

        # Overfitting warning
        if overfitting_gap > 0.2:
            logger.warning(
                "Overfitting detected at epoch %d: gap=%.2f%% "
                "(train=%.4f, val=%.4f). Training continues -- "
                "user decides when to stop.",
                epoch, overfitting_gap * 100, t_loss, v_loss,
            )

        # Preview generation (reconstruction-based for VQ-VAE)
        if effective_preview_interval > 0 and epoch > 0 and epoch % effective_preview_interval == 0:
            try:
                # Get a sample batch from val_loader for reconstruction preview
                sample_batch = next(iter(val_loader))
                preview_paths = generate_vqvae_reconstruction_preview(
                    model=model,
                    spectrogram=spectrogram,
                    sample_batch=sample_batch,
                    output_dir=preview_dir,
                    epoch=epoch,
                    device=device,
                )
                for p in preview_paths:
                    if callback is not None:
                        callback(PreviewEvent(
                            epoch=epoch,
                            audio_path=str(p),
                            sample_rate=spec_config.sample_rate,
                        ))
            except Exception:
                logger.warning("Preview generation failed at epoch %d", epoch, exc_info=True)

        # Checkpoint saving
        if config.checkpoint_interval > 0 and epoch > 0 and epoch % config.checkpoint_interval == 0:
            _save_vqvae_checkpoint_safe(
                checkpoint_dir, model, optimizer, scheduler, epoch, 0,
                t_loss, v_loss, config, spec_config, metrics_history,
                codebook_health=codebook_health,
            )
            try:
                manage_checkpoints(checkpoint_dir, max_recent=config.max_checkpoints)
            except Exception:
                logger.warning("Checkpoint management failed at epoch %d", epoch, exc_info=True)

        # Cancel check at end of epoch
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Cancel detected at end of epoch %d, saving checkpoint", epoch)
            _save_vqvae_checkpoint_safe(
                checkpoint_dir, model, optimizer, scheduler, epoch, 0,
                t_loss, v_loss, config, spec_config, metrics_history,
                codebook_health=codebook_health,
            )
            if callback is not None:
                callback(TrainingCompleteEvent(
                    total_epochs=epoch + 1,
                    total_time_s=elapsed,
                    final_train_loss=t_loss,
                    final_val_loss=v_loss,
                    best_val_loss=_get_best_vq_val_loss(metrics_history),
                    best_epoch=metrics_history.get_best_epoch(),
                ))
            return {
                "model": model,
                "metrics_history": metrics_history,
                "output_dir": output_dir,
                "best_checkpoint_path": get_best_checkpoint(checkpoint_dir),
                "final_codebook_health": codebook_health,
            }

    # -----------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------
    elapsed = time.time() - train_start_time

    # Final codebook health snapshot
    final_codebook_health = None
    if metrics_history.epoch_metrics:
        final_codebook_health = metrics_history.epoch_metrics[-1].codebook_health

    # Save final checkpoint
    final_epoch = config.max_epochs - 1
    final_train = metrics_history.epoch_metrics[-1].train_loss if metrics_history.epoch_metrics else 0.0
    final_val = metrics_history.epoch_metrics[-1].val_loss if metrics_history.epoch_metrics else 0.0

    _save_vqvae_checkpoint_safe(
        checkpoint_dir, model, optimizer, scheduler, final_epoch, 0,
        final_train, final_val, config, spec_config, metrics_history,
        codebook_health=final_codebook_health,
    )
    try:
        manage_checkpoints(checkpoint_dir, max_recent=config.max_checkpoints)
    except Exception:
        logger.warning("Final checkpoint management failed", exc_info=True)

    # Save as .distill v2 model to the library
    if models_dir is not None:
        try:
            from distill.models.persistence import ModelMetadata, save_model_v2

            metadata = ModelMetadata(
                name=model_name or dataset_name or "Untitled VQ-VAE Model",
                dataset_name=dataset_name,
                dataset_file_count=len(file_paths),
                training_epochs=config.max_epochs,
                final_train_loss=final_train,
                final_val_loss=final_val,
            )
            loss_curves = metrics_history.get_loss_curves()
            saved_model_path = save_model_v2(
                model=model,
                spectrogram_config=asdict(spec_config),
                vqvae_config=asdict(config),
                training_config=asdict(config),
                metadata=metadata,
                models_dir=Path(models_dir),
                codebook_health=final_codebook_health,
                loss_curve_history=loss_curves,
            )
            print(f"[VQ-TRAIN] Model saved to library: {saved_model_path}", flush=True)
        except Exception as exc:
            print(f"[VQ-TRAIN] Failed to save model to library: {exc}", flush=True)
            logger.warning("Failed to save model to library", exc_info=True)

    # Emit completion event
    if callback is not None:
        callback(TrainingCompleteEvent(
            total_epochs=config.max_epochs,
            total_time_s=elapsed,
            final_train_loss=final_train,
            final_val_loss=final_val,
            best_val_loss=_get_best_vq_val_loss(metrics_history),
            best_epoch=metrics_history.get_best_epoch(),
        ))

    best_ckpt = get_best_checkpoint(checkpoint_dir)
    logger.info(
        "VQ-VAE training complete: %d epochs in %.1fs (best val_loss=%.4f at epoch %d)",
        config.max_epochs, elapsed,
        _get_best_vq_val_loss(metrics_history),
        metrics_history.get_best_epoch(),
    )

    return {
        "model": model,
        "metrics_history": metrics_history,
        "output_dir": output_dir,
        "best_checkpoint_path": best_ckpt,
        "final_codebook_health": final_codebook_health,
    }


def _save_vqvae_checkpoint_safe(
    checkpoint_dir: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler",
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    config: "VQVAEConfig",
    spec_config: "SpectrogramConfig",
    metrics_history: "VQMetricsHistory",
    codebook_health: dict | None = None,
) -> None:
    """Save a VQ-VAE checkpoint, catching and logging any errors."""
    from distill.training.checkpoint import save_vqvae_checkpoint

    try:
        path = checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        save_vqvae_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            commitment_weight=config.commitment_weight,
            training_config=asdict(config),
            spectrogram_config=asdict(spec_config),
            metrics_history_dict=metrics_history.to_dict(),
            codebook_health=codebook_health,
        )
    except Exception:
        logger.warning("Failed to save VQ-VAE checkpoint at epoch %d", epoch, exc_info=True)
