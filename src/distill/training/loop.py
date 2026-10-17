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
    from distill.controls.analyzer import AnalysisResult
    from distill.models.persistence import ModelMetadata
    from distill.training.config import TrainingConfig
    from distill.training.metrics import MetricsCallback, MetricsHistory

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
        If set, training stops and saves a checkpoint.  If at least one
        epoch has completed, the run is finalised like a normal one
        (latent analysis + library save) so the work is not lost.
    resume_checkpoint:
        Path to a ``.pt`` checkpoint file to resume from.  The weights,
        optimiser moments and epoch counter come from the checkpoint; the
        learning rate, weight decay and epoch count come from *config*.
        An epoch that was cancelled part-way is repeated.  Raises if the
        checkpoint cannot be loaded.  Without it, ``output_dir`` must not
        already hold checkpoints (each run needs its own directory).
    models_dir:
        If given, the finished model (with its latent analysis) is saved
        to this library directory as a ``.distill`` file.
    dataset_name, model_name:
        Used for the saved model's metadata.

    Returns
    -------
    dict
        ``{model, metrics_history, output_dir, best_checkpoint_path,
        final_checkpoint_path, analysis, saved_model_path,
        model_metadata}``.  ``final_checkpoint_path`` is the checkpoint
        holding the returned weights *and* their latent analysis;
        ``saved_model_path`` / ``model_metadata`` are ``None`` when no
        library save happened.
    """
    import torch  # noqa: WPS433

    from distill.audio.spectrogram import AudioSpectrogram, SpectrogramConfig
    from distill.models.losses import get_kl_weight
    from distill.models.vae import ConvVAE
    from distill.training.checkpoint import (
        list_checkpoints,
        load_checkpoint,
        manage_checkpoints,
    )
    from distill.training.config import get_effective_preview_interval
    from distill.training.dataset import create_data_loaders
    from distill.training.metrics import (
        EpochMetrics,
        MetricsHistory,
        PreviewEvent,
    )
    from distill.training.preview import generate_preview

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    print(f"[TRAIN] train() called: {len(file_paths)} files, device={device}", flush=True)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    if resume_checkpoint is None and list_checkpoints(checkpoint_dir):
        # Retention and "best" selection work per directory, so a second
        # run here would delete or be confused with the first run's files.
        raise ValueError(
            f"{checkpoint_dir} already holds checkpoints from another run. "
            "Resume that run or use a new output directory."
        )
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

    if resume_checkpoint is not None:
        try:
            ckpt = load_checkpoint(
                resume_checkpoint, model, optimizer, scheduler,
                device=str(device),
            )
        except Exception as exc:
            # Never fall back to a fresh run: it would train from scratch
            # into the interrupted run's checkpoint directory.
            raise RuntimeError(
                f"Could not resume from checkpoint {resume_checkpoint}: {exc}"
            ) from exc

        start_epoch = _resume_start_epoch(ckpt)
        if start_epoch >= config.max_epochs:
            raise ValueError(
                f"The checkpoint already covers {start_epoch} epochs; "
                f"set max epochs above {start_epoch} to continue training."
            )
        _apply_config_to_resumed_optimizer(optimizer, scheduler, config)

        # Restore metrics history
        if "metrics_history" in ckpt:
            metrics_history = MetricsHistory.from_dict(ckpt["metrics_history"])

        logger.info(
            "Resumed training at epoch %d (lr=%.2e)",
            start_epoch, optimizer.param_groups[0]["lr"],
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

    # Shared arguments for _finalize_run (completion and cancellation)
    finalize_kwargs = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "config": config,
        "spec_config": spec_config,
        "spectrogram": spectrogram,
        "metrics_history": metrics_history,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "file_paths": file_paths,
        "device": device,
        "models_dir": models_dir,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "callback": callback,
    }

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
            return _finalize_run(
                **finalize_kwargs,
                final_epoch=epoch,
                final_train=train_results["train_loss"],
                final_val=float("inf"),
                final_kl=kl_weight,
                epochs_completed=epoch + 1,
                total_time_s=time.time() - train_start_time,
                # Only keep a cancelled run if it got through an epoch
                run_analysis=bool(metrics_history.epoch_metrics),
                # Validation and the scheduler step never ran for this
                # epoch, so a resume must repeat it
                epoch_complete=False,
            )

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
            return _finalize_run(
                **finalize_kwargs,
                final_epoch=epoch,
                final_train=t_loss,
                final_val=v_loss,
                final_kl=kl_weight,
                epochs_completed=epoch + 1,
                total_time_s=elapsed,
            )

    # -----------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------
    final_epoch = config.max_epochs - 1
    final_train = metrics_history.epoch_metrics[-1].train_loss if metrics_history.epoch_metrics else 0.0
    final_val = metrics_history.epoch_metrics[-1].val_loss if metrics_history.epoch_metrics else 0.0
    final_kl = get_kl_weight(
        final_epoch, config.max_epochs, config.kl_warmup_fraction,
        kl_weight_max=config.kl_weight_max,
    )

    result = _finalize_run(
        **finalize_kwargs,
        final_epoch=final_epoch,
        final_train=final_train,
        final_val=final_val,
        final_kl=final_kl,
        epochs_completed=config.max_epochs,
        total_time_s=time.time() - train_start_time,
    )
    logger.info(
        "Training complete: %d epochs (best val_loss=%.4f at epoch %d)",
        config.max_epochs,
        _get_best_val_loss(metrics_history),
        metrics_history.get_best_epoch(),
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _finalize_run(
    *,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler",
    config: "TrainingConfig",
    spec_config: "SpectrogramConfig",
    spectrogram: "AudioSpectrogram",
    metrics_history: "MetricsHistory",
    checkpoint_dir: Path,
    output_dir: Path,
    file_paths: list[Path],
    device: "torch.device",
    models_dir: "Path | None",
    dataset_name: str,
    model_name: str,
    callback: "MetricsCallback | None",
    final_epoch: int,
    final_train: float,
    final_val: float,
    final_kl: float,
    epochs_completed: int,
    total_time_s: float,
    run_analysis: bool = True,
    epoch_complete: bool = True,
) -> dict:
    """Save the final checkpoint, analyse, save to the library, and report.

    Shared by normal completion and cancellation so both leave behind the
    same artefacts: a checkpoint whose ``latent_analysis`` was computed on
    the weights it holds, and (when ``models_dir`` is set) a ``.distill``
    model built from those same weights.
    """
    from distill.training.checkpoint import get_best_checkpoint, manage_checkpoints
    from distill.training.metrics import TrainingCompleteEvent

    ckpt_args = (
        checkpoint_dir, model, optimizer, scheduler, final_epoch, 0,
        final_train, final_val, final_kl, config, spec_config, metrics_history,
    )
    _save_checkpoint_safe(*ckpt_args, epoch_complete=epoch_complete)
    try:
        manage_checkpoints(checkpoint_dir, max_recent=config.max_checkpoints)
    except Exception:
        logger.warning("Final checkpoint management failed", exc_info=True)

    analysis_result = None
    saved_model_path = None
    saved_metadata = None
    if run_analysis:
        analysis_result = _run_latent_analysis(
            model, file_paths, config, spectrogram, device,
        )
        if analysis_result is not None:
            from distill.controls.serialization import analysis_to_dict

            # Re-save the final checkpoint WITH analysis included
            _save_checkpoint_safe(
                *ckpt_args,
                latent_analysis=analysis_to_dict(analysis_result),
                epoch_complete=epoch_complete,
            )

        if models_dir is not None:
            saved_model_path, saved_metadata = _save_to_library(
                model=model,
                config=config,
                spec_config=spec_config,
                analysis=analysis_result,
                models_dir=models_dir,
                name=model_name or dataset_name or "Untitled Model",
                dataset_name=dataset_name,
                file_count=len(file_paths),
                epochs_completed=epochs_completed,
                final_train=final_train,
                final_val=final_val,
            )

    if callback is not None:
        callback(TrainingCompleteEvent(
            total_epochs=epochs_completed,
            total_time_s=total_time_s,
            final_train_loss=final_train,
            final_val_loss=final_val,
            best_val_loss=_get_best_val_loss(metrics_history),
            best_epoch=metrics_history.get_best_epoch(),
        ))

    final_ckpt = checkpoint_dir / f"checkpoint_epoch{final_epoch:04d}.pt"
    return {
        "model": model,
        "metrics_history": metrics_history,
        "output_dir": output_dir,
        "best_checkpoint_path": get_best_checkpoint(checkpoint_dir),
        "final_checkpoint_path": final_ckpt if final_ckpt.exists() else None,
        "analysis": analysis_result,
        "saved_model_path": saved_model_path,
        "model_metadata": saved_metadata,
    }


def _run_latent_analysis(
    model: "torch.nn.Module",
    file_paths: list[Path],
    config: "TrainingConfig",
    spectrogram: "AudioSpectrogram",
    device: "torch.device",
) -> "AnalysisResult | None":
    """Run PCA latent analysis over all files; ``None`` on failure."""
    try:
        from torch.utils.data import DataLoader as TorchDataLoader

        from distill.controls.analyzer import LatentSpaceAnalyzer
        from distill.training.dataset import AudioTrainingDataset

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
        return analysis_result
    except Exception:
        logger.warning("Latent space analysis failed -- model saved without analysis", exc_info=True)
        return None


def _save_to_library(
    *,
    model: "torch.nn.Module",
    config: "TrainingConfig",
    spec_config: "SpectrogramConfig",
    analysis: "AnalysisResult | None",
    models_dir: Path,
    name: str,
    dataset_name: str,
    file_count: int,
    epochs_completed: int,
    final_train: float,
    final_val: float,
) -> "tuple[Path | None, ModelMetadata | None]":
    """Save the model as a ``.distill`` file; ``(None, None)`` on failure."""
    from datetime import datetime, timezone

    try:
        from distill.models.persistence import ModelMetadata, save_model

        metadata = ModelMetadata(
            name=name,
            dataset_name=dataset_name,
            dataset_file_count=file_count,
            training_date=datetime.now(timezone.utc).isoformat(),
            training_epochs=epochs_completed,
            final_train_loss=final_train,
            final_val_loss=final_val,
        )
        saved_model_path = save_model(
            model=model,
            spectrogram_config=asdict(spec_config),
            training_config=asdict(config),
            metadata=metadata,
            models_dir=models_dir,
            analysis=analysis,
        )
        print(f"[TRAIN] Model saved to library: {saved_model_path}", flush=True)
        return saved_model_path, metadata
    except Exception as exc:
        print(f"[TRAIN] Failed to save model to library: {exc}", flush=True)
        logger.warning("Failed to save model to library", exc_info=True)
        return None, None


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
    epoch_complete: bool = True,
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
            epoch_complete=epoch_complete,
        )
    except Exception:
        logger.warning("Failed to save checkpoint at epoch %d", epoch, exc_info=True)


def _get_best_val_loss(metrics_history: "MetricsHistory") -> float:
    """Return the best finite validation loss from history (``inf`` if none)."""
    import math

    finite = [m.val_loss for m in metrics_history.epoch_metrics if math.isfinite(m.val_loss)]
    return min(finite, default=float("inf"))


def _resume_start_epoch(ckpt: dict) -> int:
    """First epoch to train after loading *ckpt*.

    A checkpoint written by a mid-epoch cancel holds a partly trained
    epoch, so that epoch is repeated.  Checkpoints from before the
    ``epoch_complete`` flag existed only stored ``val_loss=inf`` on that
    path.
    """
    epoch = int(ckpt.get("epoch", 0))
    complete = ckpt.get("epoch_complete", ckpt.get("val_loss") != float("inf"))
    return epoch + 1 if complete else epoch


def _apply_config_to_resumed_optimizer(
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.CosineAnnealingLR",
    config: "TrainingConfig",
) -> None:
    """Make a resumed run follow *config* rather than the checkpoint.

    ``load_state_dict`` restores the old learning rate, weight decay and
    cosine period (``T_max``).  Keep the restored step count and optimiser
    moments, but continue on the schedule the new config describes.
    """
    import math

    scheduler.T_max = config.max_epochs
    scheduler.base_lrs = [config.learning_rate] * len(optimizer.param_groups)
    # CosineAnnealingLR updates the lr recursively from its current value,
    # so set the closed-form value for the restored step explicitly.
    progress = math.pi * scheduler.last_epoch / config.max_epochs
    lr = scheduler.eta_min + (config.learning_rate - scheduler.eta_min) * (
        1 + math.cos(progress)
    ) / 2
    for group in optimizer.param_groups:
        group["initial_lr"] = config.learning_rate
        group["lr"] = lr
        group["weight_decay"] = config.regularization.weight_decay
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
