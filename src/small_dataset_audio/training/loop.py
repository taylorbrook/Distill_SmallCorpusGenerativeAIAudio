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

    from small_dataset_audio.audio.spectrogram import AudioSpectrogram
    from small_dataset_audio.training.config import TrainingConfig
    from small_dataset_audio.training.metrics import MetricsCallback

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
    free_bits: float = 0.5,
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

    from small_dataset_audio.models.losses import vae_loss
    from small_dataset_audio.training.metrics import StepMetrics

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
    free_bits: float = 0.5,
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

    from small_dataset_audio.models.losses import compute_kl_divergence, vae_loss

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

    from small_dataset_audio.audio.spectrogram import AudioSpectrogram, SpectrogramConfig
    from small_dataset_audio.models.losses import get_kl_weight
    from small_dataset_audio.models.vae import ConvVAE
    from small_dataset_audio.training.checkpoint import (
        get_best_checkpoint,
        load_checkpoint,
        manage_checkpoints,
        save_checkpoint,
    )
    from small_dataset_audio.training.config import get_effective_preview_interval
    from small_dataset_audio.training.dataset import create_data_loaders
    from small_dataset_audio.training.metrics import (
        EpochMetrics,
        MetricsHistory,
        PreviewEvent,
        TrainingCompleteEvent,
    )
    from small_dataset_audio.training.preview import generate_preview

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
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
            from small_dataset_audio.audio.augmentation import (
                AugmentationConfig,
                AugmentationPipeline,
            )
            aug_config = AugmentationConfig(
                expansion_ratio=config.regularization.augmentation_expansion,
            )
            augmentation_pipeline = AugmentationPipeline(config=aug_config)
        except Exception:
            logger.warning("Failed to create augmentation pipeline", exc_info=True)

    train_loader, val_loader = create_data_loaders(
        file_paths, config, augmentation_pipeline,
    )

    logger.info(
        "Training: %d train chunks, %d val chunks, batch_size=%d",
        len(train_loader.dataset), len(val_loader.dataset), config.batch_size,
    )

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    effective_preview_interval = get_effective_preview_interval(config)
    train_start_time = time.time()

    for epoch in range(start_epoch, config.max_epochs):
        epoch_start = time.time()

        # KL weight (annealing)
        kl_weight = get_kl_weight(epoch, config.max_epochs, config.kl_warmup_fraction)

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
    final_kl = get_kl_weight(final_epoch, config.max_epochs, config.kl_warmup_fraction)

    _save_checkpoint_safe(
        checkpoint_dir, model, optimizer, scheduler, final_epoch, 0,
        final_train, final_val, final_kl, config, spec_config, metrics_history,
    )
    try:
        manage_checkpoints(checkpoint_dir, max_recent=config.max_checkpoints)
    except Exception:
        logger.warning("Final checkpoint management failed", exc_info=True)

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
) -> None:
    """Save a checkpoint, catching and logging any errors."""
    from dataclasses import asdict as _asdict

    from small_dataset_audio.training.checkpoint import save_checkpoint

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
        )
    except Exception:
        logger.warning("Failed to save checkpoint at epoch %d", epoch, exc_info=True)


def _get_best_val_loss(metrics_history: "MetricsHistory") -> float:
    """Return the best validation loss from history."""
    if not metrics_history.epoch_metrics:
        return float("inf")
    return min(m.val_loss for m in metrics_history.epoch_metrics)
