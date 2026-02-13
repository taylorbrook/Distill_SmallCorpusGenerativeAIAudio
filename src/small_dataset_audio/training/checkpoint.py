"""Checkpoint management for VAE training.

Saves and loads full training state (model, optimizer, scheduler, epoch,
step, KL weight, config, and metrics history) for exact resume.  A JSON
sidecar file alongside each ``.pt`` checkpoint stores lightweight metadata
(epoch, losses) so ``manage_checkpoints`` can select the best checkpoint
without loading multi-megabyte state dicts.

Retention policy: keep the 3 most recent checkpoints plus the 1 checkpoint
with lowest validation loss (4 max).  If the best checkpoint is already
among the 3 most recent, the total count is 3.

Design notes:
- Lazy torch import (project pattern).
- All functions accept ``Path`` objects and create directories as needed.
- Errors are logged as warnings, never raised, to avoid crashing training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1

# Naming conventions
_CKPT_PATTERN = "checkpoint_epoch*.pt"
_META_SUFFIX = ".meta.json"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer",
    scheduler: "torch.optim.lr_scheduler.LRScheduler",
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    kl_weight: float,
    training_config: dict,
    spectrogram_config: dict,
    metrics_history_dict: dict,
) -> Path:
    """Save a full training checkpoint and lightweight JSON sidecar.

    Parameters
    ----------
    path : Path
        Destination file path (e.g. ``checkpoints/checkpoint_epoch0010.pt``).
    model : torch.nn.Module
        The VAE model.
    optimizer : torch.optim.Optimizer
        Optimiser to save state from.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning-rate scheduler to save state from.
    epoch : int
        Current epoch number.
    step : int
        Current global step count.
    train_loss : float
        Training loss at this checkpoint.
    val_loss : float
        Validation loss at this checkpoint.
    kl_weight : float
        Current KL annealing weight.
    training_config : dict
        Training configuration as a plain dict.
    spectrogram_config : dict
        Spectrogram configuration as a plain dict.
    metrics_history_dict : dict
        Full metrics history from ``MetricsHistory.to_dict()``.

    Returns
    -------
    Path
        The path written.
    """
    import torch  # noqa: WPS433 -- lazy import

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "kl_weight": kl_weight,
        "training_config": training_config,
        "spectrogram_config": spectrogram_config,
        "metrics_history": metrics_history_dict,
    }

    torch.save(checkpoint, path)

    # Write lightweight JSON sidecar for fast scanning
    meta_path = path.with_suffix(".pt" + _META_SUFFIX)
    meta = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
    }
    try:
        meta_path.write_text(json.dumps(meta, indent=2))
    except OSError:
        logger.warning("Failed to write checkpoint sidecar: %s", meta_path)

    logger.info("Saved checkpoint: %s (epoch %d, val_loss=%.4f)", path.name, epoch, val_loss)
    return path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: Path,
    model: "torch.nn.Module",
    optimizer: "torch.optim.Optimizer | None" = None,
    scheduler: "torch.optim.lr_scheduler.LRScheduler | None" = None,
    device: str = "cpu",
) -> dict:
    """Load a checkpoint and restore model (and optionally optimizer/scheduler) state.

    Parameters
    ----------
    path : Path
        Path to the ``.pt`` checkpoint file.
    model : torch.nn.Module
        Model to restore weights into.
    optimizer : torch.optim.Optimizer | None
        If provided, restores optimizer state.
    scheduler : torch.optim.lr_scheduler.LRScheduler | None
        If provided, restores scheduler state.
    device : str
        Device to map tensors to (default ``"cpu"``).

    Returns
    -------
    dict
        The full checkpoint dict.  Caller reads ``epoch``, ``step``,
        ``kl_weight``, ``metrics_history``, ``training_config``, etc.

    Raises
    ------
    ValueError
        If the checkpoint version is unsupported.
    """
    import torch  # noqa: WPS433 -- lazy import

    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Version compatibility check
    ckpt_version = checkpoint.get("version", 0)
    if ckpt_version > CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {ckpt_version} is newer than supported "
            f"version {CHECKPOINT_VERSION}. Update the software to load this checkpoint."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        "Loaded checkpoint: %s (epoch %d, val_loss=%.4f)",
        path.name,
        checkpoint.get("epoch", -1),
        checkpoint.get("val_loss", float("inf")),
    )
    return checkpoint


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _read_meta(pt_path: Path) -> dict | None:
    """Read the JSON sidecar for a checkpoint, or return None on failure."""
    meta_path = pt_path.with_suffix(".pt" + _META_SUFFIX)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt sidecar file: %s", meta_path)
        return None


def _epoch_from_name(pt_path: Path) -> int:
    """Extract epoch number from ``checkpoint_epochNNNN.pt``."""
    stem = pt_path.stem  # checkpoint_epoch0042
    try:
        return int(stem.split("epoch")[1])
    except (IndexError, ValueError):
        return -1


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def list_checkpoints(checkpoint_dir: Path) -> list[dict]:
    """List all checkpoints in a directory with metadata.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing ``.pt`` checkpoint files.

    Returns
    -------
    list[dict]
        Sorted by epoch: ``[{path, epoch, train_loss, val_loss}, ...]``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    results: list[dict] = []
    for pt_path in sorted(checkpoint_dir.glob(_CKPT_PATTERN)):
        meta = _read_meta(pt_path)
        if meta is None:
            # Fallback: extract what we can from filename
            epoch = _epoch_from_name(pt_path)
            meta = {"epoch": epoch, "train_loss": float("inf"), "val_loss": float("inf")}

        results.append({
            "path": pt_path,
            "epoch": meta["epoch"],
            "train_loss": meta.get("train_loss", float("inf")),
            "val_loss": meta.get("val_loss", float("inf")),
        })

    results.sort(key=lambda c: c["epoch"])
    return results


# ---------------------------------------------------------------------------
# Best checkpoint
# ---------------------------------------------------------------------------


def get_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the path to the checkpoint with lowest validation loss.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoint files.

    Returns
    -------
    Path | None
        Path to the best checkpoint, or ``None`` if no checkpoints exist.
    """
    ckpts = list_checkpoints(checkpoint_dir)
    if not ckpts:
        return None
    best = min(ckpts, key=lambda c: c["val_loss"])
    return best["path"]


# ---------------------------------------------------------------------------
# Retention management
# ---------------------------------------------------------------------------


def manage_checkpoints(
    checkpoint_dir: Path,
    max_recent: int = 3,
) -> list[Path]:
    """Apply retention policy: keep ``max_recent`` most recent + 1 best.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoint files.
    max_recent : int
        Number of most-recent checkpoints to keep (default 3).

    Returns
    -------
    list[Path]
        List of deleted checkpoint paths.
    """
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = list_checkpoints(checkpoint_dir)
    if len(ckpts) <= max_recent:
        return []

    # Identify keepers: max_recent most recent (by epoch) + best val_loss
    recent = {c["path"] for c in ckpts[-max_recent:]}
    best_path = get_best_checkpoint(checkpoint_dir)
    keep = recent
    if best_path is not None:
        keep = keep | {best_path}

    # Delete the rest
    deleted: list[Path] = []
    for ckpt in ckpts:
        if ckpt["path"] not in keep:
            try:
                ckpt["path"].unlink()
                # Also remove sidecar
                meta_path = ckpt["path"].with_suffix(".pt" + _META_SUFFIX)
                if meta_path.exists():
                    meta_path.unlink()
                deleted.append(ckpt["path"])
                logger.info("Deleted checkpoint: %s", ckpt["path"].name)
            except OSError:
                logger.warning("Failed to delete checkpoint: %s", ckpt["path"])

    return deleted
