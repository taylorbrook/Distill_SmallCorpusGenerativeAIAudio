"""Training configuration with adaptive overfitting presets.

Implements a layered control system for overfitting prevention:

1. **Automatic defaults** -- ``get_adaptive_config`` selects preset and
   validation split based on dataset size.
2. **Presets** -- Conservative / Balanced / Aggressive for different
   dataset sizes (5-50 / 50-200 / 200-500 files).
3. **Advanced toggles** -- all ``RegularizationConfig`` fields are
   individually overridable for power users.

The configuration drives every other training module: dataset splitting
(``val_fraction``), data augmentation expansion, training loop epochs,
learning rate, and checkpoint retention.

Design notes:
- Pure Python -- no torch dependency (matches Phase 1 config pattern).
- ``from __future__ import annotations`` for modern type syntax.
- All fields documented for Gradio UI auto-generation in Phase 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Overfitting Presets
# ---------------------------------------------------------------------------


class OverfittingPreset(Enum):
    """Preset regularization strategies sized to dataset scale.

    ``CONSERVATIVE`` -- for 5-50 files: heavy regularization, fewer epochs.
    ``BALANCED``     -- for 50-200 files: moderate regularization (default).
    ``AGGRESSIVE``   -- for 200-500 files: light regularization, more epochs.
    """

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


# Preset parameter lookup table.
# Keys map to RegularizationConfig fields + top-level TrainingConfig fields.
_PRESET_PARAMS: dict[OverfittingPreset, dict[str, float | int]] = {
    OverfittingPreset.CONSERVATIVE: {
        "dropout": 0.4,
        "weight_decay": 0.05,
        "augmentation_expansion": 15,
        "gradient_clip_norm": 0.5,
        "max_epochs": 100,
        "learning_rate": 5e-4,
        "kl_warmup_fraction": 0.5,
    },
    OverfittingPreset.BALANCED: {
        "dropout": 0.2,
        "weight_decay": 0.01,
        "augmentation_expansion": 10,
        "gradient_clip_norm": 1.0,
        "max_epochs": 200,
        "learning_rate": 1e-3,
        "kl_warmup_fraction": 0.3,
    },
    OverfittingPreset.AGGRESSIVE: {
        "dropout": 0.1,
        "weight_decay": 0.001,
        "augmentation_expansion": 5,
        "gradient_clip_norm": 5.0,
        "max_epochs": 500,
        "learning_rate": 2e-3,
        "kl_warmup_fraction": 0.2,
    },
}


# ---------------------------------------------------------------------------
# Regularization Config
# ---------------------------------------------------------------------------


@dataclass
class RegularizationConfig:
    """Fine-grained regularization settings.

    Power users can override individual fields after ``get_adaptive_config``
    selects defaults.  All values match the ``BALANCED`` preset by default.

    Attributes
    ----------
    dropout:
        Dropout probability applied in encoder/decoder layers.
    weight_decay:
        AdamW weight decay coefficient.
    augmentation_expansion:
        Number of augmented copies per original file (passed to
        ``AugmentationConfig.expansion_ratio``).
    gradient_clip_norm:
        Maximum gradient L2 norm for clipping (0 = disabled).
    """

    dropout: float = 0.2
    weight_decay: float = 0.01
    augmentation_expansion: int = 10
    gradient_clip_norm: float = 1.0


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Top-level training configuration.

    Drives the training loop, dataset splitting, checkpoint retention,
    and preview generation.

    Attributes
    ----------
    latent_dim:
        VAE latent space dimensionality.
    batch_size:
        Mini-batch size for DataLoader.
    max_epochs:
        Maximum training epochs (early stopping may halt sooner).
    learning_rate:
        Initial learning rate for AdamW optimizer.
    kl_warmup_fraction:
        Fraction of total epochs over which KL weight ramps from 0 to 1.
    free_bits:
        Minimum KL per latent dimension (prevents posterior collapse).
    val_fraction:
        Fraction of files held out for validation (overridden by
        ``get_adaptive_config`` based on dataset size).
    chunk_duration_s:
        Duration (seconds) of fixed-length audio chunks for training.
    checkpoint_interval:
        Save checkpoint every N epochs.
    preview_interval:
        Generate audio preview every N epochs (standard runs).
    preview_interval_short:
        Generate audio preview every N epochs (runs < 50 epochs).
    max_checkpoints:
        Number of most-recent checkpoints to retain (+ 1 best).
    preset:
        Active overfitting preset.
    regularization:
        Detailed regularization settings.
    device:
        Target device: ``"auto"`` resolves via Phase 1 hardware detection,
        or explicit ``"cpu"`` / ``"cuda"`` / ``"mps"``.
    num_workers:
        DataLoader worker processes.  0 = main process (safest cross-platform).
    """

    latent_dim: int = 64
    batch_size: int = 32
    max_epochs: int = 200
    learning_rate: float = 1e-3
    kl_warmup_fraction: float = 0.3
    free_bits: float = 0.5
    val_fraction: float = 0.2
    chunk_duration_s: float = 1.0
    checkpoint_interval: int = 10
    preview_interval: int = 5
    preview_interval_short: int = 2
    max_checkpoints: int = 3
    preset: OverfittingPreset = OverfittingPreset.BALANCED
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    device: str = "auto"
    num_workers: int = 0


# ---------------------------------------------------------------------------
# Adaptive Configuration
# ---------------------------------------------------------------------------


def get_adaptive_config(file_count: int) -> TrainingConfig:
    """Build a :class:`TrainingConfig` adapted to the dataset size.

    Selection logic:

    ======== ============= ============ =====
    Files    Preset        val_fraction batch
    ======== ============= ============ =====
    < 10     CONSERVATIVE  0.5          auto
    10-49    CONSERVATIVE  0.3          auto
    50-199   BALANCED      0.2          auto
    >= 200   AGGRESSIVE    0.1          auto
    ======== ============= ============ =====

    ``batch_size`` is capped at ``min(32, estimated_chunks // 4)`` to
    avoid excessive gradient noise on tiny datasets.

    Parameters
    ----------
    file_count:
        Number of audio files in the dataset.

    Returns
    -------
    TrainingConfig
        Fully configured for the given dataset size.
    """
    # 1. Select preset
    if file_count < 50:
        preset = OverfittingPreset.CONSERVATIVE
    elif file_count < 200:
        preset = OverfittingPreset.BALANCED
    else:
        preset = OverfittingPreset.AGGRESSIVE

    params = _PRESET_PARAMS[preset]

    # 2. Adaptive validation fraction
    if file_count < 10:
        val_fraction = 0.5
    elif file_count < 50:
        val_fraction = 0.3
    elif file_count < 200:
        val_fraction = 0.2
    else:
        val_fraction = 0.1

    # 3. Adaptive batch size
    # Estimate ~1 chunk per second of audio (conservative lower bound).
    # Average audio file ~5s -> ~5 chunks.  Actual will be higher for
    # longer files, but this keeps batches sane for tiny datasets.
    estimated_chunks = file_count * 5
    batch_size = min(32, max(1, estimated_chunks // 4))

    # 4. Build config from preset
    reg = RegularizationConfig(
        dropout=params["dropout"],  # type: ignore[arg-type]
        weight_decay=params["weight_decay"],  # type: ignore[arg-type]
        augmentation_expansion=int(params["augmentation_expansion"]),
        gradient_clip_norm=params["gradient_clip_norm"],  # type: ignore[arg-type]
    )

    return TrainingConfig(
        latent_dim=64,
        batch_size=batch_size,
        max_epochs=int(params["max_epochs"]),
        learning_rate=params["learning_rate"],  # type: ignore[arg-type]
        kl_warmup_fraction=params["kl_warmup_fraction"],  # type: ignore[arg-type]
        free_bits=0.5,
        val_fraction=val_fraction,
        chunk_duration_s=1.0,
        preset=preset,
        regularization=reg,
    )


# ---------------------------------------------------------------------------
# Preview interval helper
# ---------------------------------------------------------------------------


def get_effective_preview_interval(config: TrainingConfig) -> int:
    """Return the preview interval appropriate for this training run.

    Short runs (``max_epochs < 50``) use ``preview_interval_short`` to
    ensure the user sees at least a few previews.

    Parameters
    ----------
    config:
        Active training configuration.

    Returns
    -------
    int
        Number of epochs between audio previews.
    """
    if config.max_epochs < 50:
        return config.preview_interval_short
    return config.preview_interval
