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
        Fraction of total epochs over which KL weight ramps from 0 to kl_weight_max.
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
    kl_weight_max: float = 0.01
    free_bits: float = 0.1
    val_fraction: float = 0.2
    chunk_duration_s: float = 1.0
    checkpoint_interval: int = 10
    preview_interval: int = 20
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
        free_bits=0.1,
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


# ---------------------------------------------------------------------------
# VQ-VAE Configuration
# ---------------------------------------------------------------------------


@dataclass
class VQVAEConfig:
    """VQ-VAE model and training configuration.

    Separate from :class:`TrainingConfig` (which drives the v1.0 continuous
    VAE).  All VQ-specific parameters live here: codebook geometry, quantizer
    behaviour, and training hyper-parameters tuned for discrete codes.

    The companion :func:`get_adaptive_vqvae_config` auto-scales these fields
    based on dataset size.

    Attributes
    ----------
    codebook_dim:
        Dimensionality of each codebook embedding vector.
    codebook_size:
        Number of entries per codebook.  Auto-scaled by dataset size.
    num_quantizers:
        Number of residual quantization levels (2-4).
    decay:
        EMA decay for codebook updates (lower = faster adaptation).
    commitment_weight:
        Weight applied to commitment loss (encoder-to-codebook alignment).
    threshold_ema_dead_code:
        Minimum EMA usage count; codes below this are replaced.
    kmeans_init:
        Whether to initialise codebooks with k-means on the first batch.
    kmeans_iters:
        Number of k-means iterations for initialisation.
    dropout:
        Dropout probability in encoder/decoder layers.
    gradient_clip_norm:
        Maximum gradient L2 norm for clipping (0 = disabled).
    batch_size:
        Mini-batch size for DataLoader.
    max_epochs:
        Maximum training epochs (early stopping may halt sooner).
    learning_rate:
        Initial learning rate for AdamW optimizer.
    weight_decay:
        AdamW weight decay coefficient.
    chunk_duration_s:
        Duration (seconds) of fixed-length audio chunks for training.
    val_fraction:
        Fraction of files held out for validation.
    augmentation_expansion:
        Number of augmented copies per original file.
    checkpoint_interval:
        Save checkpoint every N epochs.
    preview_interval:
        Generate audio preview every N epochs.
    max_checkpoints:
        Number of most-recent checkpoints to retain (+ 1 best).
    device:
        Target device: ``"auto"`` resolves via hardware detection,
        or explicit ``"cpu"`` / ``"cuda"`` / ``"mps"``.
    num_workers:
        DataLoader worker processes.  0 = main process (safest cross-platform).
    """

    # Model architecture
    codebook_dim: int = 128
    codebook_size: int = 256
    num_quantizers: int = 3

    # Quantizer (vector-quantize-pytorch params)
    decay: float = 0.95
    commitment_weight: float = 0.25
    threshold_ema_dead_code: int = 2
    kmeans_init: bool = True
    kmeans_iters: int = 10

    # Regularization
    dropout: float = 0.2
    gradient_clip_norm: float = 1.0

    # Training
    batch_size: int = 32
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.01

    # Dataset
    chunk_duration_s: float = 1.0
    val_fraction: float = 0.2
    augmentation_expansion: int = 10

    # Checkpoint/preview
    checkpoint_interval: int = 10
    preview_interval: int = 20
    max_checkpoints: int = 3

    # Device
    device: str = "auto"
    num_workers: int = 0


def get_adaptive_vqvae_config(file_count: int) -> VQVAEConfig:
    """Build a :class:`VQVAEConfig` adapted to the dataset size.

    Three tiers scale codebook size, regularization, and training length
    to the amount of available data:

    ======== ============= ===== ======= ======= =============
    Files    codebook_size decay dropout epochs  learning_rate
    ======== ============= ===== ======= ======= =============
    <= 20    64            0.8   0.4     100     5e-4
    21-100   128           0.9   0.2     200     1e-3
    > 100    256           0.95  0.1     300     1e-3
    ======== ============= ===== ======= ======= =============

    All tiers share: ``num_quantizers=3``, ``codebook_dim=128``,
    ``commitment_weight=0.25``, ``threshold_ema_dead_code=2``,
    ``kmeans_init=True``, ``kmeans_iters=10``.

    ``batch_size`` is capped at ``min(32, max(1, file_count * 5 // 4))``
    to avoid excessive gradient noise on tiny datasets.

    ``val_fraction`` scales inversely with dataset size: 0.5 for < 10,
    0.3 for < 50, 0.2 for < 200, 0.1 for >= 200.

    Parameters
    ----------
    file_count:
        Number of audio files in the dataset.

    Returns
    -------
    VQVAEConfig
        Fully configured for the given dataset size.
    """
    # 1. Tier selection
    if file_count <= 20:
        codebook_size = 64
        decay = 0.8
        dropout = 0.4
        max_epochs = 100
        learning_rate = 5e-4
        augmentation_expansion = 15
    elif file_count <= 100:
        codebook_size = 128
        decay = 0.9
        dropout = 0.2
        max_epochs = 200
        learning_rate = 1e-3
        augmentation_expansion = 10
    else:  # > 100 (up to 500)
        codebook_size = 256
        decay = 0.95
        dropout = 0.1
        max_epochs = 300
        learning_rate = 1e-3
        augmentation_expansion = 5

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
    batch_size = min(32, max(1, file_count * 5 // 4))

    return VQVAEConfig(
        codebook_dim=128,
        codebook_size=codebook_size,
        num_quantizers=3,
        decay=decay,
        commitment_weight=0.25,
        threshold_ema_dead_code=2,
        kmeans_init=True,
        kmeans_iters=10,
        dropout=dropout,
        gradient_clip_norm=1.0,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        chunk_duration_s=1.0,
        val_fraction=val_fraction,
        augmentation_expansion=augmentation_expansion,
    )
