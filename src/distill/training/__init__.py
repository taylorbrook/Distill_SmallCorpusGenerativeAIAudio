"""Training loop and optimization.

Public API re-exports from all training submodules:
:mod:`config`, :mod:`dataset`, :mod:`metrics`, :mod:`checkpoint`,
:mod:`preview`, :mod:`loop`, and :mod:`runner`.
"""

# ---------------------------------------------------------------------------
# v1.0 exports
# ---------------------------------------------------------------------------

from distill.training.checkpoint import (
    get_best_checkpoint,
    list_checkpoints,
    load_checkpoint,
    manage_checkpoints,
    save_checkpoint,
)
from distill.training.config import (
    OverfittingPreset,
    RegularizationConfig,
    TrainingConfig,
    get_adaptive_config,
    get_effective_preview_interval,
)
from distill.training.dataset import (
    AudioTrainingDataset,
    create_data_loaders,
)
from distill.training.loop import train
from distill.training.metrics import (
    EpochMetrics,
    MetricsCallback,
    MetricsHistory,
    PreviewEvent,
    StepMetrics,
    TrainingCompleteEvent,
)
from distill.training.preview import (
    generate_preview,
    generate_reconstruction_preview,
    list_previews,
)
from distill.training.runner import TrainingRunner

# ---------------------------------------------------------------------------
# v1.1 exports (VQ-VAE training) -- added in Phase 13
# ---------------------------------------------------------------------------

from distill.training.checkpoint import (
    load_vqvae_checkpoint,
    save_vqvae_checkpoint,
)
from distill.training.config import VQVAEConfig, get_adaptive_vqvae_config
from distill.training.loop import train_vqvae, train_vqvae_epoch, validate_vqvae_epoch
from distill.training.metrics import VQEpochMetrics, VQMetricsHistory, VQStepMetrics
from distill.training.preview import generate_vqvae_reconstruction_preview

# ---------------------------------------------------------------------------
# v1.1 Prior training exports -- added in Phase 14
# ---------------------------------------------------------------------------

from distill.training.metrics import (
    PriorEpochMetrics,
    PriorStepMetrics,
    PriorTrainingCompleteEvent,
)
from distill.training.prior_config import PriorConfig, get_adaptive_prior_config
from distill.training.prior_loop import (
    check_memorization,
    train_prior,
    train_prior_epoch,
    validate_prior_epoch,
)

__all__ = [
    # config.py (v1.0)
    "TrainingConfig",
    "OverfittingPreset",
    "RegularizationConfig",
    "get_adaptive_config",
    "get_effective_preview_interval",
    # config.py (v1.1)
    "VQVAEConfig",
    "get_adaptive_vqvae_config",
    # dataset.py
    "AudioTrainingDataset",
    "create_data_loaders",
    # metrics.py (v1.0)
    "StepMetrics",
    "EpochMetrics",
    "PreviewEvent",
    "TrainingCompleteEvent",
    "MetricsHistory",
    "MetricsCallback",
    # metrics.py (v1.1)
    "VQStepMetrics",
    "VQEpochMetrics",
    "VQMetricsHistory",
    # checkpoint.py (v1.0)
    "save_checkpoint",
    "load_checkpoint",
    "manage_checkpoints",
    "get_best_checkpoint",
    "list_checkpoints",
    # checkpoint.py (v1.1)
    "save_vqvae_checkpoint",
    "load_vqvae_checkpoint",
    # preview.py (v1.0)
    "generate_preview",
    "generate_reconstruction_preview",
    "list_previews",
    # preview.py (v1.1)
    "generate_vqvae_reconstruction_preview",
    # loop.py (v1.0)
    "train",
    # loop.py (v1.1)
    "train_vqvae",
    "train_vqvae_epoch",
    "validate_vqvae_epoch",
    # runner.py
    "TrainingRunner",
    # prior_config.py (v1.1)
    "PriorConfig",
    "get_adaptive_prior_config",
    # prior_loop.py (v1.1)
    "train_prior",
    "train_prior_epoch",
    "validate_prior_epoch",
    "check_memorization",
    # metrics.py (v1.1 prior)
    "PriorStepMetrics",
    "PriorEpochMetrics",
    "PriorTrainingCompleteEvent",
]
