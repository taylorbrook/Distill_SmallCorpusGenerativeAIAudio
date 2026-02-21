"""Training loop and optimization.

Public API re-exports from all training submodules:
:mod:`config`, :mod:`dataset`, :mod:`metrics`, :mod:`checkpoint`,
:mod:`preview`, :mod:`loop`, and :mod:`runner`.
"""

from distill.training.checkpoint import (
    get_best_checkpoint,
    list_checkpoints,
    load_checkpoint,
    manage_checkpoints,
    save_checkpoint,
)
from distill.training.config import (
    ComplexSpectrogramConfig,
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

__all__ = [
    # config.py
    "ComplexSpectrogramConfig",
    "TrainingConfig",
    "OverfittingPreset",
    "RegularizationConfig",
    "get_adaptive_config",
    "get_effective_preview_interval",
    # dataset.py
    "AudioTrainingDataset",
    "create_data_loaders",
    # metrics.py
    "StepMetrics",
    "EpochMetrics",
    "PreviewEvent",
    "TrainingCompleteEvent",
    "MetricsHistory",
    "MetricsCallback",
    # checkpoint.py
    "save_checkpoint",
    "load_checkpoint",
    "manage_checkpoints",
    "get_best_checkpoint",
    "list_checkpoints",
    # preview.py
    "generate_preview",
    "generate_reconstruction_preview",
    "list_previews",
    # loop.py
    "train",
    # runner.py
    "TrainingRunner",
]
