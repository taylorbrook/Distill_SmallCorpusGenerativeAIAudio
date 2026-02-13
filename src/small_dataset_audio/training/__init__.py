"""Training loop and optimization.

Public API re-exports from all training submodules:
:mod:`config`, :mod:`dataset`, :mod:`metrics`, :mod:`checkpoint`,
:mod:`preview`, :mod:`loop`, and :mod:`runner`.
"""

from small_dataset_audio.training.checkpoint import (
    get_best_checkpoint,
    list_checkpoints,
    load_checkpoint,
    manage_checkpoints,
    save_checkpoint,
)
from small_dataset_audio.training.config import (
    OverfittingPreset,
    RegularizationConfig,
    TrainingConfig,
    get_adaptive_config,
    get_effective_preview_interval,
)
from small_dataset_audio.training.dataset import (
    AudioTrainingDataset,
    create_data_loaders,
)
from small_dataset_audio.training.loop import train
from small_dataset_audio.training.metrics import (
    EpochMetrics,
    MetricsCallback,
    MetricsHistory,
    PreviewEvent,
    StepMetrics,
    TrainingCompleteEvent,
)
from small_dataset_audio.training.preview import (
    generate_preview,
    generate_reconstruction_preview,
    list_previews,
)
from small_dataset_audio.training.runner import TrainingRunner

__all__ = [
    # config.py
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
