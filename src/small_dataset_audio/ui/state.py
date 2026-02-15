"""Application state singleton for the Gradio UI.

Uses module-level singleton pattern (single-user desktop app).
All heavy imports are guarded by TYPE_CHECKING to avoid importing
torch, training, inference, etc. at module load time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch

    from small_dataset_audio.data.dataset import Dataset
    from small_dataset_audio.data.summary import DatasetSummary
    from small_dataset_audio.history.comparison import ABComparison
    from small_dataset_audio.history.store import GenerationHistory
    from small_dataset_audio.inference.blending import BlendEngine
    from small_dataset_audio.inference.generation import GenerationPipeline
    from small_dataset_audio.library.catalog import ModelLibrary
    from small_dataset_audio.models.persistence import LoadedModel
    from small_dataset_audio.presets.manager import PresetManager
    from small_dataset_audio.training.runner import TrainingRunner


@dataclass
class AppState:
    """Global application state for single-user desktop app.

    Fields are initialized with safe defaults.  Call :func:`init_state`
    after loading config to populate paths and create managers.
    """

    # Paths from config (populated by init_state)
    datasets_dir: Path = field(default_factory=lambda: Path("data/datasets"))
    models_dir: Path = field(default_factory=lambda: Path("data/models"))
    generated_dir: Path = field(default_factory=lambda: Path("data/generated"))
    presets_dir: Path = field(default_factory=lambda: Path("data/presets"))
    history_dir: Path = field(default_factory=lambda: Path("data/history"))

    # Current loaded model
    loaded_model: Optional[LoadedModel] = None
    pipeline: Optional[GenerationPipeline] = None

    # Current dataset
    current_dataset: Optional[Dataset] = None
    current_summary: Optional[DatasetSummary] = None

    # Training state
    training_runner: Optional[TrainingRunner] = None
    training_active: bool = False
    metrics_buffer: dict[str, Any] = field(default_factory=dict)

    # Managers
    preset_manager: Optional[PresetManager] = None
    history_store: Optional[GenerationHistory] = None
    model_library: Optional[ModelLibrary] = None

    # A/B comparison state
    ab_comparison: Optional[ABComparison] = None

    # Multi-model blending
    blend_engine: Optional[BlendEngine] = None
    loaded_models: list = field(default_factory=list)
    """Tracking list for multiple loaded model references in blend mode."""

    # Device
    device: Optional[torch.device] = None


# Module-level singleton
app_state = AppState()


def init_state(config: dict[str, Any], device: Any) -> None:
    """Populate app_state from loaded config and selected device.

    Args:
        config: Configuration dictionary from :func:`load_config`.
        device: torch.device selected by :func:`select_device`.
    """
    from small_dataset_audio.config.settings import resolve_path
    from small_dataset_audio.library.catalog import ModelLibrary

    paths = config.get("paths", {})

    app_state.datasets_dir = resolve_path(paths.get("datasets", "data/datasets"))
    app_state.models_dir = resolve_path(paths.get("models", "data/models"))
    app_state.generated_dir = resolve_path(paths.get("generated", "data/generated"))
    app_state.presets_dir = resolve_path(paths.get("presets", "data/presets"))
    app_state.history_dir = resolve_path(paths.get("history", "data/history"))

    app_state.device = device

    # Create ModelLibrary (lightweight -- just reads JSON index)
    app_state.models_dir.mkdir(parents=True, exist_ok=True)
    app_state.model_library = ModelLibrary(app_state.models_dir)

    # Initialize metrics buffer
    reset_metrics_buffer()


def reset_metrics_buffer() -> None:
    """Clear and reinitialize the metrics buffer for training polling."""
    app_state.metrics_buffer.clear()
    app_state.metrics_buffer.update({
        "epoch_metrics": [],
        "previews": [],
        "complete": False,
    })
