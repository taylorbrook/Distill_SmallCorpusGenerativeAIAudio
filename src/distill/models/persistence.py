"""Model persistence -- save, load, delete, and convert trained models.

Provides the complete API for persisting trained VAE models as ``.distill``
files with rich metadata, loading them for immediate generation (with
slider controls restored), deleting models, and converting training
checkpoints to saved models.

The saved model format bundles ``model_state_dict``, ``spectrogram_config``,
``latent_analysis`` (optional), ``training_config``, and user-facing
``ModelMetadata`` into a single ``torch.save`` dict.  This is intentionally
distinct from training checkpoints (no optimizer/scheduler state).

Design notes:
- ``from __future__ import annotations`` for modern type hints.
- Lazy ``torch`` import inside function bodies (project pattern).
- ``logging.getLogger(__name__)`` for module-level logger.
- String annotations for torch/model types to avoid import-time cost.
- ``ValueError`` for data-corruption / format errors.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAVED_MODEL_VERSION = 1
MODEL_FORMAT_MARKER = "distill_model"
MODEL_FILE_EXTENSION = ".distill"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelMetadata:
    """User-facing metadata stored with a saved model.

    Fields with empty-string or zero defaults are auto-populated
    during save if not set by the caller.
    """

    model_id: str = ""
    name: str = "Untitled Model"
    description: str = ""
    dataset_name: str = ""
    dataset_file_count: int = 0
    dataset_total_duration_s: float = 0.0
    training_date: str = ""  # ISO 8601
    save_date: str = ""  # ISO 8601, auto-filled
    training_epochs: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    tags: list[str] = field(default_factory=list)
    has_analysis: bool = False
    n_active_components: int = 0


@dataclass
class LoadedModel:
    """Complete loaded model ready for generation.

    Contains everything needed to construct a
    :class:`~distill.inference.generation.GenerationPipeline`
    and (if analysis is present) restore slider controls immediately.
    """

    model: "ConvVAE"
    spectrogram: "AudioSpectrogram"
    analysis: "AnalysisResult | None"
    metadata: ModelMetadata
    device: "torch.device"
    complex_spectrogram: "ComplexSpectrogram | None" = None
    normalization_stats: dict | None = None


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Sanitize a model name for use as a filename.

    - Replace non-alphanumeric chars (except hyphen, underscore) with underscores.
    - Collapse multiple underscores.
    - Strip leading/trailing underscores.
    - Lowercase.
    - Truncate to 100 chars.
    - Fallback to ``"untitled"`` if empty after sanitization.

    Parameters
    ----------
    name : str
        Raw model name from user input.

    Returns
    -------
    str
        Safe filename stem (no extension).
    """
    # Replace non-alphanumeric (except - and _) with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Lowercase
    sanitized = sanitized.lower()
    # Truncate
    sanitized = sanitized[:100]
    # Fallback
    if not sanitized:
        sanitized = "untitled"
    return sanitized


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_model(
    model: "ConvVAE",
    spectrogram_config: dict,
    training_config: dict,
    metadata: ModelMetadata,
    models_dir: Path,
    analysis: "AnalysisResult | None" = None,
    normalization_stats: dict | None = None,
) -> Path:
    """Save a trained model with metadata to the model library.

    Creates a ``.distill`` file in *models_dir* and updates the
    :class:`~distill.library.catalog.ModelLibrary` index.

    Parameters
    ----------
    model : ConvVAE
        Trained model (weights are extracted via ``state_dict()``).
    spectrogram_config : dict
        Spectrogram configuration as a plain dict (from
        ``dataclasses.asdict(SpectrogramConfig())``).
    training_config : dict
        Training configuration as a plain dict.
    metadata : ModelMetadata
        User-facing metadata.  ``model_id`` and ``save_date`` are
        auto-generated if empty.
    models_dir : Path
        Directory to save the ``.distill`` file into.
    analysis : AnalysisResult | None
        Latent space analysis (slider mappings).  ``None`` if not
        yet analyzed.

    Returns
    -------
    Path
        Path to the saved ``.distill`` file.
    """
    import torch  # noqa: WPS433 -- lazy import

    from distill.library.catalog import ModelEntry, ModelLibrary

    # Auto-generate model_id if empty
    if not metadata.model_id:
        metadata.model_id = str(uuid.uuid4())

    # Auto-fill save_date if empty
    if not metadata.save_date:
        metadata.save_date = datetime.now(timezone.utc).isoformat()

    # Set analysis metadata from analysis object
    analysis_dict = None
    if analysis is not None:
        from distill.controls.serialization import analysis_to_dict

        analysis_dict = analysis_to_dict(analysis)
        metadata.has_analysis = True
        metadata.n_active_components = analysis.n_active_components

    # Build saved model dict (NO optimizer/scheduler state)
    saved = {
        "format": MODEL_FORMAT_MARKER,
        "version": SAVED_MODEL_VERSION,
        "model_state_dict": model.state_dict(),
        "latent_dim": model.latent_dim,
        "spectrogram_config": spectrogram_config,
        "latent_analysis": analysis_dict,
        "training_config": training_config,
        "metadata": asdict(metadata),
        "normalization_stats": normalization_stats,
    }

    # Sanitize filename from model name
    safe_name = _sanitize_filename(metadata.name)
    file_name = f"{safe_name}{MODEL_FILE_EXTENSION}"
    model_path = Path(models_dir) / file_name

    # Handle duplicate filenames with counter suffix
    counter = 1
    while model_path.exists():
        file_name = f"{safe_name}_{counter}{MODEL_FILE_EXTENSION}"
        model_path = Path(models_dir) / file_name
        counter += 1

    # Save the file
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(saved, model_path)

    # Create catalog entry and update library index
    entry = ModelEntry(
        model_id=metadata.model_id,
        name=metadata.name,
        description=metadata.description,
        file_path=model_path.name,
        file_size_bytes=os.path.getsize(model_path),
        dataset_name=metadata.dataset_name,
        dataset_file_count=metadata.dataset_file_count,
        dataset_total_duration_s=metadata.dataset_total_duration_s,
        training_date=metadata.training_date,
        save_date=metadata.save_date,
        training_epochs=metadata.training_epochs,
        final_train_loss=metadata.final_train_loss,
        final_val_loss=metadata.final_val_loss,
        has_analysis=metadata.has_analysis,
        n_active_components=metadata.n_active_components,
        tags=list(metadata.tags),
    )
    library = ModelLibrary(models_dir)
    library.add_entry(entry)

    logger.info(
        "Saved model '%s' (%s) to %s",
        metadata.name,
        metadata.model_id,
        model_path,
    )
    return model_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_model(
    model_path: Path,
    device: str = "cpu",
) -> LoadedModel:
    """Load a saved ``.distill`` model for immediate generation.

    Reconstructs the full model pipeline: ``ConvVAE`` with weights,
    ``AudioSpectrogram`` from saved config, ``AnalysisResult`` from
    saved latent analysis (if present), and ``ModelMetadata``.

    The returned :class:`LoadedModel` can be passed directly to
    :class:`~distill.inference.generation.GenerationPipeline`.

    Parameters
    ----------
    model_path : Path
        Path to the ``.distill`` file.
    device : str
        Device to load the model onto (default ``"cpu"``).

    Returns
    -------
    LoadedModel
        Complete model ready for generation.

    Raises
    ------
    ValueError
        If the file is not a valid ``.distill`` model or has an
        unsupported version.
    """
    import torch  # noqa: WPS433 -- lazy import

    from distill.audio.spectrogram import (
        AudioSpectrogram,
        ComplexSpectrogram,
        SpectrogramConfig,
    )
    from distill.controls.serialization import analysis_from_dict
    from distill.models.vae import ConvVAE
    from distill.training.config import ComplexSpectrogramConfig

    model_path = Path(model_path)
    saved = torch.load(model_path, map_location=device, weights_only=False)

    # Validate format marker and version
    if saved.get("format") != MODEL_FORMAT_MARKER:
        raise ValueError("Incompatible model format. Please retrain with current version.")
    version = saved.get("version", 0)
    if version > SAVED_MODEL_VERSION:
        raise ValueError(
            f"Model version {version} is newer than supported "
            f"version {SAVED_MODEL_VERSION}. Update the software."
        )

    # Reconstruct SpectrogramConfig and AudioSpectrogram
    spec_dict = saved["spectrogram_config"]
    spec_config = SpectrogramConfig(**spec_dict)
    spectrogram = AudioSpectrogram(spec_config)

    # Reconstruct ConvVAE with correct latent_dim
    latent_dim = saved.get("latent_dim", 128)
    model = ConvVAE(latent_dim=latent_dim)

    # CRITICAL: Initialize decoder (and encoder if present in state_dict)
    # linear layers BEFORE load_state_dict.  Compute spatial shape from
    # spectrogram config.
    n_mels = spec_config.n_mels
    time_frames = spec_config.sample_rate // spec_config.hop_length + 1

    # Pad to multiple of 32 (matching encoder forward pass)
    pad_h = (32 - n_mels % 32) % 32
    pad_w = (32 - time_frames % 32) % 32
    padded_h = n_mels + pad_h
    padded_w = time_frames + pad_w

    # After 5 stride-2 layers: spatial dims / 32
    spatial = (padded_h // 32, padded_w // 32)
    model.decoder._init_linear(spatial)

    # Init encoder linear layers only if they exist in the state_dict
    # (trained models have them; sample-only models may not)
    state_dict = saved["model_state_dict"]
    if "encoder.fc_mu.weight" in state_dict:
        flatten_dim = 1024 * spatial[0] * spatial[1]
        model.encoder._init_linear(flatten_dim)

    # Load weights
    model.load_state_dict(state_dict)
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    # Move spectrogram mel transform to device
    spectrogram.to(torch_device)

    # Reconstruct AnalysisResult if present
    analysis = None
    if saved.get("latent_analysis") is not None:
        analysis = analysis_from_dict(saved["latent_analysis"])

    # Reconstruct ComplexSpectrogram for ISTFT reconstruction
    complex_spec = ComplexSpectrogram(ComplexSpectrogramConfig())
    complex_spec.to(torch_device)

    # Extract normalization stats (None for old models without them)
    normalization_stats = saved.get("normalization_stats")

    # Reconstruct ModelMetadata from saved dict
    meta_dict = saved.get("metadata", {})
    metadata = ModelMetadata(**meta_dict)

    logger.info(
        "Loaded model '%s' from %s onto %s",
        metadata.name,
        model_path,
        device,
    )

    return LoadedModel(
        model=model,
        spectrogram=spectrogram,
        analysis=analysis,
        metadata=metadata,
        device=torch_device,
        complex_spectrogram=complex_spec,
        normalization_stats=normalization_stats,
    )


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def delete_model(model_id: str, models_dir: Path) -> bool:
    """Delete a model file and remove from library index.

    Parameters
    ----------
    model_id : str
        The model's unique ID.
    models_dir : Path
        Directory containing ``.distill`` files and the library index.

    Returns
    -------
    bool
        ``True`` if deletion succeeded, ``False`` if model not found.
    """
    from distill.library.catalog import ModelLibrary

    models_dir = Path(models_dir)
    library = ModelLibrary(models_dir)
    entry = library.get(model_id)

    if entry is None:
        return False

    # Delete the .distill file
    model_path = models_dir / entry.file_path
    if model_path.exists():
        model_path.unlink()

    # Remove from catalog
    library.remove(model_id)

    logger.info(
        "Deleted model '%s' (%s)",
        entry.name,
        model_id,
    )
    return True


# ---------------------------------------------------------------------------
# Convert checkpoint to saved model
# ---------------------------------------------------------------------------


def save_model_from_checkpoint(
    checkpoint_path: Path,
    metadata: ModelMetadata,
    models_dir: Path,
) -> Path:
    """Convert a training checkpoint to a saved model.

    Loads the checkpoint, extracts model weights and config,
    strips optimizer/scheduler state, and saves as a ``.distill`` file
    via :func:`save_model`.

    Parameters
    ----------
    checkpoint_path : Path
        Path to a ``.pt`` training checkpoint.
    metadata : ModelMetadata
        User-facing metadata for the saved model.
    models_dir : Path
        Directory to save the ``.distill`` file into.

    Returns
    -------
    Path
        Path to the saved ``.distill`` file.
    """
    import torch  # noqa: WPS433 -- lazy import

    from distill.audio.spectrogram import SpectrogramConfig
    from distill.controls.serialization import analysis_from_dict
    from distill.models.vae import ConvVAE

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    # Extract configs
    spectrogram_config = checkpoint.get("spectrogram_config", asdict(SpectrogramConfig()))
    training_config = checkpoint.get("training_config", {})

    # Reconstruct model
    latent_dim = checkpoint.get("latent_dim", 128)
    model = ConvVAE(latent_dim=latent_dim)

    # Initialize encoder/decoder linear layers from spectrogram config
    spec_config = SpectrogramConfig(**spectrogram_config)
    n_mels = spec_config.n_mels
    time_frames = spec_config.sample_rate // spec_config.hop_length + 1
    pad_h = (32 - n_mels % 32) % 32
    pad_w = (32 - time_frames % 32) % 32
    padded_h = n_mels + pad_h
    padded_w = time_frames + pad_w
    spatial = (padded_h // 32, padded_w // 32)  # 5 stride-2 layers
    model.decoder._init_linear(spatial)
    flatten_dim = 1024 * spatial[0] * spatial[1]
    model.encoder._init_linear(flatten_dim)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Extract analysis from checkpoint if present
    analysis = None
    if checkpoint.get("latent_analysis") is not None:
        analysis = analysis_from_dict(checkpoint["latent_analysis"])

    # Populate metadata from checkpoint data
    if checkpoint.get("epoch") is not None:
        metadata.training_epochs = checkpoint["epoch"]
    if checkpoint.get("train_loss") is not None:
        metadata.final_train_loss = float(checkpoint["train_loss"])
    if checkpoint.get("val_loss") is not None:
        metadata.final_val_loss = float(checkpoint["val_loss"])

    # Extract normalization stats from checkpoint (if present)
    normalization_stats = checkpoint.get("normalization_stats")

    # Save via the standard save_model function
    return save_model(
        model=model,
        spectrogram_config=spectrogram_config,
        training_config=training_config,
        metadata=metadata,
        models_dir=models_dir,
        analysis=analysis,
        normalization_stats=normalization_stats,
    )
