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
SAVED_MODEL_VERSION_V2 = 2
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


@dataclass
class LoadedVQModel:
    """Complete loaded VQ-VAE model ready for inference.

    Parallel to :class:`LoadedModel` but for VQ-VAE models (v2 format).
    Contains the reconstructed model, spectrogram converter, metadata,
    codebook health snapshot, and VQ-VAE configuration.

    No ``analysis`` field -- VQ-VAE replaces latent analysis with
    codebook health monitoring.

    If the model has a trained prior (``has_prior=True`` in the saved
    file), the :attr:`prior` field holds the reconstructed
    :class:`~distill.models.prior.CodePrior` in eval mode.
    """

    model: "ConvVQVAE"
    spectrogram: "AudioSpectrogram"
    metadata: ModelMetadata
    device: "torch.device"
    codebook_health: dict | None
    vqvae_config: dict | None
    prior: "CodePrior | None" = None
    prior_config: dict | None = None
    prior_metadata: dict | None = None


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
        SpectrogramConfig,
    )
    from distill.controls.serialization import analysis_from_dict
    from distill.models.vae import ConvVAE

    model_path = Path(model_path)
    saved = torch.load(model_path, map_location=device, weights_only=False)

    # Validate format marker and version
    if saved.get("format") != MODEL_FORMAT_MARKER:
        raise ValueError(f"Not a valid .distill model file: {model_path}")
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
    latent_dim = saved.get("latent_dim", 64)
    model = ConvVAE(latent_dim=latent_dim)

    # CRITICAL: Initialize decoder (and encoder if present in state_dict)
    # linear layers BEFORE load_state_dict.  Compute spatial shape from
    # spectrogram config.
    n_mels = spec_config.n_mels
    time_frames = spec_config.sample_rate // spec_config.hop_length + 1

    # Pad to multiple of 16 (matching encoder forward pass)
    pad_h = (16 - n_mels % 16) % 16
    pad_w = (16 - time_frames % 16) % 16
    padded_h = n_mels + pad_h
    padded_w = time_frames + pad_w

    # After 4 stride-2 layers: spatial dims / 16
    spatial = (padded_h // 16, padded_w // 16)
    model.decoder._init_linear(spatial)

    # Init encoder linear layers only if they exist in the state_dict
    # (trained models have them; sample-only models may not)
    state_dict = saved["model_state_dict"]
    if "encoder.fc_mu.weight" in state_dict:
        flatten_dim = 256 * spatial[0] * spatial[1]
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
    latent_dim = checkpoint.get("latent_dim", 64)
    model = ConvVAE(latent_dim=latent_dim)

    # Initialize encoder/decoder linear layers from spectrogram config
    spec_config = SpectrogramConfig(**spectrogram_config)
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

    # Save via the standard save_model function
    return save_model(
        model=model,
        spectrogram_config=spectrogram_config,
        training_config=training_config,
        metadata=metadata,
        models_dir=models_dir,
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# VQ-VAE v2 Save (v1.1)
# ---------------------------------------------------------------------------


def save_model_v2(
    model: "ConvVQVAE",
    spectrogram_config: dict,
    vqvae_config: dict,
    training_config: dict,
    metadata: ModelMetadata,
    models_dir: Path,
    codebook_health: dict | None = None,
    loss_curve_history: dict | None = None,
) -> Path:
    """Save a trained VQ-VAE model as a v2 ``.distill`` file.

    Creates a version 2 ``.distill`` file with VQ-specific metadata
    including codebook health snapshot and loss curve history.  Updates
    the :class:`~distill.library.catalog.ModelLibrary` index.

    Parameters
    ----------
    model : ConvVQVAE
        Trained VQ-VAE model (weights extracted via ``state_dict()``).
    spectrogram_config : dict
        Spectrogram configuration as a plain dict.
    vqvae_config : dict
        VQ-VAE configuration as a plain dict (codebook_dim, codebook_size,
        num_quantizers, etc.).
    training_config : dict
        Full training configuration as a plain dict.
    metadata : ModelMetadata
        User-facing metadata.  ``model_id`` and ``save_date`` are
        auto-generated if empty.
    models_dir : Path
        Directory to save the ``.distill`` file into.
    codebook_health : dict | None
        Per-level codebook health snapshot from the final validation pass.
    loss_curve_history : dict | None
        Epoch-level loss curves (train_losses, val_losses, recon_losses,
        commit_losses).

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

    # Build v2 saved model dict
    saved = {
        "format": MODEL_FORMAT_MARKER,
        "version": SAVED_MODEL_VERSION_V2,
        "model_type": "vqvae",
        "model_state_dict": model.state_dict(),
        "vqvae_config": vqvae_config,
        "spectrogram_config": spectrogram_config,
        "training_config": training_config,
        "codebook_health_snapshot": codebook_health,
        "loss_curve_history": loss_curve_history,
        "metadata": asdict(metadata),
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
        has_analysis=False,  # VQ-VAE uses codebook health, not analysis
        n_active_components=0,
        tags=list(metadata.tags),
    )
    library = ModelLibrary(models_dir)
    library.add_entry(entry)

    logger.info(
        "Saved VQ-VAE model '%s' (%s) v2 to %s",
        metadata.name,
        metadata.model_id,
        model_path,
    )
    return model_path


# ---------------------------------------------------------------------------
# Prior bundling into existing .sda / .distill files (v1.1 Phase 14)
# ---------------------------------------------------------------------------


def save_prior_to_model(
    model_path: Path,
    prior_model: "CodePrior",
    prior_config: dict,
    prior_metadata: dict,
) -> Path:
    """Bundle a trained prior into an existing v2 VQ-VAE model file.

    Atomically updates the saved ``.distill`` file by writing to a
    temporary file first, then replacing the original (per RESEARCH.md
    pitfall 6: atomic write to avoid corruption).

    Parameters
    ----------
    model_path : Path
        Path to the existing ``.distill`` VQ-VAE v2 model file.
    prior_model : CodePrior
        Trained :class:`~distill.models.prior.CodePrior` model.
    prior_config : dict
        Prior configuration dict (hidden_size, num_layers, num_heads,
        seq_len, num_quantizers, dropout).
    prior_metadata : dict
        Training metadata dict (epochs_trained, final_perplexity,
        best_perplexity, training_date).

    Returns
    -------
    Path
        The same *model_path* (now updated with prior data).

    Raises
    ------
    ValueError
        If the file is not a valid v2 VQ-VAE model.
    """
    import torch  # noqa: WPS433 -- lazy import

    model_path = Path(model_path)
    saved = torch.load(model_path, map_location="cpu", weights_only=False)

    # Validate format marker
    if saved.get("format") != MODEL_FORMAT_MARKER:
        raise ValueError(f"Not a valid .distill model file: {model_path}")

    # Validate version and model type
    version = saved.get("version", 0)
    model_type = saved.get("model_type", "")
    if version < 2 or model_type != "vqvae":
        raise ValueError(
            "Not a v2 VQ-VAE model. Prior can only be bundled into "
            "v2 VQ-VAE model files."
        )

    # Add / update prior keys
    saved["has_prior"] = True
    saved["prior_state_dict"] = prior_model.state_dict()
    saved["prior_config"] = prior_config
    saved["prior_metadata"] = prior_metadata

    # Atomic write: write to temp file, then replace
    temp_path = model_path.with_suffix(".tmp")
    torch.save(saved, temp_path)
    os.replace(temp_path, model_path)

    logger.info("Bundled prior into %s", model_path)
    return model_path


# ---------------------------------------------------------------------------
# VQ-VAE v2 Load (v1.1)
# ---------------------------------------------------------------------------


def load_model_v2(
    model_path: Path,
    device: str = "cpu",
) -> LoadedVQModel:
    """Load a v2 ``.distill`` VQ-VAE model for inference.

    Reconstructs the ``ConvVQVAE`` from saved configuration, loads weights,
    and returns a :class:`LoadedVQModel` with codebook health and config.

    Parameters
    ----------
    model_path : Path
        Path to the ``.distill`` file.
    device : str
        Device to load the model onto (default ``"cpu"``).

    Returns
    -------
    LoadedVQModel
        Complete VQ-VAE model ready for inference.

    Raises
    ------
    ValueError
        If the file is not a valid v2 VQ-VAE model.
    """
    import torch  # noqa: WPS433 -- lazy import

    from distill.audio.spectrogram import AudioSpectrogram, SpectrogramConfig
    from distill.models.vqvae import ConvVQVAE

    model_path = Path(model_path)
    saved = torch.load(model_path, map_location=device, weights_only=False)

    # Validate format marker
    if saved.get("format") != MODEL_FORMAT_MARKER:
        raise ValueError(f"Not a valid .distill model file: {model_path}")

    # Validate version and model type
    version = saved.get("version", 0)
    model_type = saved.get("model_type", "")

    if version < 2 or model_type != "vqvae":
        raise ValueError(
            "Not a v2 VQ-VAE model. Use load_model() for v1 models."
        )

    # Reconstruct SpectrogramConfig and AudioSpectrogram
    spec_dict = saved["spectrogram_config"]
    spec_config = SpectrogramConfig(**spec_dict)
    spectrogram = AudioSpectrogram(spec_config)

    # Reconstruct ConvVQVAE from vqvae_config
    vq_cfg = saved["vqvae_config"]
    model = ConvVQVAE(
        codebook_dim=vq_cfg.get("codebook_dim", 128),
        codebook_size=vq_cfg.get("codebook_size", 256),
        num_quantizers=vq_cfg.get("num_quantizers", 3),
        decay=vq_cfg.get("decay", 0.95),
        commitment_weight=vq_cfg.get("commitment_weight", 0.25),
        threshold_ema_dead_code=vq_cfg.get("threshold_ema_dead_code", 2),
        dropout=vq_cfg.get("dropout", 0.2),
    )

    # Initialize dimensions by running a dummy forward pass
    # ConvVQVAE uses Conv2d (no lazy init needed for spatial layers),
    # but ResidualVQ internal state needs initialization
    n_mels = spec_config.n_mels
    time_frames = spec_config.sample_rate // spec_config.hop_length + 1
    dummy_input = torch.zeros(1, 1, n_mels, time_frames)
    with torch.no_grad():
        model.eval()
        model(dummy_input)

    # Load weights
    model.load_state_dict(saved["model_state_dict"])
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    # Move spectrogram mel transform to device
    spectrogram.to(torch_device)

    # Reconstruct ModelMetadata from saved dict
    meta_dict = saved.get("metadata", {})
    metadata = ModelMetadata(**meta_dict)

    # ------------------------------------------------------------------
    # Reconstruct prior if present (Phase 14)
    # ------------------------------------------------------------------
    prior = None
    prior_config_dict = None
    prior_metadata_dict = None

    if saved.get("has_prior", False):
        from distill.models.prior import CodePrior  # lazy import

        prior_config_dict = saved["prior_config"]
        prior_metadata_dict = saved.get("prior_metadata")
        codebook_size = vq_cfg.get("codebook_size", 256)

        prior = CodePrior(
            codebook_size=codebook_size,
            seq_len=prior_config_dict["seq_len"],
            num_quantizers=prior_config_dict["num_quantizers"],
            hidden_size=prior_config_dict["hidden_size"],
            num_layers=prior_config_dict["num_layers"],
            num_heads=prior_config_dict["num_heads"],
            dropout=prior_config_dict.get("dropout", 0.1),
        )
        prior.load_state_dict(saved["prior_state_dict"])
        prior = prior.to(torch_device)
        prior.eval()

        logger.info(
            "Loaded prior (hidden=%d, layers=%d) from %s",
            prior_config_dict["hidden_size"],
            prior_config_dict["num_layers"],
            model_path,
        )

    logger.info(
        "Loaded VQ-VAE model '%s' (v2) from %s onto %s",
        metadata.name,
        model_path,
        device,
    )

    return LoadedVQModel(
        model=model,
        spectrogram=spectrogram,
        metadata=metadata,
        device=torch_device,
        codebook_health=saved.get("codebook_health_snapshot"),
        vqvae_config=saved.get("vqvae_config"),
        prior=prior,
        prior_config=prior_config_dict,
        prior_metadata=prior_metadata_dict,
    )
