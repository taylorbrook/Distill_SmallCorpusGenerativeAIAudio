# Phase 6: Model Persistence & Management - Research

**Researched:** 2026-02-13
**Domain:** Model serialization, metadata management, model library catalog, file-based persistence
**Confidence:** HIGH

## Summary

Phase 6 builds a model persistence and management layer on top of the existing training checkpoint system (Phase 3) and latent space analysis serialization (Phase 5). The core challenge is NOT serialization itself -- `torch.save`/`torch.load` with dict-based checkpoints is already working -- but rather creating a clean "saved model" format that bundles model weights, spectrogram config, latent space analysis, and rich metadata into a single `.sda` file, plus a model library catalog that enables browsing, searching, and filtering without loading heavy model files.

The existing codebase provides strong foundations: `training/checkpoint.py` already saves/loads model state_dicts with training config, spectrogram config, and metrics history via `torch.save`. `controls/serialization.py` already serializes `AnalysisResult` (PCA components, safe ranges, feature correlations) as numpy arrays within checkpoint dicts. The key gap is: (1) a "finished model" format distinct from training checkpoints (no optimizer/scheduler state, adds user-facing metadata like model name, description, dataset info); (2) a model library catalog stored as a JSON index file for fast scanning without loading `.pt` files; and (3) API functions for save/load/list/search/delete operations.

The approach is straightforward: define a `SavedModel` dataclass representing the complete saved state, write it via `torch.save` as a `.sda` file (which is internally a ZIP of pickled tensors), and maintain a `model_library.json` index file in the models directory for fast catalog operations. No database (SQLite) is needed -- the model count will be small (tens to low hundreds), and JSON provides human-readable inspection and zero-dependency querying.

**Primary recommendation:** Create a `models/persistence.py` module with `save_model()`, `load_model()`, `delete_model()` functions and a `library/catalog.py` module with `ModelLibrary` class managing the JSON index. The saved model format bundles `model_state_dict`, `spectrogram_config`, `latent_analysis`, `training_config`, and user metadata into a single `torch.save` dict with a format version field. Loading a model returns everything needed to immediately construct a `GenerationPipeline` with working slider controls.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.10.0 | `torch.save`/`torch.load` for model file I/O | Already in project; its ZIP-based serialization handles state_dicts + numpy arrays + Python primitives in a single file |
| json (stdlib) | N/A | Model library index catalog | Zero-dependency; human-readable; sufficient for tens-to-hundreds of model entries |
| pathlib (stdlib) | N/A | File path management | Already used throughout project |
| shutil (stdlib) | N/A | Safe file deletion, atomic file operations | Reliable cross-platform file operations |
| datetime (stdlib) | N/A | Training date, save date timestamps | ISO 8601 format for metadata |
| uuid (stdlib) | N/A | Unique model IDs for catalog entries | Collision-free identifiers without a database |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=1.26 | Already in project; latent analysis arrays in saved models | Serialized within `torch.save` dict |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON index file | SQLite database | SQLite is overkill for <1000 models; JSON is human-readable, zero-dependency, and trivially debuggable. SQLite would add complexity without benefit at this scale. |
| `torch.save` format | safetensors | safetensors adds a dependency; our files contain Python dicts and numpy arrays (not just tensors); `torch.save` is already proven in this codebase for checkpoints |
| UUID model IDs | Auto-increment integers | UUIDs allow safe concurrent operations (though unlikely in v1) and never conflict; integers require centralized counter management |
| Single `.sda` file per model | Directory per model | Single file is simpler to copy/backup/share; directory approach adds complexity with no benefit since all data fits in one file |

**Installation:**
```bash
# No new dependencies needed -- all stdlib + existing project dependencies
```

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── models/
│   ├── __init__.py          # [MODIFY] Add persistence exports
│   ├── vae.py               # [EXISTING] ConvVAE model
│   ├── losses.py            # [EXISTING] Loss functions
│   └── persistence.py       # [NEW] save_model(), load_model(), delete_model()
├── library/
│   ├── __init__.py          # [NEW] Public API exports
│   └── catalog.py           # [NEW] ModelLibrary, ModelEntry, search/filter
├── controls/
│   ├── serialization.py     # [EXISTING] analysis_to_dict, analysis_from_dict
│   └── ...                  # [EXISTING] No changes needed
├── training/
│   └── checkpoint.py        # [EXISTING] No changes needed
```

### Pattern 1: Saved Model Format (Single Dict via torch.save)
**What:** A standardized dict structure saved via `torch.save` that bundles everything needed to reconstruct a working GenerationPipeline with slider controls.
**When to use:** Every time a user saves a trained model.
**Example:**
```python
# Source: Existing checkpoint.py pattern + PyTorch torch.save docs
SAVED_MODEL_VERSION = 1

def _build_saved_model_dict(
    model: "ConvVAE",
    spectrogram_config: dict,
    analysis_dict: dict | None,
    training_config: dict,
    metadata: "ModelMetadata",
) -> dict:
    """Build the dict structure for torch.save.

    This is the canonical saved model format. All fields are
    Python primitives, numpy arrays, or PyTorch state dicts.
    No sklearn objects, no custom classes -- checkpoint portability
    is preserved (Phase 5 decision).
    """
    return {
        "format": "sda_model",
        "version": SAVED_MODEL_VERSION,
        # Model weights
        "model_state_dict": model.state_dict(),
        "latent_dim": model.latent_dim,
        # Spectrogram config (needed to reconstruct AudioSpectrogram)
        "spectrogram_config": spectrogram_config,
        # Latent space analysis (slider mappings)
        "latent_analysis": analysis_dict,  # None if not yet analyzed
        # Training parameters (for metadata display)
        "training_config": training_config,
        # User-facing metadata
        "metadata": {
            "model_id": metadata.model_id,
            "name": metadata.name,
            "description": metadata.description,
            "dataset_name": metadata.dataset_name,
            "dataset_file_count": metadata.dataset_file_count,
            "dataset_total_duration_s": metadata.dataset_total_duration_s,
            "training_date": metadata.training_date,
            "save_date": metadata.save_date,
            "training_epochs": metadata.training_epochs,
            "final_train_loss": metadata.final_train_loss,
            "final_val_loss": metadata.final_val_loss,
            "tags": metadata.tags,
        },
    }
```

### Pattern 2: Model Library JSON Index
**What:** A JSON file (`model_library.json`) in the models directory that stores lightweight metadata for all saved models. Enables fast catalog browsing without loading `.sda` files.
**When to use:** On every save/delete/search operation.
**Example:**
```python
# Source: Standard file-based index pattern
# model_library.json structure:
{
    "version": 1,
    "models": {
        "abc123-uuid": {
            "model_id": "abc123-uuid",
            "name": "My Ambient Model",
            "description": "Trained on field recordings",
            "file_path": "my_ambient_model.sda",
            "file_size_bytes": 12582912,
            "dataset_name": "field_recordings",
            "dataset_file_count": 42,
            "dataset_total_duration_s": 315.5,
            "training_date": "2026-02-13T14:30:00Z",
            "save_date": "2026-02-13T15:00:00Z",
            "training_epochs": 200,
            "final_train_loss": 0.0234,
            "final_val_loss": 0.0289,
            "has_analysis": True,
            "n_active_components": 8,
            "tags": ["ambient", "field recordings"]
        }
    }
}
```

### Pattern 3: Load Model -> Immediate Generation
**What:** Loading a model returns all components needed to construct a `GenerationPipeline` and (if analysis exists) restore slider controls immediately. No additional steps required.
**When to use:** When user selects a model from the library.
**Example:**
```python
# Source: Existing inference/generation.py pattern
from dataclasses import dataclass

@dataclass
class LoadedModel:
    """Complete loaded model ready for generation."""
    model: "ConvVAE"
    spectrogram: "AudioSpectrogram"
    analysis: "AnalysisResult | None"
    metadata: "ModelMetadata"
    device: "torch.device"

def load_model(
    model_path: Path,
    device: str = "cpu",
) -> LoadedModel:
    """Load a saved model and return everything needed for generation.

    Steps:
    1. torch.load the .sda file
    2. Reconstruct ConvVAE with correct latent_dim
    3. Load model_state_dict
    4. Reconstruct AudioSpectrogram from spectrogram_config
    5. Reconstruct AnalysisResult from latent_analysis (if present)
    6. Return LoadedModel ready for GenerationPipeline

    The loaded model can be passed directly to GenerationPipeline:
        loaded = load_model(path, device="mps")
        pipeline = GenerationPipeline(loaded.model, loaded.spectrogram, loaded.device)
        # If analysis exists, sliders work immediately:
        if loaded.analysis:
            slider_info = get_slider_info(loaded.analysis)
    """
    ...
```

### Pattern 4: Catalog Search and Filter
**What:** The ModelLibrary class provides search by name/tags and filter by metadata fields, operating entirely on the in-memory JSON index (no file I/O per query).
**When to use:** When user browses the model library in the UI.
**Example:**
```python
class ModelLibrary:
    """Manages the model catalog index.

    Loads the JSON index on construction and provides
    search/filter/sort operations over model entries.
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._index_path = models_dir / "model_library.json"
        self._entries: dict[str, ModelEntry] = {}
        self._load_index()

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        sort_by: str = "save_date",
        reverse: bool = True,
    ) -> list[ModelEntry]:
        """Search and filter model entries.

        Query matches against name and description (case-insensitive).
        Tags filter by exact match (any tag matches).
        """
        results = list(self._entries.values())

        if query:
            q = query.lower()
            results = [
                e for e in results
                if q in e.name.lower() or q in e.description.lower()
            ]

        if tags:
            tag_set = set(t.lower() for t in tags)
            results = [
                e for e in results
                if tag_set & set(t.lower() for t in e.tags)
            ]

        # Sort
        results.sort(
            key=lambda e: getattr(e, sort_by, ""),
            reverse=reverse,
        )
        return results
```

### Anti-Patterns to Avoid
- **Saving optimizer/scheduler state in finished models:** Checkpoints need optimizer state for resume; finished models do NOT. Including it roughly doubles file size for no benefit. Strip it during save_model.
- **Loading full .sda files for catalog browsing:** The JSON index exists specifically to avoid this. Never `torch.load` a model file just to display its metadata in the library browser.
- **Using pickle directly instead of torch.save:** `torch.save` uses pickle internally but wraps it in a ZIP format that handles tensors efficiently. Direct pickle would lose this optimization.
- **Storing model files outside the configured models directory:** Respect `config.toml paths.models` as the canonical location. All `.sda` files go there.
- **Re-inventing analysis serialization:** `controls/serialization.py` already has `analysis_to_dict()` and `analysis_from_dict()`. Use them directly.
- **Requiring analysis before saving:** Users must be able to save a model before running latent space analysis. The `latent_analysis` field in the saved model dict can be `None`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model file serialization | Custom binary format | `torch.save` (ZIP-based pickle) | Already proven in project; handles state_dicts, numpy arrays, Python primitives; version-compatible across PyTorch releases |
| Analysis serialization | New serialization for PCA data | `controls.serialization.analysis_to_dict()` / `analysis_from_dict()` | Already built and tested in Phase 5; includes version field for migration |
| Spectrogram reconstruction | Parse config dict manually | `dataclasses.asdict()` / `SpectrogramConfig(**dict)` | Dataclass round-trips cleanly through dict serialization |
| Unique IDs | Auto-increment counter file | `uuid.uuid4()` | Zero state management; collision-free; no counter file to corrupt |
| File locking for JSON index | Custom file locking mechanism | Atomic write pattern (write to temp, rename) | `os.replace()` is atomic on POSIX; handles crash-safety without locks |

**Key insight:** Phase 6 is primarily a data management and API design problem, not a serialization problem. The hard serialization work (model weights, PCA arrays, spectrogram config) is already solved by Phases 3 and 5. Phase 6 wraps it in a user-friendly package.

## Common Pitfalls

### Pitfall 1: Index File Corruption on Crash
**What goes wrong:** Writing the JSON index file is interrupted mid-write (power loss, crash, Ctrl-C). Next load finds truncated/invalid JSON.
**Why it happens:** Direct `open(path, 'w')` followed by `json.dump` is not atomic. A crash during write leaves a partial file.
**How to avoid:** Use the atomic write pattern: write to a temporary file in the same directory, then `os.replace(temp_path, index_path)`. `os.replace` is atomic on POSIX (macOS, Linux). Also keep a `.bak` copy of the previous index before overwriting.
**Warning signs:** `json.JSONDecodeError` on startup; empty or truncated `model_library.json`.

### Pitfall 2: Decoder Linear Layer Not Initialized After Load
**What goes wrong:** Loading a model's `state_dict` into a fresh `ConvVAE` instance restores the weights but the decoder's lazy linear layer (`decoder.fc`) may not be properly initialized, causing `RuntimeError` on first decode.
**Why it happens:** `ConvVAE` uses lazy initialization -- `decoder.fc` is created on first forward pass. `load_state_dict()` restores the weight values but the module must exist first.
**How to avoid:** After creating the `ConvVAE` instance, initialize the decoder's linear layer from the spectrogram config BEFORE calling `load_state_dict()`. Compute the spatial shape from `SpectrogramConfig` values: `n_mels` padded to multiple of 16, hop-derived time frames padded to 16, divided by 16 (4 stride-2 layers). Call `model.decoder._init_linear(spatial)` before loading weights.
**Warning signs:** `RuntimeError: Decoder linear layer not initialised`; `KeyError` during `load_state_dict` if keys mismatch.

### Pitfall 3: Model File and Index Out of Sync
**What goes wrong:** Index references a model file that doesn't exist (deleted externally), or a model file exists but isn't in the index (added externally, or save crashed after file write but before index update).
**Why it happens:** Two sources of truth (files on disk, JSON index) can diverge.
**How to avoid:** On library load, run a lightweight consistency check: verify all indexed model paths exist, scan for `.sda` files not in the index. Log warnings for inconsistencies. Provide a `repair_index()` method that re-scans the models directory and rebuilds missing entries by reading metadata from `.sda` files (this requires `torch.load` but only for orphaned files).
**Warning signs:** Models appearing in library but failing to load; models saved but not appearing in library.

### Pitfall 4: Saving Without Flushing Spectrogram Config
**What goes wrong:** The saved model's `spectrogram_config` doesn't match the config used during training, so loaded model produces garbled audio.
**Why it happens:** Spectrogram config is created inside the training loop (`SpectrogramConfig()` with defaults). If defaults ever change, older saved models break.
**How to avoid:** The existing checkpoint already stores `spectrogram_config` as a dict. The save_model function MUST extract this from the checkpoint, not create a fresh `SpectrogramConfig()`. On load, reconstruct `SpectrogramConfig` from the saved dict, not from defaults.
**Warning signs:** Loaded models producing audio at wrong sample rate or with wrong mel parameters.

### Pitfall 5: Large File Size from Including Training History
**What goes wrong:** Including the full `metrics_history` (potentially thousands of step-level metrics) in the saved model file bloats it by megabytes.
**Why it happens:** The training checkpoint stores the full metrics history for resume capability. Copying it into the saved model is unnecessary.
**How to avoid:** The saved model format includes only summary metrics (final_train_loss, final_val_loss, training_epochs) in the metadata dict. Do NOT include the full `metrics_history` from the checkpoint.
**Warning signs:** Saved model files that are significantly larger than the model weights alone (~12-15 MB for the ConvVAE).

### Pitfall 6: torch.load weights_only Incompatibility
**What goes wrong:** Using `weights_only=True` with `torch.load` fails because our saved model dicts contain numpy arrays and Python primitives, not just tensor weights.
**Why it happens:** `weights_only=True` restricts deserialization for security but is too restrictive for our format.
**How to avoid:** Use `weights_only=False` when loading `.sda` files (same as existing checkpoint.py pattern). These are user-generated local files, not untrusted downloads. Document this choice.
**Warning signs:** `UnpicklingError` or missing keys when loading saved models.

## Code Examples

Verified patterns from the existing codebase and official sources:

### Save Model from Checkpoint
```python
# Source: Existing checkpoint.py + controls/serialization.py patterns
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

@dataclass
class ModelMetadata:
    """User-facing metadata stored with a saved model."""
    model_id: str = ""
    name: str = "Untitled Model"
    description: str = ""
    dataset_name: str = ""
    dataset_file_count: int = 0
    dataset_total_duration_s: float = 0.0
    training_date: str = ""  # ISO 8601
    save_date: str = ""      # ISO 8601
    training_epochs: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    tags: list[str] = field(default_factory=list)
    has_analysis: bool = False
    n_active_components: int = 0

SAVED_MODEL_VERSION = 1
MODEL_FORMAT_MARKER = "sda_model"
MODEL_FILE_EXTENSION = ".sda"

def save_model(
    model: "ConvVAE",
    spectrogram_config: dict,
    training_config: dict,
    metadata: ModelMetadata,
    models_dir: Path,
    analysis: "AnalysisResult | None" = None,
) -> Path:
    """Save a trained model with metadata to the model library.

    Creates a .sda file in models_dir and updates the library index.
    """
    import torch  # noqa: WPS433

    from small_dataset_audio.controls.serialization import analysis_to_dict

    # Generate ID and save date if not set
    if not metadata.model_id:
        metadata.model_id = str(uuid.uuid4())
    if not metadata.save_date:
        metadata.save_date = datetime.now(timezone.utc).isoformat()

    # Set analysis metadata
    analysis_dict = None
    if analysis is not None:
        analysis_dict = analysis_to_dict(analysis)
        metadata.has_analysis = True
        metadata.n_active_components = analysis.n_active_components

    # Build the saved model dict
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
    model_path = models_dir / file_name

    # Handle duplicate filenames
    counter = 1
    while model_path.exists():
        file_name = f"{safe_name}_{counter}{MODEL_FILE_EXTENSION}"
        model_path = models_dir / file_name
        counter += 1

    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(saved, model_path)

    return model_path
```

### Load Model for Immediate Generation
```python
# Source: Existing checkpoint.py load_checkpoint pattern
def load_model(
    model_path: Path,
    device: str = "cpu",
) -> "LoadedModel":
    """Load a saved .sda model for generation.

    Returns a LoadedModel with model, spectrogram, analysis,
    and metadata -- everything needed for GenerationPipeline.
    """
    import torch  # noqa: WPS433

    from small_dataset_audio.audio.spectrogram import (
        AudioSpectrogram,
        SpectrogramConfig,
    )
    from small_dataset_audio.controls.serialization import analysis_from_dict
    from small_dataset_audio.models.vae import ConvVAE

    saved = torch.load(model_path, map_location=device, weights_only=False)

    # Version check
    version = saved.get("version", 0)
    if saved.get("format") != MODEL_FORMAT_MARKER:
        raise ValueError(f"Not a valid .sda model file: {model_path}")
    if version > SAVED_MODEL_VERSION:
        raise ValueError(
            f"Model version {version} is newer than supported "
            f"{SAVED_MODEL_VERSION}. Update the software."
        )

    # Reconstruct spectrogram
    spec_dict = saved["spectrogram_config"]
    spec_config = SpectrogramConfig(**spec_dict)
    spectrogram = AudioSpectrogram(spec_config)

    # Reconstruct model
    latent_dim = saved.get("latent_dim", 64)
    model = ConvVAE(latent_dim=latent_dim)

    # CRITICAL: Initialize decoder before loading state_dict
    # Compute spatial shape from spectrogram config
    n_mels = spec_config.n_mels
    time_frames = spec_config.sample_rate // spec_config.hop_length + 1
    pad_h = (16 - n_mels % 16) % 16
    pad_w = (16 - time_frames % 16) % 16
    spatial = ((n_mels + pad_h) // 16, (time_frames + pad_w) // 16)
    model.decoder._init_linear(spatial)
    # Also init encoder linear layers
    flatten_dim = 256 * spatial[0] * spatial[1]
    model.encoder._init_linear(flatten_dim)

    model.load_state_dict(saved["model_state_dict"])
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    spectrogram.to(torch_device)

    # Reconstruct analysis (if present)
    analysis = None
    if saved.get("latent_analysis") is not None:
        analysis = analysis_from_dict(saved["latent_analysis"])

    # Reconstruct metadata
    meta_dict = saved.get("metadata", {})
    metadata = ModelMetadata(**meta_dict)

    return LoadedModel(
        model=model,
        spectrogram=spectrogram,
        analysis=analysis,
        metadata=metadata,
        device=torch_device,
    )
```

### Atomic JSON Index Write
```python
# Source: Standard atomic write pattern
import json
import os
import tempfile

def _write_index_atomic(index_path: Path, data: dict) -> None:
    """Write JSON index atomically to prevent corruption.

    Writes to a temp file in the same directory, then
    os.replace (atomic on POSIX) to the target path.
    """
    # Backup existing index
    if index_path.exists():
        backup_path = index_path.with_suffix(".json.bak")
        try:
            import shutil
            shutil.copy2(index_path, backup_path)
        except OSError:
            pass  # Best effort

    # Write to temp file in same directory (same filesystem)
    fd, tmp_path = tempfile.mkstemp(
        dir=index_path.parent,
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, index_path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

### Delete Model with Index Update
```python
# Source: Standard file management pattern
def delete_model(
    model_id: str,
    models_dir: Path,
) -> bool:
    """Delete a model file and remove from library index.

    Returns True if deletion succeeded, False if model not found.
    """
    library = ModelLibrary(models_dir)
    entry = library.get(model_id)
    if entry is None:
        return False

    # Delete file first, then update index
    model_path = models_dir / entry.file_path
    if model_path.exists():
        model_path.unlink()

    library.remove(model_id)
    library.save()
    return True
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full model pickle (`torch.save(model)`) | State dict only (`torch.save(model.state_dict())`) | PyTorch best practice since 2019 | Decouples saved data from code structure; survives refactors |
| Separate metadata files alongside model files | Single bundled file (dict via `torch.save`) | Common pattern | Atomic save/load; no sync issues between weight file and metadata file |
| Database-backed model registries (MLflow, W&B) | File-based JSON index for local tools | Appropriate at this scale | No server dependency; human-readable; suits desktop application |
| sklearn PCA objects in checkpoints | Numpy arrays only (project Phase 5 decision) | Phase 5 | Cross-version portability; no sklearn version lock-in |

**Deprecated/outdated:**
- `torch.save(model)` (saving entire model object): Version-fragile; breaks on code refactors. Use state_dict only.
- `weights_only=True` for complex checkpoints: Too restrictive for dicts containing numpy arrays and nested Python structures. Use `weights_only=False` for local model files.

## Open Questions

1. **Filename sanitization strategy**
   - What we know: Model names come from user input and may contain spaces, special characters, unicode.
   - What's unclear: Exact sanitization rules for cross-platform filenames (macOS allows more than Windows).
   - Recommendation: Replace non-alphanumeric characters (except hyphens and underscores) with underscores. Truncate to 100 chars. Lowercase. This is conservative but safe across platforms.

2. **Maximum model library size**
   - What we know: JSON index works well for hundreds of entries. Performance would degrade at thousands.
   - What's unclear: Whether users will ever have more than ~100 models.
   - Recommendation: JSON is fine for v1. Document the assumption (<1000 models). If needed, add SQLite migration in a future phase. The index format's `version` field enables this migration.

3. **Saving model from checkpoint vs. from live training result**
   - What we know: `train()` returns `{model, metrics_history, output_dir, best_checkpoint_path}`. Users may also want to save from an existing checkpoint file.
   - What's unclear: Whether to support both "save from memory" (after training just completed) and "save from checkpoint file" (converting an old checkpoint to a saved model).
   - Recommendation: Support both paths. `save_model()` takes a live model + metadata. A separate `save_model_from_checkpoint()` loads a checkpoint, strips optimizer state, and saves as `.sda`. This covers the common case (save after training) and the recovery case (convert old checkpoints).

4. **Model file size estimation**
   - What we know: ConvVAE has ~3.1M parameters. At float32, that's ~12.4 MB for the state_dict alone. Spectrogram config, training config, and metadata are negligible (~1 KB). Analysis data adds ~10-100 KB.
   - What's unclear: Exact overhead from `torch.save` ZIP format and pickle framing.
   - Recommendation: Expect ~13-15 MB per saved model. Document this for users. No compression needed -- the ZIP format in `torch.save` already handles this.

## Sources

### Primary (HIGH confidence)
- [PyTorch Saving and Loading Models Tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) -- torch.save/load patterns, state_dict best practices, weights_only parameter
- [PyTorch Save and Load Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) -- Confirmed ZIP-based format, dict-based checkpoint pattern
- Existing codebase: `training/checkpoint.py` (save/load checkpoint dict pattern), `controls/serialization.py` (analysis to/from dict), `models/vae.py` (ConvVAE architecture, lazy init), `inference/generation.py` (GenerationPipeline constructor), `config/defaults.py` (paths.models), `config/settings.py` (resolve_path)

### Secondary (MEDIUM confidence)
- [PyTorch torch.package docs](https://docs.pytorch.org/docs/stable/package.html) -- Considered but unnecessary; `torch.save` dict format is sufficient for our needs
- [safetensors by HuggingFace](https://huggingface.co/docs/safetensors/index) -- Evaluated as alternative; adds dependency without benefit for local-only model files with mixed data types

### Tertiary (LOW confidence)
- [model-index PyPI](https://pypi.org/project/model-index/) -- Inspiration for JSON-based model catalog pattern; validated that JSON index is a standard approach for small-scale model management

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All stdlib or existing project deps; `torch.save` dict pattern already proven in checkpoint.py
- Architecture: HIGH -- Extends existing patterns (checkpoint save/load, analysis serialization) into a user-facing model format; no novel technology
- Saved model format: HIGH -- Direct extension of existing checkpoint dict with training state stripped and metadata added
- Model library catalog: HIGH -- JSON index is a well-understood pattern; search/filter on small collections is trivial
- Lazy init handling on load: HIGH -- Verified by reading vae.py source; spatial shape computation from spectrogram config is deterministic
- Pitfalls: HIGH -- Identified from codebase analysis (lazy init, atomic writes, index sync) and PyTorch documentation (weights_only)

**Research date:** 2026-02-13
**Valid until:** 2026-03-15 (stable domain; file I/O and model serialization are mature patterns)
