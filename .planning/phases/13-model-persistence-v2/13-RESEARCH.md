# Phase 13: Model Persistence v2 - Research

**Researched:** 2026-02-21
**Domain:** Model file format redesign, vocoder state bundling, catalog extension
**Confidence:** HIGH

## Summary

Phase 13 replaces the v1 `.distill` model format with a new `.distillgan` format that supports optional per-model vocoder state bundling. The user has made a deliberate decision to **break backward compatibility entirely** -- no migration, no dual-format support, no conversion tools. Old `.distill` files are invisible to the application; users retrain models with the new code.

The implementation scope is well-bounded: update 3 constants (extension, format marker, version), add an optional vocoder state dict to the save/load pipeline, extend `ModelEntry`/`ModelMetadata` with vocoder fields, update the catalog index to show vocoder training stats, and sweep all references to `.distill` across the codebase. The existing `torch.save`/`torch.load` pipeline handles vocoder state bundling natively -- no new serialization machinery needed.

The riskiest aspect is the comprehensive sweep of `.distill` references across the codebase (persistence, catalog, CLI, UI, __init__.py re-exports). Missing a reference means silent breakage or confusing error messages. A systematic grep-and-replace approach with verification is essential.

**Primary recommendation:** Treat this as a clean-room format swap (constants + schema extension) rather than a migration. Update the 3 constants in `persistence.py`, extend the save/load/catalog dataclasses, add v1 rejection logic, and sweep all file references. No new libraries or complex patterns needed.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **No backward compatibility** -- PERS-02 is dropped entirely
- v1 `.distill` files are completely ignored by the application (not recognized, not loaded, not scanned)
- Users must retrain models with the new code; no conversion tooling provided
- When someone attempts to load a v1 file directly (e.g., via CLI path argument), raise a clear error: "This model was saved in v1 format which is no longer supported. Please retrain your model."
- **New extension:** `.distillgan` (replaces `.distill`)
- **New format marker:** `distillgan_model` (replaces `distill_model`)
- **Version:** starts at `1` for the new format (clean slate, not `2` of the old format)
- Training checkpoint files stay as `.pt` -- only the user-facing saved model format changes
- Old `.distill` files are invisible to the app -- treated like any non-model file
- **Single file** -- per-model vocoder weights are stored inside the `.distillgan` file alongside model weights, config, analysis, and metadata
- Store vocoder state_dict, model config (architecture params), AND training metadata (epochs, final loss, training date)
- Show vocoder training stats (epochs, loss) in the catalog -- not just presence/absence
- No vocoder-based filtering in this phase -- keep search as-is
- Vocoder stats visible alongside existing model metrics

### Claude's Discretion
- Whether to use null marker or omit vocoder key when no vocoder is present
- Whether to allow stripping vocoder state on re-save
- Catalog display approach (badge, column, or icon) -- pick what fits existing UI patterns
- Nested sub-object vs flat fields in the JSON catalog index -- pick the better data modeling approach

### Deferred Ideas (OUT OF SCOPE)
- Vocoder-based filtering in model catalog -- can add later if model count warrants it
- Batch conversion tool for v1->v2 -- explicitly rejected, but could revisit if users complain
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PERS-01 | Model format v2 stores optional per-model HiFi-GAN vocoder state | Extend `save_model` dict with `vocoder_state` key containing state_dict + config + training metadata. `torch.save` handles this natively. New `ModelMetadata` fields track vocoder presence. |
| PERS-02 | Existing v1.0 .sda files load without error (backward compatible) | **DROPPED by user decision.** v1 `.distill` files are completely ignored. Attempting to load one via CLI raises a clear error message. No migration path provided. |
| PERS-03 | Model catalog indicates whether a model has a trained per-model vocoder | Extend `ModelEntry` with vocoder fields (has_vocoder, vocoder_epochs, vocoder_loss). Display in catalog UI (model cards, CLI table, info command). |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10.0 | `torch.save` / `torch.load` for model serialization | Already used for all persistence; handles arbitrary nested dicts with tensors |
| dataclasses | stdlib | `ModelMetadata`, `ModelEntry`, `LoadedModel` dataclasses | Already used throughout; `asdict()` for JSON serialization |
| json | stdlib | Catalog index (`model_library.json`) | Already used for atomic index writes |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | File path operations | Already used everywhere in persistence and catalog |
| uuid | stdlib | Model ID generation | Already used in `save_model` |
| os | stdlib | `os.path.getsize`, `os.replace` for atomic writes | Already used in catalog |

### Alternatives Considered
None -- the existing stack handles everything needed. No new dependencies required.

**Installation:**
No new packages needed. This phase is purely internal refactoring of existing serialization code.

## Architecture Patterns

### Recommended Project Structure
```
src/distill/
├── models/
│   ├── persistence.py    # MODIFY: constants, save/load, metadata, LoadedModel
│   └── __init__.py       # MODIFY: re-export updated constants
├── library/
│   └── catalog.py        # MODIFY: ModelEntry fields, repair_index glob
├── cli/
│   ├── model.py          # MODIFY: model info display, v1 error messages
│   └── generate.py       # MODIFY: .distillgan extension check in resolve_model
├── ui/
│   ├── components/
│   │   └── model_card.py # MODIFY: vocoder stats display in cards
│   └── tabs/
│       └── library_tab.py # MODIFY: table headers, vocoder column
├── training/
│   └── loop.py           # MODIFY: docstring references to .distill
└── vocoder/
    └── __init__.py       # No change (HiFi-GAN loading is Phase 16)
```

### Pattern 1: Optional Vocoder State in Save Dict
**What:** Add a `vocoder_state` key to the `torch.save` dict. When no vocoder is trained, omit the key entirely (not `None`). This is cleaner than a null marker because `saved.get("vocoder_state")` returns `None` naturally.
**When to use:** Every save and load operation.
**Example:**
```python
# In save_model():
saved = {
    "format": MODEL_FORMAT_MARKER,       # "distillgan_model"
    "version": SAVED_MODEL_VERSION,       # 1
    "model_state_dict": model.state_dict(),
    "latent_dim": model.latent_dim,
    "spectrogram_config": spectrogram_config,
    "latent_analysis": analysis_dict,
    "training_config": training_config,
    "metadata": asdict(metadata),
}

# Only add vocoder state if present (omit key entirely when absent)
if vocoder_state is not None:
    saved["vocoder_state"] = vocoder_state

# In load_model():
vocoder_state = saved.get("vocoder_state")  # None if absent
```

### Pattern 2: Nested Vocoder Sub-Object in Catalog Index
**What:** Use a nested `vocoder` sub-object in `ModelEntry` rather than flat fields. This groups related data logically and makes it easy to check `entry.vocoder is not None` vs inspecting multiple flat fields.
**When to use:** `ModelEntry` dataclass and JSON catalog index.
**Example:**
```python
@dataclass
class VocoderInfo:
    """Vocoder training metadata for catalog display."""
    type: str              # "hifigan_v2" (future-proofed for other vocoder types)
    epochs: int
    final_loss: float
    training_date: str     # ISO 8601

@dataclass
class ModelEntry:
    # ... existing fields ...
    vocoder: VocoderInfo | None = None  # None = no vocoder trained
```

**Rationale for nested over flat:** The vocoder fields are conceptually grouped (they all relate to vocoder training), `None` vs present is a clean presence check, and it avoids polluting `ModelEntry` with `vocoder_epochs`, `vocoder_loss`, `vocoder_type`, `vocoder_training_date` flat fields that are all-or-nothing. The existing `asdict()` serialization handles nested dataclasses correctly.

### Pattern 3: v1 Detection and Rejection
**What:** When a user tries to load a `.distill` file via CLI path argument, detect the old format and raise a clear error.
**When to use:** `resolve_model` in `generate.py` and `load_model` in `persistence.py`.
**Example:**
```python
# In resolve_model() (cli/generate.py):
if model_ref.endswith(".distill"):
    raise typer.BadParameter(
        "This model was saved in v1 format (.distill) which is no longer "
        "supported. Please retrain your model."
    )

# In load_model() (persistence.py):
if saved.get("format") == "distill_model":
    raise ValueError(
        "This model was saved in v1 format which is no longer supported. "
        "Please retrain your model."
    )
```

### Pattern 4: Vocoder State Dict Structure
**What:** Define the structure of the vocoder state stored in the model file. Must include everything needed to reconstruct the HiFi-GAN V2 vocoder.
**When to use:** When saving a model that has a trained vocoder (Phase 16 will produce these).
**Example:**
```python
vocoder_state = {
    "type": "hifigan_v2",
    "state_dict": hifigan_model.state_dict(),
    "config": {
        # HiFi-GAN V2 architecture params needed for reconstruction
        "upsample_rates": [8, 8, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4, 4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        # ... other arch params
    },
    "training_metadata": {
        "epochs": 200,
        "final_loss": 0.0342,
        "training_date": "2026-02-21T10:30:00+00:00",
    },
}
```

### Anti-Patterns to Avoid
- **Don't version the format as v2:** The user decided on a clean slate. `SAVED_MODEL_VERSION = 1` for the new `.distillgan` format. This is a NEW format, not an upgrade of the old one.
- **Don't scan for .distill files anywhere:** The catalog `repair_index` globs for `*.distill` -- this must change to `*.distillgan`. Leaving old globs means orphan detection scans old files.
- **Don't store vocoder weights separately:** User decided on single-file bundling. No sidecar vocoder files.
- **Don't add vocoder loading logic in this phase:** Phase 16 handles HiFi-GAN training and the actual vocoder construction from saved state. Phase 13 only stores and loads the raw state dict/config. The `LoadedModel` dataclass should get a `vocoder_state` field, but actual vocoder reconstruction is deferred.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tensor serialization | Custom binary format | `torch.save` / `torch.load` | Already handles nested dicts with tensors, state_dicts, arbitrary Python objects |
| JSON catalog with atomic writes | Custom file locking | Existing `_write_index_atomic` | Already proven pattern with temp file + `os.replace` |
| Dataclass serialization | Custom dict conversion | `dataclasses.asdict()` | Handles nested dataclasses, already used throughout |
| File extension checks | Regex patterns | `str.endswith()` / `Path.suffix` | Simple, readable, consistent with existing code |

**Key insight:** This phase adds no new complexity domains. It modifies existing well-understood patterns. The discipline is in thoroughness (finding all references) not in novel engineering.

## Common Pitfalls

### Pitfall 1: Missing a .distill Reference
**What goes wrong:** Some code path still uses `.distill` extension or `distill_model` format marker, causing failures or silent mismatches.
**Why it happens:** References to the old format are spread across 8+ files (persistence.py, catalog.py, __init__.py, cli/model.py, cli/generate.py, ui/model_card.py, ui/tabs/library_tab.py, training/loop.py).
**How to avoid:** Run a comprehensive grep for `.distill`, `distill_model`, `MODEL_FILE_EXTENSION`, `MODEL_FORMAT_MARKER` before marking the phase complete. Verify every hit is updated.
**Warning signs:** Tests pass but generate files with `.distill` extension; CLI says "not a valid .distill model file" when loading a `.distillgan` file.

### Pitfall 2: Catalog Index Incompatibility
**What goes wrong:** Adding `vocoder` field to `ModelEntry` causes existing catalog entries (without vocoder field) to fail deserialization with `TypeError: unexpected keyword argument`.
**Why it happens:** `ModelEntry(**entry_dict)` in `_load_index()` passes all dict keys as kwargs. New required fields break old entries.
**How to avoid:** Make the `vocoder` field optional (`vocoder: VocoderInfo | None = None`). Handle missing keys gracefully in deserialization. The existing `_load_index` already wraps entry creation in try/except.
**Warning signs:** Library shows empty after upgrade; log warnings about "corrupt entry" for every model.

### Pitfall 3: Model File Size Reporting
**What goes wrong:** File size display in catalog/CLI doesn't account for the dramatic size increase when vocoder is bundled (~6MB without vs ~50-100MB with).
**Why it happens:** No change is made to file size display, but the scale changes dramatically.
**How to avoid:** The existing `_format_file_size` helper already handles MB formatting. Just ensure file size updates correctly in catalog entry when re-saving with vocoder.
**Warning signs:** File sizes look wrong in the library view.

### Pitfall 4: Catalog Index Version Not Bumped
**What goes wrong:** The `model_library.json` index version stays at 1, but the schema now has a `vocoder` field. If someone downgrades, the old code doesn't know the index has new fields.
**Why it happens:** Forgetting to bump `_INDEX_VERSION` in catalog.py.
**How to avoid:** Bump `_INDEX_VERSION` to 2 in catalog.py. Add a check that warns if index version is newer than supported.
**Warning signs:** Old code silently ignores vocoder data in catalog.

### Pitfall 5: VocoderInfo Deserialization from JSON
**What goes wrong:** `ModelEntry(**entry_dict)` receives a raw dict for the `vocoder` field but expects `VocoderInfo | None`. Plain `dataclasses.asdict()` writes nested dicts, but construction from dict requires manual reconstruction.
**Why it happens:** `asdict()` flattens nested dataclasses to dicts, but `ModelEntry(**dict)` doesn't auto-reconstruct nested dataclasses.
**How to avoid:** In `_load_index()`, manually reconstruct `VocoderInfo` from the nested dict before constructing `ModelEntry`. Or use a custom `from_dict` classmethod.
**Warning signs:** `TypeError` during catalog load; vocoder info shows as raw dict instead of dataclass.

### Pitfall 6: LoadedModel Not Extended for Vocoder State
**What goes wrong:** `LoadedModel` dataclass lacks a `vocoder_state` field, so even though the file stores vocoder state, `load_model()` doesn't return it.
**Why it happens:** Forgetting that downstream code (Phase 16) needs access to the vocoder state from the loaded model.
**How to avoid:** Add `vocoder_state: dict | None = None` to `LoadedModel`. `load_model()` reads `saved.get("vocoder_state")` and passes it through.
**Warning signs:** Phase 16 can't access vocoder weights from loaded models.

## Code Examples

Verified patterns from the existing codebase:

### Constant Updates (persistence.py)
```python
# Source: src/distill/models/persistence.py lines 37-39
# BEFORE:
SAVED_MODEL_VERSION = 1
MODEL_FORMAT_MARKER = "distill_model"
MODEL_FILE_EXTENSION = ".distill"

# AFTER:
SAVED_MODEL_VERSION = 1                      # stays 1 (new format, clean slate)
MODEL_FORMAT_MARKER = "distillgan_model"     # new marker
MODEL_FILE_EXTENSION = ".distillgan"         # new extension
```

### Save Model with Optional Vocoder State
```python
# Source: Modified from src/distill/models/persistence.py save_model()
def save_model(
    model: "ConvVAE",
    spectrogram_config: dict,
    training_config: dict,
    metadata: ModelMetadata,
    models_dir: Path,
    analysis: "AnalysisResult | None" = None,
    vocoder_state: dict | None = None,       # NEW parameter
) -> Path:
    # ... existing setup ...

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

    # Only include vocoder state when present (omit key when absent)
    if vocoder_state is not None:
        saved["vocoder_state"] = vocoder_state

    # ... rest of save logic unchanged ...
```

### Load Model with v1 Rejection and Vocoder State
```python
# Source: Modified from src/distill/models/persistence.py load_model()
def load_model(model_path: Path, device: str = "cpu") -> LoadedModel:
    # ... torch.load ...

    # Reject v1 format with clear error message
    fmt = saved.get("format", "")
    if fmt == "distill_model":
        raise ValueError(
            "This model was saved in v1 format which is no longer supported. "
            "Please retrain your model."
        )
    if fmt != MODEL_FORMAT_MARKER:
        raise ValueError(f"Not a valid .distillgan model file: {model_path}")

    # ... existing model reconstruction ...

    # Extract vocoder state if present
    vocoder_state = saved.get("vocoder_state")

    return LoadedModel(
        model=model,
        spectrogram=spectrogram,
        analysis=analysis,
        metadata=metadata,
        device=torch_device,
        vocoder_state=vocoder_state,  # NEW field
    )
```

### ModelEntry with Nested VocoderInfo
```python
# Source: Modified from src/distill/library/catalog.py
@dataclass
class VocoderInfo:
    """Vocoder training metadata for catalog display."""
    type: str                    # "hifigan_v2"
    epochs: int
    final_loss: float
    training_date: str           # ISO 8601

@dataclass
class ModelEntry:
    # ... all existing fields ...
    vocoder: VocoderInfo | None = None

# Custom deserialization in _load_index():
vocoder_raw = entry_dict.pop("vocoder", None)
vocoder = VocoderInfo(**vocoder_raw) if vocoder_raw else None
entries[model_id] = ModelEntry(**entry_dict, vocoder=vocoder)
```

### Catalog Repair Index Update
```python
# Source: Modified from src/distill/library/catalog.py repair_index()
# Change glob from *.distill to *.distillgan
for model_path in self.models_dir.glob("*.distillgan"):
    # ... existing orphan detection logic ...
```

### CLI Generate Model Resolution
```python
# Source: Modified from src/distill/cli/generate.py resolve_model()
def resolve_model(model_ref: str, models_dir: Path, device: str) -> "LoadedModel":
    # Reject v1 format with clear message
    if model_ref.endswith(".distill"):
        raise typer.BadParameter(
            "This model was saved in v1 format (.distill) which is no longer "
            "supported. Please retrain your model."
        )

    # Load new format
    if model_ref.endswith(".distillgan"):
        sda_path = Path(model_ref)
        if sda_path.exists():
            return load_model(sda_path, device=device)
        raise typer.BadParameter(f"Model file not found: {model_ref}")

    # ... rest of UUID/name lookup unchanged ...
```

### Model Card Vocoder Display
```python
# Source: Modified from src/distill/ui/components/model_card.py render_single_card()
# Add vocoder badge after existing stats
vocoder_str = ""
if model.vocoder is not None:
    vocoder_str = (
        f'<p style="margin: 4px 0; font-size: 0.9em;">'
        f'<span style="background: #d1fae5; color: #065f46; '
        f'padding: 2px 8px; border-radius: 12px; font-size: 0.85em;">'
        f'HiFi-GAN</span> '
        f'{model.vocoder.epochs} epochs &middot; '
        f'loss {model.vocoder.final_loss:.4f}'
        f'</p>'
    )
```

## Comprehensive Reference Sweep

All files containing `.distill`, `distill_model`, or related constants that must be updated:

| File | What to Change | Confidence |
|------|----------------|------------|
| `src/distill/models/persistence.py` | Constants, save_model, load_model, save_model_from_checkpoint, delete_model, ModelMetadata, LoadedModel, docstrings | HIGH |
| `src/distill/models/__init__.py` | Re-exports of updated constants | HIGH |
| `src/distill/library/catalog.py` | ModelEntry fields, VocoderInfo, repair_index glob, _load_index deserialization, _INDEX_VERSION | HIGH |
| `src/distill/cli/generate.py` | `resolve_model` extension checks, docstrings | HIGH |
| `src/distill/cli/model.py` | `model_info` display (vocoder stats), `_find_model_entry` docstrings, `model` argument help text | HIGH |
| `src/distill/ui/components/model_card.py` | `render_single_card` vocoder display | HIGH |
| `src/distill/ui/tabs/library_tab.py` | Table headers, vocoder column | HIGH |
| `src/distill/training/loop.py` | Comment references to `.distill` (line 707) | HIGH |
| `src/distill/training/runner.py` | Docstring reference to `.distill` (line 100) | HIGH |
| `src/distill/audio/metadata.py` | `DISTILL_MODEL` tag key -- keep as-is (audio tags, not file format) | HIGH |

**Note on `src/distill/audio/metadata.py`:** This file contains `DISTILL_MODEL` as an audio metadata tag key (ID3/Vorbis) for provenance embedding in exported audio files. This is unrelated to the model file format and should NOT be changed. The tag name `DISTILL_MODEL` identifies audio provenance, not the model file format.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `.distill` extension | `.distillgan` extension | Phase 13 | All model file operations |
| `distill_model` format marker | `distillgan_model` format marker | Phase 13 | Format validation |
| VAE-only model files (~6MB) | VAE + optional vocoder (~6MB or ~50-100MB) | Phase 13 | File size, save/load |
| No vocoder info in catalog | Vocoder training stats in catalog | Phase 13 | UI display |

**Deprecated/outdated:**
- `.distill` extension: Replaced by `.distillgan`, v1 files completely ignored
- `distill_model` format marker: Replaced by `distillgan_model`
- PERS-02 (backward compatibility): Explicitly dropped by user decision

## Discretion Recommendations

### 1. Null marker vs omit vocoder key
**Recommendation: Omit the key entirely when no vocoder is present.**
- `saved.get("vocoder_state")` returns `None` naturally when key is absent
- Smaller file size (no null serialization overhead)
- Consistent with existing pattern: `latent_analysis` is sometimes absent from the dict
- Cleaner in JSON catalog too: `"vocoder": null` adds noise vs just omitting the field

### 2. Allow stripping vocoder state on re-save
**Recommendation: Do not implement stripping in this phase.**
- The save function already accepts `vocoder_state=None` which naturally omits vocoder
- No UI or CLI flow currently re-saves models
- If needed later, it's a trivial load-then-save-without-vocoder operation
- Keep the phase minimal; stripping can be added when there's a use case

### 3. Catalog display approach
**Recommendation: Use a badge/pill approach in model cards, a column in the CLI table.**
- Model cards already use styled `<span>` pills for tags (see `model_card.py` lines 62-67)
- A green pill labeled "HiFi-GAN" with epoch/loss stats matches the existing visual language
- CLI table: add a "Vocoder" column showing "HiFi-GAN (200ep)" or empty string
- CLI info: add "Vocoder" row in the detail table showing full stats or "(none)"

### 4. Nested sub-object vs flat fields
**Recommendation: Use nested `VocoderInfo` dataclass.**
- Vocoder data is all-or-nothing: if a vocoder is present, all fields exist; if not, none exist
- `vocoder: VocoderInfo | None` is a clean presence check
- Avoids polluting `ModelEntry` with 4+ flat fields that are all-or-nothing
- `dataclasses.asdict()` handles nested dataclasses automatically for JSON serialization
- Requires manual reconstruction in `_load_index()` (see Pitfall 5), but this is 2 lines of code

## Open Questions

1. **HiFi-GAN V2 architecture params for vocoder config**
   - What we know: Phase 16 will define the exact HiFi-GAN V2 architecture. The vocoder config dict in the saved state needs to capture all params needed to reconstruct the model.
   - What's unclear: The exact field names and values for HiFi-GAN V2 config (upsample_rates, kernel_sizes, etc.)
   - Recommendation: Define the vocoder state dict structure as `{"type": str, "state_dict": dict, "config": dict, "training_metadata": dict}`. The `config` dict is opaque to Phase 13 -- Phase 16 owns its content. Phase 13 just stores and retrieves it.

2. **Index version migration**
   - What we know: `_INDEX_VERSION` is currently 1. Adding `vocoder` field changes the schema.
   - What's unclear: Whether old indexes need migration or can just be regenerated.
   - Recommendation: Bump `_INDEX_VERSION` to 2. Since old `.distill` files are ignored, the entire catalog should be rebuilt anyway -- there are no `.distillgan` files yet. The `_load_index()` graceful handling (try/except per entry) handles any edge cases.

## Sources

### Primary (HIGH confidence)
- Source code inspection of `src/distill/models/persistence.py` -- complete save/load/delete pipeline
- Source code inspection of `src/distill/library/catalog.py` -- ModelEntry, ModelLibrary, index management
- Source code inspection of `src/distill/cli/generate.py` -- resolve_model, extension checks
- Source code inspection of `src/distill/cli/model.py` -- list/info/delete commands
- Source code inspection of `src/distill/ui/components/model_card.py` -- HTML card rendering
- Source code inspection of `src/distill/ui/tabs/library_tab.py` -- library tab with table/card views
- Source code inspection of `src/distill/training/loop.py` -- training loop with model save
- Source code inspection of `src/distill/training/checkpoint.py` -- checkpoint structure
- Source code inspection of `src/distill/vocoder/base.py` -- VocoderBase interface
- Source code inspection of `src/distill/vocoder/__init__.py` -- get_vocoder factory

### Secondary (MEDIUM confidence)
- `torch.save`/`torch.load` documentation -- handles arbitrary nested dicts with tensors natively, no special handling needed for adding vocoder state_dict

### Tertiary (LOW confidence)
None. All findings are based on direct source code inspection.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all existing patterns
- Architecture: HIGH -- modifying existing well-understood code with clear patterns
- Pitfalls: HIGH -- identified through direct code inspection; the sweep completeness pitfall is the primary risk

**Research date:** 2026-02-21
**Valid until:** Indefinite (internal codebase patterns, no external dependencies)
