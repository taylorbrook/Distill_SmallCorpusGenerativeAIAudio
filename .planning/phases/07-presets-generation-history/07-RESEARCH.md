# Phase 7: Presets & Generation History - Research

**Researched:** 2026-02-13
**Domain:** Preset management, generation history indexing, waveform thumbnails, A/B comparison data layer
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Preset organization
- Presets are model-specific -- each preset belongs to one model (PCA axes differ per model)
- User-defined folders for organizing presets (e.g., "Pads", "Textures", "Percussion")
- Presets store slider positions + seed (no duration -- duration is set per-generation)
- No audio preview on save -- presets are parameter snapshots only, user loads and generates to hear

#### History browsing
- Reverse-chronological list, most recent first
- Each entry shows: waveform thumbnail, preset name used (or "custom"), and timestamp
- Expand/click for full parameter details (slider values, seed, model, duration)
- Unlimited history retention -- user manually deletes entries they don't want
- History entries store both the audio file (WAV) and full parameter snapshot -- instant replay plus exact settings

#### A/B comparison
- Toggle A/B button -- single play control, toggle switches between A and B at the same playback position
- Audio-only comparison -- no visual parameter diff (parameters visible in each entry's details separately)
- Default mode: compare current/latest generation against one history entry
- Also supports picking any two entries from history to compare
- "Keep this one" action after comparing -- saves the winner's parameters as a preset

### Claude's Discretion
- Preset file format and storage structure
- History index implementation
- Waveform thumbnail generation approach for history entries
- Folder management UX details (create, rename, delete folders)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Summary

Phase 7 builds the data layer and API for three interconnected features: preset management, generation history, and A/B comparison. This phase does NOT build the Gradio UI (that is Phase 8) -- it builds the storage, indexing, and API functions that the UI will consume. The approach follows established project patterns exactly: JSON indexes with atomic writes, UUID-based IDs, dataclasses for structured data, and file-based storage in the project data directory.

The core challenge is designing clean data structures that capture the relationships between models, presets, and history entries while keeping the storage format simple and human-readable. Presets are model-specific (because PCA axes differ per model), so presets must reference their parent model by model_id. History entries combine a WAV file on disk with a parameter snapshot in the history index, creating a two-source-of-truth situation that needs the same consistency-check pattern used in Phase 6's model library. The A/B comparison feature is entirely stateless at the data layer -- it is a runtime UI concept that references two history entries and provides a "keep this one" action that delegates to the existing preset-save API.

The existing codebase provides strong foundations: `library/catalog.py` establishes the JSON index + atomic write pattern; `audio/thumbnails.py` provides waveform thumbnail generation from numpy arrays; `controls/mapping.py` defines `SliderState` (integer positions) which is the exact data presets store; `inference/generation.py` has `GenerationResult` containing all metadata needed for history entries; and `inference/export.py` already writes sidecar JSON alongside WAV files. Phase 7 extends these patterns into two new modules: `presets/` for preset CRUD + folder management, and `history/` for generation history indexing + A/B comparison state management.

**Primary recommendation:** Create a `presets/` subpackage with `PresetEntry` dataclass and `PresetManager` class (JSON index per model, folder support), and a `history/` subpackage with `HistoryEntry` dataclass and `GenerationHistory` class (single JSON index, WAV file references, waveform thumbnail generation on save). A/B comparison is a lightweight `ABComparison` dataclass holding two history entry IDs with a `keep_winner()` method that delegates to preset save.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| json (stdlib) | N/A | Preset and history JSON indexes | Same pattern as Phase 6 model library; zero-dependency, human-readable |
| pathlib (stdlib) | N/A | File path management | Already used throughout project |
| uuid (stdlib) | N/A | Unique preset and history entry IDs | Same pattern as Phase 6 model_id; collision-free |
| datetime (stdlib) | N/A | Timestamps for presets and history | ISO 8601 format matching existing metadata patterns |
| dataclasses (stdlib) | N/A | Structured data for PresetEntry, HistoryEntry | Project-wide pattern for all domain objects |
| shutil (stdlib) | N/A | Folder rename/delete, file copy | Reliable cross-platform operations |
| os (stdlib) | N/A | Atomic write (os.replace), file size | Same atomic write pattern as library/catalog.py |
| tempfile (stdlib) | N/A | Atomic write temp files | Same pattern as library/catalog.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=1.26 | Waveform array handling for thumbnail generation | Already in project; used in audio/thumbnails.py |
| matplotlib | >=3.8 | Waveform thumbnail rendering | Already in project; used in audio/thumbnails.py |
| soundfile | >=0.12 | WAV file I/O for history playback | Already in project; used in inference/export.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| JSON preset index per model | Single global JSON index | Per-model files are self-contained, easily shared/backed up, and naturally scope presets to models |
| JSON history index | SQLite database | Overkill for browsing a few hundred to low thousands of entries; JSON keeps parity with model library pattern |
| PNG waveform thumbnails | Binary waveform data in index | PNG is cacheable, viewable externally, and already implemented in audio/thumbnails.py |
| File-based folders (subdirectories) | Virtual folders (folder field in JSON) | Virtual folders in JSON are simpler -- no filesystem operations, folder rename is just a string update, presets can be flat files with a folder tag |

**Installation:**
```bash
# No new dependencies needed -- all stdlib + existing project dependencies
```

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── presets/
│   ├── __init__.py          # [NEW] Public API exports
│   └── manager.py           # [NEW] PresetEntry, PresetFolder, PresetManager
├── history/
│   ├── __init__.py          # [NEW] Public API exports
│   ├── store.py             # [NEW] HistoryEntry, GenerationHistory
│   └── comparison.py        # [NEW] ABComparison, keep_winner
├── audio/
│   └── thumbnails.py        # [EXISTING] generate_waveform_thumbnail (reuse)
├── controls/
│   └── mapping.py           # [EXISTING] SliderState (presets store this)
├── inference/
│   └── generation.py        # [EXISTING] GenerationResult, GenerationPipeline (history captures this)
├── library/
│   └── catalog.py           # [EXISTING] ModelLibrary pattern to follow
├── config/
│   └── defaults.py          # [MODIFY] Add paths.presets, paths.history
```

### Data Directory Layout
```
data/
├── models/                  # [EXISTING] .sda files + model_library.json
├── generated/               # [EXISTING] exported WAV files + sidecar JSON
├── presets/                  # [NEW] preset storage
│   ├── {model_id}/          # One directory per model
│   │   └── presets.json     # Preset index for this model
│   └── ...
└── history/                 # [NEW] generation history
    ├── history.json         # History index
    ├── audio/               # WAV files for history entries
    │   ├── {entry_id}.wav
    │   └── ...
    └── thumbnails/          # Waveform thumbnail PNGs
        ├── {entry_id}.png
        └── ...
```

### Pattern 1: Preset Storage (JSON Index per Model)
**What:** Each model gets its own `presets.json` file inside a `{model_id}/` directory under the presets root. This naturally scopes presets to models (locked decision: presets are model-specific).
**When to use:** Every preset CRUD operation.
**Example:**
```python
# Source: Follows library/catalog.py JSON index pattern
# data/presets/{model_id}/presets.json structure:
{
    "version": 1,
    "model_id": "abc123-uuid",
    "folders": [
        {"name": "Pads", "created": "2026-02-13T14:00:00Z"},
        {"name": "Textures", "created": "2026-02-13T14:05:00Z"}
    ],
    "presets": {
        "preset-uuid-1": {
            "preset_id": "preset-uuid-1",
            "name": "Warm Pad",
            "folder": "Pads",
            "slider_positions": [3, -2, 5, 0, 1, -4, 2, 0],
            "n_components": 8,
            "seed": 42,
            "created": "2026-02-13T14:30:00Z",
            "modified": "2026-02-13T15:00:00Z",
            "description": ""
        }
    }
}
```

### Pattern 2: Preset Entry Dataclass
**What:** A dataclass capturing the exact slider state and seed for a model-specific preset.
**When to use:** Core data structure for all preset operations.
**Example:**
```python
# Source: Follows project dataclass pattern
from dataclasses import dataclass, field

@dataclass
class PresetEntry:
    """A named slider configuration preset for a specific model."""
    preset_id: str
    name: str
    folder: str  # empty string = root (no folder)
    slider_positions: list[int]  # matches SliderState.positions
    n_components: int  # matches SliderState.n_components
    seed: int | None  # None = no seed preference
    created: str  # ISO 8601
    modified: str  # ISO 8601
    description: str = ""
```

### Pattern 3: History Entry with WAV Reference
**What:** A dataclass capturing the full parameter snapshot plus a reference to the WAV file and thumbnail on disk.
**When to use:** Every time a generation is saved to history.
**Example:**
```python
# Source: Follows model library entry pattern
@dataclass
class HistoryEntry:
    """A single generation in the history log."""
    entry_id: str  # UUID
    timestamp: str  # ISO 8601

    # Model reference
    model_id: str
    model_name: str

    # Generation parameters (full snapshot)
    slider_positions: list[int] | None  # None if generated without sliders
    n_components: int
    seed: int
    duration_s: float
    sample_rate: int
    stereo_mode: str
    preset_name: str  # "custom" if no preset was used

    # File references (relative to history dir)
    audio_file: str  # e.g., "audio/{entry_id}.wav"
    thumbnail_file: str  # e.g., "thumbnails/{entry_id}.png"

    # Quality metadata
    quality_score: dict = field(default_factory=dict)

    # Latent vector stored as list for JSON serialization
    latent_vector: list[float] | None = None
```

### Pattern 4: Virtual Folders for Presets
**What:** Folders are stored as a list of names in the JSON index, not as filesystem directories. Each preset has a `folder` field (empty string for root). Renaming a folder is a bulk string update on all presets in that folder, not a filesystem operation.
**When to use:** Folder CRUD operations.
**Why virtual:** Avoids filesystem edge cases (reserved names, path separators in folder names, permissions). Folder operations are instant (no file moves). Simpler consistency model.
**Example:**
```python
class PresetManager:
    """Manage presets for a specific model."""

    def create_folder(self, name: str) -> None:
        """Create a new folder. Raises ValueError if name already exists."""
        ...

    def rename_folder(self, old_name: str, new_name: str) -> int:
        """Rename a folder. Returns count of presets updated.
        Raises ValueError if old_name not found or new_name exists.
        """
        # Update all presets with folder == old_name
        # Update the folders list
        ...

    def delete_folder(self, name: str, move_presets_to: str = "") -> int:
        """Delete a folder. Moves its presets to move_presets_to (default: root).
        Returns count of presets moved. Raises ValueError if name not found.
        """
        ...

    def list_folders(self) -> list[str]:
        """Return sorted list of folder names."""
        ...
```

### Pattern 5: History Captures GenerationResult
**What:** When a generation completes, the history module receives the `GenerationResult` plus contextual info (model_id, model_name, preset_name, slider_positions) and creates a `HistoryEntry` with WAV file + thumbnail.
**When to use:** After every successful generation (called from the UI layer or API).
**Example:**
```python
# Source: Follows GenerationPipeline.export pattern
def add_to_history(
    result: "GenerationResult",
    model_id: str,
    model_name: str,
    slider_positions: list[int] | None,
    n_components: int,
    preset_name: str,
    history_dir: Path,
) -> HistoryEntry:
    """Save a generation result to history.

    Steps:
    1. Generate entry_id (UUID)
    2. Save WAV to history_dir/audio/{entry_id}.wav
    3. Generate waveform thumbnail to history_dir/thumbnails/{entry_id}.png
    4. Create HistoryEntry with full parameter snapshot
    5. Add entry to history index (JSON)
    6. Return the new entry
    """
    ...
```

### Pattern 6: A/B Comparison as Runtime State
**What:** A/B comparison is not a persisted data structure -- it is a runtime state object holding references to two history entries (or one history entry + the current live generation). The only persistent action is "keep this one" which saves the winner's parameters as a preset.
**When to use:** When user initiates A/B comparison in the UI.
**Example:**
```python
@dataclass
class ABComparison:
    """Runtime state for A/B comparison."""
    entry_a_id: str | None  # History entry ID (None = current/live generation)
    entry_b_id: str | None  # History entry ID
    active_side: str  # "a" or "b"

    def get_audio_paths(
        self, history: "GenerationHistory"
    ) -> tuple[Path | None, Path | None]:
        """Return (path_a, path_b) for audio playback.
        None for a path means use the current live audio buffer.
        """
        ...

    def keep_winner(
        self,
        winner: str,  # "a" or "b"
        preset_name: str,
        history: "GenerationHistory",
        preset_manager: "PresetManager",
    ) -> "PresetEntry":
        """Save the winner's parameters as a preset."""
        entry_id = self.entry_a_id if winner == "a" else self.entry_b_id
        entry = history.get(entry_id)
        return preset_manager.save_preset(
            name=preset_name,
            slider_positions=entry.slider_positions,
            n_components=entry.n_components,
            seed=entry.seed,
        )
```

### Anti-Patterns to Avoid
- **Storing presets globally (not per model):** PCA axes differ per model. A preset's slider_positions are meaningless for a different model. Every preset MUST be scoped to a model_id.
- **Storing duration in presets:** Locked decision -- duration is set per-generation, not saved in presets. Presets are parameter snapshots (slider positions + seed) only.
- **Generating audio on preset save:** Locked decision -- no audio preview on save. Presets are pure parameter snapshots.
- **Making A/B comparison persistent:** A/B state is ephemeral UI state. Persisting it adds complexity with no user value.
- **Storing waveform data in JSON index:** Waveform data is large. Store as separate PNG files (thumbnails) and WAV files (audio). JSON index only has file path references.
- **Filesystem directories for preset folders:** Virtual folders (string field in JSON) are simpler and avoid OS-specific edge cases with directory naming.
- **Breaking the atomic write pattern:** All JSON index writes MUST use the temp file + os.replace pattern from library/catalog.py. Direct json.dump to the target file risks corruption.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Waveform thumbnail generation | Custom image rendering | `audio.thumbnails.generate_waveform_thumbnail()` | Already built in Phase 2; handles Agg backend, memory cleanup, downsampling |
| Atomic JSON writes | Custom file locking | `_write_index_atomic()` pattern from `library/catalog.py` | Proven pattern; temp file + os.replace is POSIX-atomic |
| Unique IDs | Auto-increment counters | `uuid.uuid4()` | Same pattern as model_id; collision-free |
| WAV file I/O | Custom binary writing | `inference.export.export_wav()` | Already handles sample rate, bit depth, soundfile format |
| Slider position validation | Custom range checking | `SliderState` from `controls.mapping` | Already defines the canonical slider data structure |
| Filename sanitization | Custom regex | `_sanitize_filename()` from `models.persistence` | Already handles special chars, truncation, fallback |

**Key insight:** Phase 7 is a data management and API design problem, not a signal processing or ML problem. Every heavy operation (thumbnail generation, WAV export, slider state management) is already implemented. Phase 7 wraps these existing capabilities in preset and history management layers.

## Common Pitfalls

### Pitfall 1: Preset-Model ID Mismatch on Load
**What goes wrong:** User loads a preset but the active model has changed. The preset's slider_positions has a different n_components than the current model's analysis.
**Why it happens:** Models have different numbers of active PCA components. A preset saved with 8 components makes no sense for a model with 5.
**How to avoid:** `PresetManager` is always scoped to a specific model_id. The load_preset API validates that the preset's n_components matches the current analysis's n_active_components. If mismatched, raise ValueError with a clear message: "Preset '{name}' was created for model '{model_name}' with {n} components, but current model has {m} components."
**Warning signs:** Sliders producing unexpected sound; IndexError on slider_positions access.

### Pitfall 2: History Audio File Deleted Externally
**What goes wrong:** History index references a WAV file that no longer exists on disk (user deleted it manually, or disk cleanup).
**Why it happens:** Two sources of truth (JSON index + files on disk) can diverge. Same pattern as model library pitfall #3.
**How to avoid:** On history load, optionally run a lightweight consistency check (same as ModelLibrary.repair_index). Mark entries with missing audio as "audio unavailable" rather than deleting the entry (parameters are still useful). Provide a `repair_history()` method.
**Warning signs:** FileNotFoundError when trying to play history entry; broken thumbnail display.

### Pitfall 3: Large History Index from Unlimited Retention
**What goes wrong:** With unlimited retention (locked decision), the history.json file grows unbounded. After thousands of generations, JSON parse time becomes noticeable.
**Why it happens:** Each history entry adds ~500 bytes to the JSON index (parameters, metadata, file paths). At 10,000 entries, that is ~5 MB of JSON.
**How to avoid:** Document the scaling expectation. At typical usage (a few dozen generations per session), even years of use would produce at most a few thousand entries. JSON handles this fine. If performance becomes an issue, the `version` field enables future migration to SQLite. For v1, this is acceptable.
**Warning signs:** Slow history load time (>1s); large history.json file (>10 MB).

### Pitfall 4: Thumbnail Generation Blocking the UI
**What goes wrong:** Generating a waveform thumbnail from the audio takes noticeable time (~100-300ms for matplotlib rendering), blocking the generation-complete callback.
**Why it happens:** matplotlib figure creation and PNG rendering is CPU-intensive.
**How to avoid:** Generate thumbnails synchronously but use the existing `generate_waveform_thumbnail()` which is already optimized (downsamples to 2x display width, uses Agg backend). For v1, 100-300ms latency on generation complete is acceptable. If needed, Phase 8 can add async thumbnail generation.
**Warning signs:** Noticeable lag after generation completes before history entry appears.

### Pitfall 5: Seed None vs Seed 0 Confusion
**What goes wrong:** A preset with `seed=None` (no preference) is serialized as `null` in JSON, but `seed=0` means "use seed 0 specifically". If deserialization converts `null` to `0`, the preset loses its "random seed" intent.
**Why it happens:** JSON null and integer 0 are different, but Python's `int | None` requires careful handling.
**How to avoid:** Explicitly handle `None` in serialization: store as JSON `null`, deserialize back to Python `None`. In `PresetEntry`, `seed: int | None` where `None` means "generate random seed on each use" and any integer means "use this exact seed."
**Warning signs:** Presets always generating the same output when they should vary; presets generating different output when they should be deterministic.

### Pitfall 6: Folder Name Collisions and Empty Names
**What goes wrong:** User creates a folder with the same name as an existing folder, or creates a folder with an empty/whitespace-only name.
**Why it happens:** No validation on folder creation.
**How to avoid:** Validate folder names: strip whitespace, reject empty, reject duplicates (case-insensitive comparison). Folder names should be treated as display strings, not filesystem paths, so most characters are fine -- just reject empty and enforce uniqueness.
**Warning signs:** Duplicate folder entries in the folders list; presets in "wrong" folder due to name collision.

### Pitfall 7: History Entry Created Before WAV Write Completes
**What goes wrong:** History index is updated with entry, but WAV file write fails (disk full, permissions). Index references a file that does not exist.
**Why it happens:** Writing WAV first, then index, leaves a window where the WAV exists but the index does not (orphan file -- acceptable). Writing index first, then WAV, leaves a window where the index references a missing file (broken entry -- bad).
**How to avoid:** Write WAV file and thumbnail FIRST, then add the entry to the history index. Same ordering principle as sidecar JSON (Phase 4 decision: write metadata before audio). If WAV write fails, no index entry is created.
**Warning signs:** History entry shows but audio cannot play.

## Code Examples

Verified patterns from the existing codebase, extended for Phase 7:

### PresetEntry Dataclass
```python
# Source: Follows library/catalog.py ModelEntry pattern
from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class PresetEntry:
    """A named slider configuration preset for a specific model.

    Stores integer slider positions (matching SliderState.positions)
    plus an optional seed.  Duration is NOT stored (locked decision:
    duration is set per-generation).
    """

    preset_id: str
    name: str
    folder: str = ""  # empty = root (no folder)
    slider_positions: list[int] = field(default_factory=list)
    n_components: int = 0
    seed: int | None = None  # None = random seed each time
    created: str = ""  # ISO 8601
    modified: str = ""  # ISO 8601
    description: str = ""
```

### PresetManager Initialization
```python
# Source: Follows ModelLibrary pattern from library/catalog.py
class PresetManager:
    """Manage presets for a specific model.

    Each model has its own presets directory and JSON index.
    """

    def __init__(self, presets_dir: Path, model_id: str) -> None:
        self.presets_dir = Path(presets_dir)
        self.model_id = model_id
        self._model_dir = self.presets_dir / model_id
        self._index_path = self._model_dir / "presets.json"
        self._entries: dict[str, PresetEntry] = {}
        self._folders: list[dict] = []  # [{"name": ..., "created": ...}]
        self._load_index()

    def _load_index(self) -> None:
        """Load preset index from JSON. Handles missing/corrupt gracefully."""
        if not self._index_path.exists():
            self._entries = {}
            self._folders = []
            return
        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            # ... deserialize entries and folders
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load preset index: %s", exc)
            self._entries = {}
            self._folders = []
```

### Save Preset from SliderState
```python
# Source: Follows save_model pattern from models/persistence.py
def save_preset(
    self,
    name: str,
    slider_positions: list[int],
    n_components: int,
    seed: int | None = None,
    folder: str = "",
    description: str = "",
) -> PresetEntry:
    """Save a new preset.

    Parameters
    ----------
    name : str
        Display name for the preset.
    slider_positions : list[int]
        Integer step positions from SliderState.positions.
    n_components : int
        Number of active PCA components (from SliderState.n_components).
    seed : int | None
        Optional seed (None = random each time).
    folder : str
        Folder name (empty string = root).
    description : str
        Optional description.

    Returns
    -------
    PresetEntry
        The created preset entry.
    """
    now = datetime.now(timezone.utc).isoformat()
    entry = PresetEntry(
        preset_id=str(uuid.uuid4()),
        name=name,
        folder=folder,
        slider_positions=list(slider_positions),
        n_components=n_components,
        seed=seed,
        created=now,
        modified=now,
        description=description,
    )
    self._entries[entry.preset_id] = entry
    self._save_index()
    return entry
```

### Load Preset to SliderState
```python
# Source: Follows controls/mapping.py SliderState pattern
from small_dataset_audio.controls.mapping import SliderState

def load_preset(
    self,
    preset_id: str,
) -> tuple[SliderState, int | None]:
    """Load a preset and return (SliderState, seed).

    Returns
    -------
    tuple[SliderState, int | None]
        The slider state and optional seed.

    Raises
    ------
    KeyError
        If preset_id not found.
    """
    entry = self._entries.get(preset_id)
    if entry is None:
        raise KeyError(f"Preset not found: {preset_id}")

    slider_state = SliderState(
        positions=list(entry.slider_positions),
        n_components=entry.n_components,
    )
    return slider_state, entry.seed
```

### Add Generation to History
```python
# Source: Follows GenerationPipeline.export pattern
def add_to_history(
    self,
    result: "GenerationResult",
    model_id: str,
    model_name: str,
    slider_positions: list[int] | None,
    n_components: int,
    preset_name: str,
) -> HistoryEntry:
    """Record a generation in the history.

    Order of operations (critical for consistency):
    1. Save WAV file to history/audio/{entry_id}.wav
    2. Generate waveform thumbnail to history/thumbnails/{entry_id}.png
    3. Create HistoryEntry
    4. Add to history index (JSON with atomic write)
    """
    from small_dataset_audio.audio.thumbnails import generate_waveform_thumbnail
    from small_dataset_audio.inference.export import export_wav

    entry_id = str(uuid.uuid4())

    # 1. Save WAV file
    audio_dir = self.history_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_filename = f"{entry_id}.wav"
    audio_path = audio_dir / audio_filename
    export_wav(
        audio=result.audio,
        path=audio_path,
        sample_rate=result.sample_rate,
        bit_depth=result.config.bit_depth,
    )

    # 2. Generate waveform thumbnail
    thumb_dir = self.history_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_filename = f"{entry_id}.png"
    thumb_path = thumb_dir / thumb_filename
    generate_waveform_thumbnail(
        waveform=result.audio,
        output_path=thumb_path,
        width=400,  # smaller than dataset thumbnails
        height=60,
    )

    # 3. Create entry
    now = datetime.now(timezone.utc).isoformat()
    latent_list = None
    if result.config.latent_vector is not None:
        latent_list = result.config.latent_vector.tolist()

    entry = HistoryEntry(
        entry_id=entry_id,
        timestamp=now,
        model_id=model_id,
        model_name=model_name,
        slider_positions=list(slider_positions) if slider_positions else None,
        n_components=n_components,
        seed=result.seed_used,
        duration_s=result.duration_s,
        sample_rate=result.sample_rate,
        stereo_mode=result.config.stereo_mode,
        preset_name=preset_name,
        audio_file=f"audio/{audio_filename}",
        thumbnail_file=f"thumbnails/{thumb_filename}",
        quality_score=result.quality,
        latent_vector=latent_list,
    )

    # 4. Add to index (atomic write)
    self._entries[entry_id] = entry
    self._save_index()

    return entry
```

### Delete History Entry with File Cleanup
```python
# Source: Follows delete_model pattern from models/persistence.py
def delete_entry(self, entry_id: str) -> bool:
    """Delete a history entry and its associated files.

    Deletes: WAV file, thumbnail PNG, and index entry.
    Order: files first, then index (same principle as WAV-before-index).

    Returns True if deleted, False if not found.
    """
    entry = self._entries.get(entry_id)
    if entry is None:
        return False

    # Delete audio file
    audio_path = self.history_dir / entry.audio_file
    if audio_path.exists():
        audio_path.unlink()

    # Delete thumbnail
    thumb_path = self.history_dir / entry.thumbnail_file
    if thumb_path.exists():
        thumb_path.unlink()

    # Remove from index
    del self._entries[entry_id]
    self._save_index()
    return True
```

### History Index JSON Structure
```python
# Source: Follows model_library.json structure from library/catalog.py
# data/history/history.json structure:
{
    "version": 1,
    "entries": {
        "entry-uuid-1": {
            "entry_id": "entry-uuid-1",
            "timestamp": "2026-02-13T16:00:00Z",
            "model_id": "model-uuid",
            "model_name": "My Ambient Model",
            "slider_positions": [3, -2, 5, 0, 1, -4, 2, 0],
            "n_components": 8,
            "seed": 42,
            "duration_s": 2.5,
            "sample_rate": 48000,
            "stereo_mode": "mono",
            "preset_name": "Warm Pad",
            "audio_file": "audio/entry-uuid-1.wav",
            "thumbnail_file": "thumbnails/entry-uuid-1.png",
            "quality_score": {"snr_db": 35.2, "clipping_pct": 0.0, "overall": "green"},
            "latent_vector": [0.123, -0.456, ...]
        }
    }
}
```

### Config Defaults Update
```python
# Source: Extends config/defaults.py
DEFAULT_CONFIG: dict = {
    "general": { ... },
    "paths": {
        "datasets": "data/datasets",
        "models": "data/models",
        "generated": "data/generated",
        "presets": "data/presets",     # NEW
        "history": "data/history",     # NEW
    },
    "hardware": { ... },
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Filesystem directories for preset categories | Virtual folders (string field in JSON) | Common in modern apps | Simpler; no OS edge cases; instant rename |
| Database-backed history (SQLite) | JSON index for local tools | Appropriate at this scale | Human-readable; zero-dependency; matches existing catalog pattern |
| Store audio bytes in history database | File references in JSON + WAV on disk | Standard practice | Keeps index small; WAV files playable externally |
| Separate config file per preset | Single JSON index per model | Standard for small-scale preset management | One file to read/write; atomic updates; bulk operations easy |

**Deprecated/outdated:**
- Pickle-based preset storage: Not human-readable, not portable. JSON is the standard for configuration data.
- Real-time A/B toggle via multiple audio streams: Unnecessary complexity for non-real-time tool. Single play control with position-synced toggle is simpler and sufficient.

## Open Questions

1. **History entry disk space accumulation**
   - What we know: Each history entry stores a WAV file (~200 KB/s at 48kHz mono, ~400 KB/s stereo). A 2.5s mono generation is ~500 KB. Thumbnail is ~5 KB.
   - What's unclear: Whether to display cumulative disk usage or provide bulk cleanup tools.
   - Recommendation: For v1, just implement per-entry delete. Add a `get_total_size()` method to GenerationHistory that sums file sizes for display. Defer bulk cleanup UI to Phase 8.

2. **Preset sharing between models with identical analysis**
   - What we know: Two models trained on the same data might have compatible PCA axes. Presets could theoretically transfer.
   - What's unclear: Whether the user would benefit from cross-model preset import.
   - Recommendation: Out of scope for v1. The model_id scoping is strict. A future "import preset from other model" feature could re-map positions if analyses are compatible. For now, presets are strictly model-specific per locked decision.

3. **History playback audio format**
   - What we know: History stores WAV files for instant replay. The generation may have been at various sample rates and bit depths.
   - What's unclear: Whether to standardize history WAV format or preserve the original generation format.
   - Recommendation: Preserve the original generation format (sample_rate, bit_depth from GenerationConfig). The HistoryEntry stores these values for the UI to handle playback correctly. No format conversion needed.

4. **Latent vector storage in history entries**
   - What we know: The latent_vector can be a 64-dimensional float32 array (~256 bytes as JSON list). Storing it enables exact regeneration.
   - What's unclear: Whether storing latent_vector in addition to slider_positions is redundant.
   - Recommendation: Store both. Slider positions + seed enable regeneration through the slider API. The raw latent_vector enables regeneration even if the slider-to-latent mapping implementation changes. The storage cost is negligible (~500 bytes per entry).

## Integration Points

### With Existing Modules

| Module | Integration | Direction |
|--------|-------------|-----------|
| `controls.mapping.SliderState` | Presets serialize/deserialize to/from SliderState | Preset -> SliderState on load |
| `controls.mapping.sliders_to_latent` | Convert loaded preset to latent vector for generation | Called after preset load |
| `inference.generation.GenerationResult` | History captures result metadata | GenerationResult -> HistoryEntry |
| `inference.export.export_wav` | History saves WAV files | Called during history add |
| `audio.thumbnails.generate_waveform_thumbnail` | History generates thumbnails | Called during history add |
| `models.persistence.ModelMetadata` | History references model_id and model_name | Read from loaded model metadata |
| `library.catalog._write_index_atomic` | All JSON index writes use this pattern | Copy pattern, do not import directly |
| `config.defaults.DEFAULT_CONFIG` | Add presets and history paths | Modify existing config |

### With Phase 8 (Gradio UI)

Phase 7 outputs a clean API that Phase 8 consumes:
- `PresetManager.save_preset()` / `load_preset()` / `list_presets()` / `delete_preset()` / `rename_preset()`
- `PresetManager.create_folder()` / `rename_folder()` / `delete_folder()` / `list_folders()`
- `GenerationHistory.add_to_history()` / `list_entries()` / `get()` / `delete_entry()`
- `ABComparison` dataclass with `get_audio_paths()` and `keep_winner()`

Phase 8 does NOT need to know about JSON files, atomic writes, or file paths -- it calls the API and gets back dataclasses.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `library/catalog.py` (JSON index + atomic write pattern), `models/persistence.py` (save/load/delete pattern, filename sanitization), `controls/mapping.py` (SliderState, slider position semantics), `inference/generation.py` (GenerationResult, GenerationConfig, GenerationPipeline.export), `inference/export.py` (export_wav, write_sidecar_json), `audio/thumbnails.py` (generate_waveform_thumbnail), `config/defaults.py` (path configuration)
- Phase 6 Research: JSON index design, atomic write pattern, UUID-based IDs, consistency check pattern (repair_index)

### Secondary (MEDIUM confidence)
- Standard software patterns: Virtual folders (tag-based categorization vs filesystem hierarchy), event sourcing for history logs, compositor pattern for A/B comparison

### Tertiary (LOW confidence)
- None -- this phase requires no external library research

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All stdlib or existing project deps; no new dependencies required
- Architecture: HIGH -- Direct extension of Phase 6 patterns (JSON index, atomic writes, UUID IDs, dataclasses) into preset and history domains
- Preset format: HIGH -- Simple JSON with slider_positions + seed; maps directly to SliderState
- History format: HIGH -- WAV files on disk + JSON index; follows model library pattern with file references
- A/B comparison: HIGH -- Lightweight runtime state; only persistent action is "keep winner" which delegates to preset save
- Pitfalls: HIGH -- Identified from Phase 6 experience (index/file sync, atomic writes) and domain analysis (model-preset mismatch, seed None vs 0)
- Integration: HIGH -- All integration points verified by reading existing module source code

**Research date:** 2026-02-13
**Valid until:** 2026-03-15 (stable domain; file I/O and data management are mature patterns)
