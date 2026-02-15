# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-12)

**Core value:** Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters
**Current focus:** Phase 10 in progress -- Multi-format export and spatial audio (plan 03 of 5 complete)

## Current Position

Phase: 10 of 10 (Multi-Format Export & Spatial Audio)
Plan: 3 of 5 in current phase (10-03 complete)
Status: Plan 10-03 complete -- multi-model blending engine with latent-space and audio-domain modes
Last activity: 2026-02-15 -- Phase 10-03 executed (Multi-Model Blending)

Progress: [████████████████████] ~96%

## Performance Metrics

**Velocity:**
- Total plans completed: 31
- Average duration: 3 min
- Total execution time: 1.29 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| Phase 01 P01 | 3min | 2 tasks, 20 files | 3min |
| Phase 01 P02 | 3min | 2 tasks, 3 files | 3min |
| Phase 01 P03 | 3min | 2 tasks, 4 files | 3min |
| Phase 02 P01 | 2min | 2 tasks, 5 files | 2min |
| Phase 02 P02 | 3min | 2 tasks, 2 files | 3min |
| Phase 02 P03 | 2min | 2 tasks, 5 files | 2min |
| Phase 03 P01 | 3min | 2 tasks, 3 files | 3min |
| Phase 03 P02 | 3min | 2 tasks, 3 files | 3min |
| Phase 03 P03 | 2min | 2 tasks, 2 files | 2min |
| Phase 03 P04 | 4min | 3 tasks, 5 files | 4min |
| Phase 04 P01 | 2min | 2 tasks, 4 files | 2min |
| Phase 04 P02 | 2min | 2 tasks, 2 files | 2min |
| Phase 04 P03 | 3min | 2 tasks, 4 files | 3min |
| Phase 05 P01 | 3min | 2 tasks, 5 files | 3min |
| Phase 05 P02 | 3min | 2 tasks, 4 files | 3min |
| Phase 06 P01 | 4min | 3 tasks, 4 files | 4min |
| Phase 07 P01 | 2min | 2 tasks, 3 files | 2min |
| Phase 07 P02 | 3min | 2 tasks, 2 files | 3min |
| Phase 07 P03 | 2min | 2 tasks, 2 files | 2min |
| Phase 08 P01 | 3min | 2 tasks, 8 files | 3min |
| Phase 08 P02 | 4min | 2 tasks, 3 files | 4min |
| Phase 08 P03 | 3min | 2 tasks, 2 files | 3min |
| Phase 08 P04 | 4min | 2 tasks, 3 files | 4min |
| Phase 08 P05 | 4min | 2 tasks, 5 files | 4min |
| Phase 09 P01 | 2min | 2 tasks, 5 files | 2min |
| Phase 09 P02 | 2min | 2 tasks, 3 files | 2min |
| Phase 09 P03 | 2min | 2 tasks, 2 files | 2min |
| Phase 10 P01 | 3min | 2 tasks, 5 files | 3min |
| Phase 10 P02 | 4min | 2 tasks, 5 files | 4min |
| Phase 10 P03 | 3min | 2 tasks, 2 files | 3min |

**Recent Trend:**
- Last 5 plans: 2min, 2min, 3min, 4min, 3min (avg 2.8min)
- Trend: Consistent

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Phase 1: Non-real-time generation (quality over latency)
- Phase 1: Python + Gradio for v1 (fastest path to usable tool)
- Phase 4: 48kHz/24bit baseline (professional audio production standard)
- Phase 5: Novel approach emphasis (research-grade innovation, not just wrapper)
- [Phase 01]: Used dependency-groups.dev instead of deprecated tool.uv.dev-dependencies
- [Phase 01]: TOML config uses 0/0.0 for unset numeric fields (TOML has no null type)
- [Phase 01]: Config module has zero PyTorch dependency for error reporting resilience
- [Phase 01]: Deep merge strategy for config forward compatibility
- [Phase 01]: Explicit CUDA -> MPS -> CPU detection chain for predictability (not torch.accelerator)
- [Phase 01]: Smoke test on non-CPU devices catches "available but broken" GPUs
- [Phase 01]: OOM recovery outside except block to release frame references
- [Phase 01]: CPU benchmark returns default 32 (skip slow binary search)
- [Phase 01]: MPS memory via psutil unified memory + torch.mps.current_allocated_memory()
- [Phase 01]: packaging.version for PyTorch version comparison (handles +cu128 suffixes)
- [Phase 01]: Path validation returns warnings not errors (auto-created on first run)
- [Phase 01]: First-run stores detected device type in config for auto-selection
- [Phase 02]: soundfile for audio I/O (avoids TorchCodec/FFmpeg dependency broken on macOS)
- [Phase 02]: Cache torchaudio.transforms.Resample instances per (orig_freq, new_freq) pair
- [Phase 02]: Validation collects issues without raising -- Phase 1 error-collection pattern
- [Phase 02]: collect_audio_files skips hidden files/dirs for clean dataset import
- [Phase 02]: Pre-create SpeedPerturbation/AddNoise at init; PitchShift/Vol per-call (varying params)
- [Phase 02]: PitchShift n_fft=2048 for 48kHz to avoid bass artifacts
- [Phase 02]: Independent probability gating per augmentation (not all-or-nothing)
- [Phase 02]: expand_dataset preserves unaugmented originals alongside augmented copies
- [Phase 02]: preprocess_dataset skips corrupt files with warning (per-file try/except)
- [Phase 02]: Dataset class stores only metadata -- waveforms deferred to preprocessing/training
- [Phase 02]: Thumbnail mtime-based caching avoids redundant regeneration
- [Phase 02]: matplotlib.use('Agg') before pyplot import for headless compatibility
- [Phase 02]: Per-file try/except in batch thumbnail generation -- one failure does not stop batch
- [Phase 03]: Lazy linear init for encoder/decoder to handle variable mel time dimensions
- [Phase 03]: Sigmoid decoder output (log1p-normalized mel is always >= 0)
- [Phase 03]: Pad both height and width to multiple of 16 for robustness
- [Phase 03]: Preview interval threshold at 50 epochs (short runs get preview every 2 epochs)
- [Phase 03]: Dataset returns raw waveforms -- mel conversion on GPU in training loop
- [Phase 03]: Chunk index built from metadata only (no waveform load during indexing)
- [Phase 03]: Augmentation per-chunk with 50% probability for training variety
- [Phase 03]: Fixed split seed (42) for reproducible train/val partitions
- [Phase 03]: JSON sidecar (.meta.json) alongside .pt checkpoints for fast scanning without loading full state dicts
- [Phase 03]: Retention policy ceiling of 4 (3 recent + 1 best); fewer when best overlaps recent
- [Phase 03]: Peak normalization before WAV export prevents clipping from untrained decoder
- [Phase 03]: Reconstruction previews limited to 2 items per epoch to control disk usage
- [Phase 03]: NaN detection skips gradient update instead of crashing for MPS stability
- [Phase 03]: Cancel event saves checkpoint immediately (no wait for epoch boundary)
- [Phase 03]: Overfitting gap >20% triggers warning but continues training (user decides when to stop)
- [Phase 03]: Training loop emits step-level and epoch-level metrics via callback for UI decoupling
- [Phase 04]: 8th-order Butterworth with sosfiltfilt for zero-phase anti-aliasing (no phase distortion)
- [Phase 04]: 50ms (2400 samples at 48kHz) crossfade overlap with Hann window
- [Phase 04]: SLERP falls back to lerp when dot product > 0.9995 (near-parallel vectors)
- [Phase 04]: Chunks processed one at a time to limit memory for long generation (up to 60s)
- [Phase 04]: Peak normalize to -1 dBFS (0.891) not 1.0 for professional headroom
- [Phase 04]: Stereo width parameter continuous 0.0-1.5 with clamping and warning
- [Phase 04]: SNR frame-based (10ms) with RMS > 0.01 silence threshold
- [Phase 04]: Vectorised clipping consecutive-run detection (np.diff, not Python loop)
- [Phase 04]: Traffic light quality thresholds: green >30dB no-clip, yellow 15-30dB, red <15dB or >0.1% clip
- [Phase 04]: All internal processing at 48kHz; resample only at end via cached torchaudio.transforms.Resample
- [Phase 04]: Sidecar JSON written before WAV in export pipeline (research pitfall #6)
- [Phase 04]: Dual-seed stereo uses seed and seed+1 for deterministic L/R channel generation
- [Phase 04]: Auto-generated export filenames: gen_{timestamp}_seed{seed}.wav
- [Phase 05]: 2% variance threshold for active PCA components (conservative, filters below uniform baseline)
- [Phase 05]: 21 discrete slider steps giving integer range -10 to +10
- [Phase 05]: Pearson r with |r|>0.5 and p<0.05 dual gate for label suggestions
- [Phase 05]: numpy/scipy only for audio features (no librosa, avoids numba dependency)
- [Phase 05]: Store numpy arrays not sklearn PCA objects for checkpoint portability
- [Phase 05]: Integer step indices are ground truth; continuous values derived from step * step_size
- [Phase 05]: 0.1-scaled random perturbations for multi-chunk slider-controlled generation variety
- [Phase 05]: Serialization version field (v1) for future checkpoint migration compatibility
- [Phase 06]: JSON index (not SQLite) for model catalog -- sufficient for <1000 models, human-readable, zero-dependency
- [Phase 06]: Conditional encoder linear layer init on load -- handles both fully-trained and sample-only models
- [Phase 06]: Atomic write pattern (temp file + os.replace + .bak backup) for crash-safe JSON index
- [Phase 06]: Saved model format strips optimizer/scheduler state -- ~6 MB vs ~12 MB for checkpoints
- [Phase 06]: UUID model IDs for collision-free identification without counter management
- [Phase 06]: weights_only=False for torch.load -- our dicts contain numpy arrays and Python primitives
- [Phase 07]: Copied _write_index_atomic pattern locally in presets/manager.py (module independence from catalog.py)
- [Phase 07]: Virtual folders stored as list of dicts with name+created (not just strings) for future metadata
- [Phase 07]: Case-insensitive folder name duplicate check to prevent confusing near-duplicates
- [Phase 07]: Copied _write_index_atomic pattern locally in history/store.py (module independence from catalog.py)
- [Phase 07]: Smaller thumbnail dimensions (400x60) for history entries vs dataset thumbnails (800x120)
- [Phase 07]: repair_history reports orphan audio files but does NOT delete them (user may want them)
- [Phase 07]: ABComparison is ephemeral runtime state (not persisted) -- A/B UI state, not disk data
- [Phase 07]: keep_winner raises ValueError for live generation (must be in history before preset save)
- [Phase 07]: TYPE_CHECKING guard for cross-module imports avoids circular dependencies in history/comparison.py
- [Phase 08]: Module-level AppState singleton (not gr.State) for non-deepcopyable backend objects
- [Phase 08]: TYPE_CHECKING guards for all heavy imports (torch, backend classes) in ui/state.py
- [Phase 08]: create_app() calls init_state internally -- state ready before UI builds
- [Phase 08]: Duplicate filename avoidance with counter suffix during file import to datasets_dir
- [Phase 08]: gr.Timer(value=2, active=False) for training dashboard -- activated only during training
- [Phase 08]: 20 pre-created gr.Audio slots revealed progressively as previews arrive
- [Phase 08]: Preset dropdown auto-populates epochs/LR/advanced from _PRESET_DEFAULTS mapping
- [Phase 08]: MAX_SLIDERS=12 pre-created with dynamic visibility (Gradio cannot add/remove components at runtime)
- [Phase 08]: 3-column slider layout with keyword-based category assignment (timbral/temporal/spatial)
- [Phase 08]: Quality badge uses traffic light emoji for instant visual feedback
- [Phase 08]: Last generation result stored in metrics_buffer for export access
- [Phase 08]: Preset manager and history store initialized lazily on first model load
- [Phase 08]: Training callback stores events in app_state.metrics_buffer; Timer reads and builds chart
- [Phase 08]: Dropdown-based model selection paired with card grid (gr.HTML click events are limited)
- [Phase 08]: Library load handler reloads ModelLibrary from disk after delete/save for catalog consistency
- [Phase 08]: Cross-tab wiring via component dict returns from tab builders (load_btn.click chains to _update_sliders_for_model)
- [Phase 08]: History gallery uses (thumbnail_path, caption) tuples with seed+timestamp for informative browsing
- [Phase 08]: A/B dropdown choices use entry_id[:8] prefix for collision-free lookup from display string
- [Phase 08]: launch_ui/create_app accept optional config/device to avoid duplicate detection when called from CLI
- [Phase 08]: Accordion-based collapsible sections for secondary features (History, A/B) within Generate tab
- [Phase 09]: Typer with no_args_is_help=False and invoke_without_command=True callback for backward-compatible bare sda
- [Phase 09]: Module-level _cli_state dict to pass --device/--verbose/--config from callback to subcommands
- [Phase 09]: try/except ImportError for generate/train/model sub-typers so plan 01 works before plans 02/03
- [Phase 09]: resolve_model uses 3-tier lookup: .sda file path, UUID, then name search with ambiguity detection
- [Phase 09]: All CLI Rich console output to stderr; stdout reserved for file paths or JSON (enables piping)
- [Phase 09]: delete command uses persistence.delete_model (handles both file + index removal)
- [Phase 09]: Typer callback(invoke_without_command=True) for single-command sub-typer avoids nested command names
- [Phase 09]: Direct call to train() not TrainingRunner for CLI (no background thread needed)
- [Phase 09]: Exit code 3 for cancelled training, 1 for errors, 0 for success
- [Phase 09]: Removed all try/except ImportError guards in __init__.py now that all CLI modules exist
- [Phase 10]: OGG Vorbis quality default 0.6 (~192 kbps VBR) for good quality/size balance
- [Phase 10]: lameenc quality=2 (highest encoding quality) for MP3
- [Phase 10]: FLAC at PCM_24 subtype to match project's 24-bit professional standard
- [Phase 10]: WAV embed_metadata is no-op (sidecar JSON only, per Phase 4 pattern)
- [Phase 10]: Custom TXXX frames for SDA provenance in MP3 (SDA_SEED, SDA_MODEL, SDA_PRESET)
- [Phase 10]: Union sliders merged by PCA component index order (component 0 aligns across models)
- [Phase 10]: Zero-fill for models lacking a given PCA component (neutral/mean position)
- [Phase 10]: Latent-space blending validates matching latent_dim; audio-domain works universally
- [Phase 10]: Single-model fast path bypasses blending for efficiency
- [Phase 10]: Inactive models moved to CPU to free GPU memory (research pitfall #6)
- [Phase 10]: Default blend weight 25.0 per slot (equal share of 4 models at max capacity)
- [Phase 10]: sofar library for SOFA file loading (HRTF standard format)
- [Phase 10]: 1st-order Butterworth low-pass for binaural depth rolloff (20kHz->8kHz proportional to depth)
- [Phase 10]: Early reflection pattern for stereo depth effect (0-40ms delay, 0.15 mix level)
- [Phase 10]: Width control blends linearly between center and full binaural separation
- [Phase 10]: migrate_stereo_config maps old Phase 4 stereo_mode/stereo_width to new SpatialConfig

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-15
Stopped at: Completed 10-02-PLAN.md -- Spatial audio system with HRTF binaural and stereo/mono modes
Resume file: None
