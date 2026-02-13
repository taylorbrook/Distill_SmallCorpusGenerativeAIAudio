# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-12)

**Core value:** Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters
**Current focus:** Phase 5 (Musically Meaningful Controls)

## Current Position

Phase: 5 of 10 (Musically Meaningful Controls)
Plan: 2 of 2 in current phase (PHASE COMPLETE)
Status: Phase 05 Complete
Last activity: 2026-02-13 — Completed 05-02-PLAN.md (slider mapping and generation integration)

Progress: [█████████░] ~50%

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: 3 min
- Total execution time: 0.65 hours

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

**Recent Trend:**
- Last 5 plans: 2min, 3min, 3min, 3min (avg 2.75min)
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

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-13
Stopped at: Completed 05-02-PLAN.md (slider mapping and generation integration) -- Phase 05 complete
Resume file: None
