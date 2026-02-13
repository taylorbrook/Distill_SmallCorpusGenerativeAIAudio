# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-12)

**Core value:** Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters
**Current focus:** Phase 3 (Core Training Engine)

## Current Position

Phase: 3 of 10 (Core Training Engine)
Plan: 2 of 4 in current phase
Status: Executing phase 03
Last activity: 2026-02-13 — Completed 03-02-PLAN.md (training config, dataset, metrics)

Progress: [██████░░░░] ~27%

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 3 min
- Total execution time: 0.38 hours

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

**Recent Trend:**
- Last 5 plans: 3min, 2min, 3min, 3min, 3min
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

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-13
Stopped at: Completed 03-02-PLAN.md
Resume file: None
