---
phase: 15-generation-pipeline
plan: 01
subsystem: inference
tags: [autoregressive, sampling, generation, prior, vq-vae, top-k, top-p, temperature]

# Dependency graph
requires:
  - phase: 14-autoregressive-prior
    provides: CodePrior model, flatten_codes/unflatten_codes, extract_code_sequences
provides:
  - sample_code_sequence() autoregressive sampling with temperature/top-k/top-p
  - generate_audio_from_prior() end-to-end prior-to-audio pipeline
  - Public API exports for both functions
affects: [15-02, 15-03, 16-code-editor, cli, gradio-ui]

# Tech tracking
tech-stack:
  added: []
  patterns: [standalone-function-not-class-method, lazy-imports, crossfade-stitching]

key-files:
  created: []
  modified:
    - src/distill/models/prior.py
    - src/distill/inference/generation.py
    - src/distill/models/__init__.py
    - src/distill/inference/__init__.py

key-decisions:
  - "sample_code_sequence is module-level function, not CodePrior method"
  - "generate_audio_from_prior is standalone function, not GenerationPipeline method (v1.0 class untouched)"
  - "Each chunk gets unique seed (actual_seed + i) for variety across multi-chunk generation"

patterns-established:
  - "Prior-based generation as standalone functions alongside v1.0 GenerationPipeline class"
  - "crossfade_chunks for multi-chunk stitching in prior pipeline (consistent with v1.0)"

requirements-completed: [GEN-02, GEN-03, GEN-04]

# Metrics
duration: 3min
completed: 2026-02-27
---

# Phase 15 Plan 01: Core Sampling Engine Summary

**Autoregressive code sampling with temperature/top-k/top-p controls and end-to-end prior-to-audio generation pipeline with multi-chunk crossfade stitching**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-27T17:50:35Z
- **Completed:** 2026-02-27T17:54:01Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- sample_code_sequence() generates [1, seq_len] code tensors with temperature, top-k, and top-p sampling controls
- generate_audio_from_prior() provides complete prior-to-audio pipeline: sample codes, decode VQ-VAE, mel-to-waveform
- Multi-chunk generation (duration > 1s) uses crossfade_chunks() for seamless audio stitching
- Progress callback support for UI/CLI integration
- Public API exports from distill.models and distill.inference

## Task Commits

Each task was committed atomically:

1. **Task 1: Add sample_code_sequence() to prior.py** - `f64864d` (feat)
2. **Task 2: Create generate_audio_from_prior() with multi-chunk stitching and public API exports** - `3a5ad86` (feat)

## Files Created/Modified
- `src/distill/models/prior.py` - Added sample_code_sequence() with temperature/top-k/top-p autoregressive sampling
- `src/distill/inference/generation.py` - Added generate_audio_from_prior() end-to-end pipeline with multi-chunk support
- `src/distill/models/__init__.py` - Exported sample_code_sequence from public API
- `src/distill/inference/__init__.py` - Exported generate_audio_from_prior from public API

## Decisions Made
- sample_code_sequence is a module-level function (not a CodePrior method) per plan spec
- generate_audio_from_prior is a standalone function (not a GenerationPipeline method) to avoid modifying v1.0 class
- Each chunk receives seed = actual_seed + chunk_index for varied but reproducible multi-chunk generation
- Spatial shape computed from spectrogram config with padding awareness (pad_h/pad_w for 16x downsampling)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Core sampling engine and generation pipeline ready for CLI and UI integration (Plans 02 and 03)
- generate_audio_from_prior() is the backend that CLI `generate` command and Gradio UI will call
- All key integration points wired: sample_code_sequence -> codes_to_embeddings -> decode -> mel_to_waveform -> crossfade_chunks

## Self-Check: PASSED

- All 4 modified files exist on disk
- Both task commits found in git history (f64864d, 3a5ad86)
- All imports verified working

---
*Phase: 15-generation-pipeline*
*Completed: 2026-02-27*
