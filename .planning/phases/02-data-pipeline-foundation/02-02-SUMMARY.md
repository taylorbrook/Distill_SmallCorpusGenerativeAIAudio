---
phase: 02-data-pipeline-foundation
plan: 02
subsystem: audio
tags: [torchaudio, augmentation, preprocessing, pitch-shift, speed-perturbation, noise-injection, caching, dataclass]

# Dependency graph
requires:
  - phase: 02-data-pipeline-foundation
    plan: 01
    provides: "Audio I/O abstraction (load_audio, AudioFile), soundfile-based loading, lazy torch imports"
provides:
  - "AugmentationPipeline with 4 composable transforms (pitch, speed, noise, volume)"
  - "AugmentationConfig dataclass with per-augmentation probabilities"
  - "expand_dataset for small dataset expansion (default 10x)"
  - "preprocess_for_training for single-file resampling and normalization"
  - "preprocess_dataset for batch preprocessing with augmentation and caching"
  - "load_cached_dataset and clear_cache for .pt tensor caching"
  - "PreprocessingConfig dataclass with augmentation and caching options"
affects: [02-03-PLAN, training-engine, dataset-module, phase-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [composable-augmentation-pipeline, independent-probability-gating, tensor-caching, progress-callback, pre-created-transforms]

key-files:
  created:
    - src/small_dataset_audio/audio/augmentation.py
    - src/small_dataset_audio/audio/preprocessing.py

key-decisions:
  - "Pre-create SpeedPerturbation and AddNoise at init for reuse; PitchShift and Vol created per-call (varying parameters)"
  - "PitchShift n_fft=2048 for 48kHz audio to avoid bass artifacts"
  - "Each augmentation gated by independent probability (not all-or-nothing)"
  - "expand_dataset always preserves unaugmented originals alongside augmented copies"
  - "preprocess_dataset skips corrupt files with warning instead of failing the batch"

patterns-established:
  - "Augmentation pipeline: composable transforms with independent per-augmentation probability gating"
  - "Dataset expansion: original + N augmented copies per input waveform"
  - "Tensor caching: save preprocessed tensors as .pt files for fast training restarts"
  - "Progress callback: optional callback(current, total, filename) for UI integration"

# Metrics
duration: 3min
completed: 2026-02-12
---

# Phase 2 Plan 2: Data Augmentation and Preprocessing Summary

**Composable augmentation pipeline (pitch shift, speed perturbation, noise injection, volume) with 10x dataset expansion and preprocessing with .pt tensor caching**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-12T23:19:53Z
- **Completed:** 2026-02-12T23:23:21Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- AugmentationPipeline applies pitch shift, speed perturbation, noise injection, and volume variation with independent per-augmentation probabilities (0.3-0.5)
- expand_dataset produces original + N augmented copies per input, expanding small datasets by default 10x
- preprocess_for_training resamples to 48kHz and peak-normalizes single files
- preprocess_dataset handles batch preprocessing with optional augmentation, .pt caching, and progress callbacks
- Cache utilities (load_cached_dataset, clear_cache) for fast training restarts

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement composable augmentation pipeline** - `bd49906` (feat)
2. **Task 2: Implement preprocessing pipeline with caching** - `d18b708` (feat)

## Files Created/Modified
- `src/small_dataset_audio/audio/augmentation.py` - AugmentationPipeline with 4 transforms, AugmentationConfig, expand_dataset
- `src/small_dataset_audio/audio/preprocessing.py` - preprocess_for_training, preprocess_dataset, load_cached_dataset, clear_cache, PreprocessingConfig

## Decisions Made
- Pre-create SpeedPerturbation and AddNoise at init for reuse; PitchShift created per-call (n_steps varies), Vol created per-call (gain varies, but lightweight)
- PitchShift uses n_fft=2048 for 48kHz audio (avoids bass artifacts per research pitfall #6)
- Each augmentation gated independently (random.random() < probability), not all-or-nothing
- expand_dataset always preserves unaugmented originals: output = original + expansion_ratio augmented copies
- preprocess_dataset wraps each file load in try/except -- corrupt files skipped with warning, valid files continue
- load_cached_dataset uses weights_only=True in torch.load for security best practice

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- augmentation.py ready for import by preprocessing.py and future training engine
- preprocessing.py ready for dataset import workflow and Phase 3 training integration
- Progress callback API supports future Gradio UI integration (Phase 8)
- .pt caching eliminates repeated preprocessing during training iterations
- Exports not yet added to audio/__init__.py (will be done when all Plan 02/03 modules are finalized)

## Self-Check: PASSED

All files verified present. All commit hashes verified in git log.

---
*Phase: 02-data-pipeline-foundation*
*Completed: 2026-02-12*
