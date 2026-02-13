---
phase: 03-core-training-engine
plan: 02
subsystem: training
tags: [pytorch, dataclass, config, dataset, metrics, overfitting, vae]

# Dependency graph
requires:
  - phase: 02-data-pipeline
    provides: "audio.io (load_audio, get_metadata), audio.augmentation (AugmentationPipeline)"
provides:
  - "TrainingConfig with 3 overfitting presets and adaptive dataset-size configuration"
  - "AudioTrainingDataset (PyTorch Dataset) with fixed-length 1s chunking"
  - "create_data_loaders with file-level validation split (no data leakage)"
  - "MetricsHistory with loss curves, overfitting detection, ETA, serialisation"
  - "MetricsCallback type alias for decoupled event subscription"
  - "StepMetrics, EpochMetrics, PreviewEvent, TrainingCompleteEvent event types"
affects: [03-03 (model architecture), 03-04 (training loop), 06-gradio-ui]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Adaptive config selection: preset/split/batch auto-sized to file count"
    - "File-level train/val split with fixed seed for reproducibility"
    - "Callback-based metrics decoupling (training loop emits, UI subscribes)"
    - "Lazy chunk indexing: pre-scan metadata at init, load audio on __getitem__"

key-files:
  created:
    - "src/small_dataset_audio/training/config.py"
    - "src/small_dataset_audio/training/dataset.py"
    - "src/small_dataset_audio/training/metrics.py"
  modified: []

key-decisions:
  - "Preview interval threshold at 50 epochs: short runs (<50) get preview every 2 epochs, standard runs every 5"
  - "Dataset returns raw waveforms (not mel spectrograms) -- mel conversion on GPU in training loop for efficiency"
  - "Chunk index built from metadata at construction time -- no full audio load during indexing"
  - "Augmentation applied per-chunk with 50% probability (not per-file) for better training variety"
  - "Fixed split seed (42) ensures identical train/val splits across restarts"

patterns-established:
  - "Adaptive config: get_adaptive_config(file_count) returns fully configured TrainingConfig"
  - "File-level split: validation split at file boundaries prevents leakage"
  - "Event-driven metrics: training loop emits typed events, consumers subscribe via MetricsCallback"
  - "Serialisable history: MetricsHistory.to_dict/from_dict for checkpoint inclusion"

# Metrics
duration: 3min
completed: 2026-02-13
---

# Phase 3 Plan 2: Training Infrastructure Summary

**Adaptive training config with 3 overfitting presets, PyTorch Dataset with file-level split chunking, and callback-based metrics collection system**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-13T00:51:28Z
- **Completed:** 2026-02-13T00:54:42Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Training configuration auto-adapts regularization, validation split, and batch size to dataset size (5-500 files)
- AudioTrainingDataset chunks variable-length audio into fixed 1-second segments with lazy loading
- File-level validation split prevents data leakage (chunks from same file never in both sets)
- MetricsHistory provides loss curve extraction, overfitting detection (20% gap threshold), ETA estimation, and JSON serialisation

## Task Commits

Each task was committed atomically:

1. **Task 1: Training configuration with overfitting presets** - `a9f729f` (feat)
2. **Task 2: PyTorch training dataset and metrics collection** - `68e1ad1` (feat)

## Files Created/Modified
- `src/small_dataset_audio/training/config.py` - TrainingConfig, OverfittingPreset, RegularizationConfig, get_adaptive_config, get_effective_preview_interval
- `src/small_dataset_audio/training/dataset.py` - AudioTrainingDataset (PyTorch Dataset), create_data_loaders with file-level split
- `src/small_dataset_audio/training/metrics.py` - StepMetrics, EpochMetrics, PreviewEvent, TrainingCompleteEvent, MetricsHistory, MetricsCallback

## Decisions Made
- Preview interval uses 50-epoch threshold (all 3 presets have max_epochs >= 100, so short interval only activates on manual override)
- Dataset returns raw waveforms, not mel spectrograms (GPU-side conversion is faster)
- Chunk index uses metadata-only scanning (get_metadata, no waveform load) for fast construction
- Augmentation probability is per-chunk (50%) not per-file for better training stochasticity
- Split seed fixed at 42 for reproducible train/val partitions across restarts

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected plan verification assertion for preview interval**
- **Found during:** Task 1 (Training configuration verification)
- **Issue:** Plan verification asserted `get_effective_preview_interval(tiny) == 2` for CONSERVATIVE preset (10 files), but CONSERVATIVE has max_epochs=100 which is >= 50, so the standard interval (5) applies -- not the short interval (2)
- **Fix:** Verified function works correctly; the plan's assertion was inconsistent with its own preset definition. Function correctly returns 5 for max_epochs >= 50 and 2 for max_epochs < 50.
- **Files modified:** None (logic correct, only verification script adjusted)
- **Verification:** Created explicit test with max_epochs=30 config confirming short path works

---

**Total deviations:** 1 auto-fixed (1 bug in plan verification)
**Impact on plan:** Minimal -- the code logic is correct; only the plan's test assertion conflicted with its own preset parameters.

## Issues Encountered
None -- both tasks executed cleanly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TrainingConfig ready for model architecture (Plan 03) to consume latent_dim, regularization
- AudioTrainingDataset and create_data_loaders ready for training loop (Plan 04)
- MetricsCallback type and event classes ready for training loop emission and UI subscription

## Self-Check: PASSED

All 3 created files verified on disk. Both task commits (a9f729f, 68e1ad1) confirmed in git log.

---
*Phase: 03-core-training-engine*
*Completed: 2026-02-13*
