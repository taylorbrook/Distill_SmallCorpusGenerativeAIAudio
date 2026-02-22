---
phase: 13-vq-vae-training-pipeline
plan: 01
subsystem: training
tags: [vqvae, training-loop, codebook-health, metrics, persistence, checkpoint]

# Dependency graph
requires:
  - phase: 12-rvq-vae-core-architecture
    provides: ConvVQVAE model, vqvae_loss, VQVAEConfig, get_adaptive_vqvae_config, QuantizerWrapper
provides:
  - VQ-VAE training loop (train_vqvae, train_vqvae_epoch, validate_vqvae_epoch)
  - VQ metrics dataclasses (VQStepMetrics, VQEpochMetrics, VQMetricsHistory)
  - Codebook health monitoring with low utilization warnings (<30%)
  - VQ-VAE checkpoint save/load (save_vqvae_checkpoint, load_vqvae_checkpoint)
  - VQ-VAE reconstruction preview generation
  - v2 model persistence (save_model_v2, load_model_v2, LoadedVQModel)
  - TrainingRunner.start_vqvae() for background VQ-VAE training
affects: [13-02, 13-03, 14-autoregressive-prior, 16-code-editor-ui]

# Tech tracking
tech-stack:
  added: []
  patterns: [parallel-vq-training-path, vq-metrics-dataclasses, v2-model-format]

key-files:
  created: []
  modified:
    - src/distill/training/metrics.py
    - src/distill/training/loop.py
    - src/distill/training/checkpoint.py
    - src/distill/training/preview.py
    - src/distill/training/runner.py
    - src/distill/training/__init__.py
    - src/distill/models/persistence.py
    - src/distill/models/__init__.py

key-decisions:
  - "Parallel VQ training functions alongside v1.0 (no modifications to existing train/validate/save)"
  - "Codebook health computed every 10 steps during training, full validation set at epoch end"
  - "Skip codebook health at step 0 epoch 0 (k-means not yet initialized)"
  - "Low utilization warnings only after epoch 0 to avoid false positives"
  - "v2 persistence uses dummy forward pass for ConvVQVAE initialization before weight loading"

patterns-established:
  - "VQ-specific dataclasses parallel to v1.0 (VQStepMetrics vs StepMetrics, etc.)"
  - "MetricsCallback type union accepts both v1.0 and v1.1 event types"
  - "v2 .distill format with model_type='vqvae' and version=2 fields"

requirements-completed: [VQVAE-04, VQVAE-07, PERS-01]

# Metrics
duration: 7min
completed: 2026-02-22
---

# Phase 13 Plan 01: VQ-VAE Training Loop and v2 Persistence Summary

**VQ-VAE training loop with codebook health monitoring, low-utilization warnings, VQ metrics dataclasses, and v2 .distill model persistence**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-22T00:49:39Z
- **Completed:** 2026-02-22T00:56:41Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- VQ-VAE training loop mirrors v1.0 structure with VQ-specific forward pass, loss computation, and codebook health monitoring
- Per-level codebook utilization, perplexity, and dead code metrics computed during validation and emitted via callback
- Low utilization warnings generated at <30% threshold (VQVAE-07) with epoch 0 skip for k-means init
- v2 .distill format saves ConvVQVAE with codebook health snapshot, loss curves, and VQ config
- v2 load reconstructs ConvVQVAE from saved config using dummy forward pass initialization
- All v1.0 functions remain unchanged and fully functional

## Task Commits

Each task was committed atomically:

1. **Task 1: VQ metrics dataclasses and VQ-VAE training loop with codebook health** - `d3b540f` (feat)
2. **Task 2: Implement v2 model persistence (save and load)** - `72c71a8` (feat)
3. **Training __init__ exports** - `3a03d1e` (chore)

## Files Created/Modified
- `src/distill/training/metrics.py` - VQStepMetrics, VQEpochMetrics, VQMetricsHistory dataclasses; updated MetricsCallback union type
- `src/distill/training/loop.py` - train_vqvae_epoch, validate_vqvae_epoch, train_vqvae orchestrator with codebook health and low utilization warnings
- `src/distill/training/checkpoint.py` - save_vqvae_checkpoint, load_vqvae_checkpoint with VQ-specific fields
- `src/distill/training/preview.py` - generate_vqvae_reconstruction_preview for waveform-to-mel-to-recon preview pairs
- `src/distill/training/runner.py` - TrainingRunner.start_vqvae() and _run_vqvae_training() thread target
- `src/distill/training/__init__.py` - v1.1 VQ-VAE symbol exports
- `src/distill/models/persistence.py` - SAVED_MODEL_VERSION_V2, LoadedVQModel, save_model_v2, load_model_v2
- `src/distill/models/__init__.py` - v2 persistence exports added to public API

## Decisions Made
- Parallel VQ training functions alongside v1.0 to avoid regression risk (no modifications to existing train/validate/save)
- Codebook health computed every 10 steps during training for periodic monitoring without excessive overhead
- Skip codebook health at step 0 of epoch 0 to avoid misleading 0% utilization from uninitialized k-means (Pitfall 1)
- Low utilization warnings only emitted after epoch 0 to prevent false positives during initialization
- v2 load uses dummy forward pass to initialize ResidualVQ internal state before loading state dict
- Reuse create_data_loaders with a shim TrainingConfig for VQ-VAE data loading

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Python path conflict: system had a different `distill` package installed globally. Resolved by using explicit PYTHONPATH for verification. Does not affect actual usage since the project has its own editable install.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- VQ-VAE training loop ready for UI integration (Plan 02: train tab with codebook health display)
- VQ-VAE training loop ready for CLI integration (Plan 03: CLI training with VQ flags)
- v2 persistence ready for model library integration
- All VQ metrics dataclasses ready for callback consumers (UI timer, CLI callback)

## Self-Check: PASSED

All 8 modified files verified present. All 3 task commits verified in git log.

---
*Phase: 13-vq-vae-training-pipeline*
*Completed: 2026-02-22*
