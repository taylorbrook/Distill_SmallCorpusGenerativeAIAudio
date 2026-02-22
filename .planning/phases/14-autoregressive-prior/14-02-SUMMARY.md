---
phase: 14-autoregressive-prior
plan: 02
subsystem: training
tags: [autoregressive, prior, training-loop, memorization-detection, perplexity, cross-entropy, pytorch]

# Dependency graph
requires:
  - phase: 14-autoregressive-prior
    provides: CodePrior model, PriorConfig, extract_code_sequences, flatten_codes
  - phase: 13-vqvae-training
    provides: VQ-VAE training loop patterns (train_vqvae, train_vqvae_epoch, validate_vqvae_epoch)
  - phase: 12-vqvae-model
    provides: ConvVQVAE model producing [B, seq_len, num_quantizers] indices
provides:
  - train_prior() full pipeline orchestrator (load VQ-VAE, extract codes, train prior, track best checkpoint)
  - train_prior_epoch() single-epoch training with next-token prediction
  - validate_prior_epoch() validation with accurate per-token averaging
  - check_memorization() with adaptive thresholds by dataset size tier
  - PriorStepMetrics, PriorEpochMetrics, PriorTrainingCompleteEvent dataclasses
  - MetricsCallback extended to accept prior metric events
affects: [14-03 prior CLI/persistence, 15-generation, 16-dashboard]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Prior trains purely on pre-extracted code tensors (VQ-VAE only used for extraction, not during training)"
    - "Adaptive memorization thresholds: 2.0/3.0/5.0 by dataset tier (<=20/<=100/>100 files)"
    - "Best checkpoint tracked via deepcopy(state_dict) on val_perplexity improvement"
    - "Validation loss uses reduction=sum / total_tokens for accurate cross-batch averaging"

key-files:
  created:
    - src/distill/training/prior_loop.py
  modified:
    - src/distill/training/metrics.py

key-decisions:
  - "VQ-VAE completely frozen and only used for code extraction -- prior trains on pre-extracted tensors to prevent gradient leaks"
  - "Memorization thresholds relaxed and adaptive: 2.0 for <=20 files, 3.0 for <=100, 5.0 for >100"
  - "Best checkpoint tracked via deepcopy of state_dict (not disk checkpointing) -- simpler for in-memory prior models"
  - "Validation perplexity clamped via exp(min(val_loss, 20.0)) to prevent float overflow"

patterns-established:
  - "Code-tensor training: extract codes once, train on tensors with randperm shuffling (no DataLoader needed for prior)"
  - "Prior metric events parallel VQ metrics: PriorStepMetrics/PriorEpochMetrics/PriorTrainingCompleteEvent"

requirements-completed: [GEN-01, GEN-06]

# Metrics
duration: 4min
completed: 2026-02-22
---

# Phase 14 Plan 02: Prior Training Loop Summary

**Cross-entropy training loop with validation perplexity monitoring, adaptive memorization detection (2.0/3.0/5.0 thresholds by dataset tier), and best-checkpoint tracking via state_dict deepcopy**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-22T06:11:07Z
- **Completed:** 2026-02-22T06:15:30Z
- **Tasks:** 2
- **Files created/modified:** 2

## Accomplishments
- train_prior() orchestrates full pipeline: load VQ-VAE -> freeze -> extract codes -> flatten -> train CodePrior -> return best weights
- Validation perplexity (exp of cross-entropy) monitored each epoch with best checkpoint automatically tracked
- Memorization detection uses relaxed adaptive thresholds per user decision (only warns when very likely)
- Prior metric dataclasses (PriorStepMetrics, PriorEpochMetrics, PriorTrainingCompleteEvent) follow established VQ metrics pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Add PriorStepMetrics and PriorEpochMetrics to metrics.py** - `1b098f0` (feat)
2. **Task 2: Create prior training loop with memorization detection and best-checkpoint tracking** - `d00d299` (feat)

## Files Created/Modified
- `src/distill/training/metrics.py` - Added PriorStepMetrics, PriorEpochMetrics, PriorTrainingCompleteEvent dataclasses; extended MetricsCallback union type (746 lines)
- `src/distill/training/prior_loop.py` - Full prior training pipeline: train_prior, train_prior_epoch, validate_prior_epoch, check_memorization (684 lines)

## Decisions Made
- VQ-VAE is completely frozen and only used for code extraction, not during prior training -- prevents gradient leaks per RESEARCH.md pitfall 5
- Prior trains on pre-extracted code tensors with torch.randperm shuffling per epoch -- simpler than DataLoader for code-level training
- Validation loss uses reduction="sum" divided by total tokens for accurate cross-batch averaging (not per-batch averaging which would be biased)
- Best checkpoint tracked via deepcopy(state_dict) in memory rather than saving to disk -- prior models are small enough for this approach
- Perplexity clamped via exp(min(val_loss, 20.0)) to avoid float overflow on early high-loss epochs

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- PYTHONPATH conflict: another project (Distill-complex-spec) was installed in the same Python environment, shadowing imports. Resolved by using explicit PYTHONPATH for verification. Does not affect runtime behavior since the project is installed in editable mode.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Prior training loop ready for CLI integration (14-03)
- train_prior returns dict with prior_model, prior_config, prior_metadata for persistence
- PriorTrainingCompleteEvent ready for dashboard integration
- check_memorization ready for UI warning display

## Self-Check: PASSED

- [x] src/distill/training/prior_loop.py exists (684 lines, min 200)
- [x] src/distill/training/metrics.py contains PriorEpochMetrics
- [x] Commit 1b098f0 found in git log
- [x] Commit d00d299 found in git log
- [x] All 6 verifications passed

---
*Phase: 14-autoregressive-prior*
*Completed: 2026-02-22*
