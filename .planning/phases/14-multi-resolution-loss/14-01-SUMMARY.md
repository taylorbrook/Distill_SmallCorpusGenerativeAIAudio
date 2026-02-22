---
phase: 14-multi-resolution-loss
plan: 01
subsystem: training
tags: [auraloss, stft, loss-function, vae, multi-resolution, l1, kl-divergence]

# Dependency graph
requires:
  - phase: 13-2channel-vae
    provides: 2-channel magnitude+IF VAE architecture and training loop
provides:
  - LossConfig dataclass with nested STFTLossConfig, ReconLossConfig, KLLossConfig
  - compute_combined_loss function combining STFT + L1 recon + KL losses
  - create_stft_loss factory for MultiResolutionSTFTLoss
  - auraloss project dependency
affects: [14-02-integration-testing, training-loop, loss-logging]

# Tech tracking
tech-stack:
  added: [auraloss>=0.4.0]
  patterns: [nested-dataclass-config, magnitude-weighted-if-loss, multi-resolution-stft]

key-files:
  created: []
  modified:
    - pyproject.toml
    - src/distill/training/config.py
    - src/distill/models/losses.py

key-decisions:
  - "STFT loss weight 1.0 vs reconstruction 0.1 -- spectral quality takes precedence per user decision"
  - "STFT loss applied to flattened magnitude channel only (IF is derivative signal, not spectral content)"
  - "Magnitude-weighted IF loss normalizes weights so mean ~= 1 to preserve loss scale"
  - "Lazy STFT loss initialization supported but caller-created instance preferred for performance"

patterns-established:
  - "Nested dataclass config: sub-dataclasses composed in parent for dot-notation access (config.loss.stft.weight)"
  - "Loss dict return: compute_combined_loss returns dict of individual components for granular logging"

requirements-completed: [LOSS-01, LOSS-02, LOSS-03, LOSS-04]

# Metrics
duration: 4min
completed: 2026-02-22
---

# Phase 14 Plan 01: Multi-Resolution Loss Config and Implementation Summary

**Combined multi-resolution STFT loss (auraloss) + magnitude-weighted IF L1 reconstruction + KL divergence with nested LossConfig dataclass**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-22T06:04:10Z
- **Completed:** 2026-02-22T06:07:50Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- LossConfig dataclass with nested STFTLossConfig, ReconLossConfig, KLLossConfig supporting dot-notation access (config.loss.stft.weight, config.loss.reconstruction.magnitude_weight, config.loss.kl.weight_max)
- compute_combined_loss function computing multi-resolution STFT + per-channel L1 recon (magnitude-weighted IF) + KL divergence, returning dict of all individual loss components
- auraloss>=0.4.0 installed as project dependency for MultiResolutionSTFTLoss at 3 resolutions (512, 1024, 2048)
- Existing vae_loss and get_kl_weight functions preserved unchanged for backward compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: Add auraloss dependency and LossConfig dataclass** - `01521e1` (feat)
2. **Task 2: Implement compute_combined_loss in losses.py** - `c1e1302` (feat)

**Plan metadata:** pending (docs: complete plan)

## Files Created/Modified
- `pyproject.toml` - Added auraloss>=0.4.0 dependency
- `src/distill/training/config.py` - Added STFTLossConfig, ReconLossConfig, KLLossConfig, LossConfig dataclasses; added loss field to TrainingConfig
- `src/distill/models/losses.py` - Added compute_combined_loss, create_stft_loss; updated module docstring for v2.0 combined loss

## Decisions Made
- STFT loss weight 1.0 vs reconstruction 0.1 -- spectral quality takes precedence per user decision
- STFT loss applied to flattened magnitude channel only (IF is a derivative signal, not spectral content)
- Magnitude-weighted IF loss normalizes weights so mean ~= 1 to preserve loss scale
- Lazy STFT loss initialization supported but caller-created instance preferred for performance
- Existing kl_warmup_fraction, kl_weight_max, free_bits on TrainingConfig kept for backward compatibility, documented as superseded by loss.kl.*

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LossConfig and compute_combined_loss ready for integration into training loop (14-02)
- create_stft_loss factory ready for one-time creation at training start
- All loss components return finite values on random input
- Existing vae_loss preserved for any code still using the legacy API

## Self-Check: PASSED

- All 4 files verified present
- Both task commits (01521e1, c1e1302) verified in git log

---
*Phase: 14-multi-resolution-loss*
*Completed: 2026-02-22*
