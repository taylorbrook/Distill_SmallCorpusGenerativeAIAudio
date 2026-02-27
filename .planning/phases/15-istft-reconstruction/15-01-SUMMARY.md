---
phase: 15-istft-reconstruction
plan: 01
subsystem: audio
tags: [istft, phase-reconstruction, inverse-mel, cumulative-sum, spectrogram]

# Dependency graph
requires:
  - phase: 12-2-channel-data-pipeline
    provides: "ComplexSpectrogram with waveform_to_complex_mel, normalize, denormalize"
provides:
  - "complex_mel_to_waveform method on ComplexSpectrogram (ISTFT reconstruction)"
  - "Phase reconstruction via cumulative sum of IF"
  - "Mel-to-linear frequency inversion via InverseMelScale"
affects: [15-02-griffin-lim-removal, 16-pipeline-wiring]

# Tech tracking
tech-stack:
  added: [torchaudio.transforms.InverseMelScale (on ComplexSpectrogram)]
  patterns: [IF cumsum phase reconstruction, CPU-only InverseMelScale inversion]

key-files:
  created:
    - tests/test_istft_reconstruction.py
  modified:
    - src/distill/audio/spectrogram.py

key-decisions:
  - "InverseMelScale used for both magnitude and phase mel-to-linear inversion (least-squares projection)"
  - "Phase cumulative sum starts at zero, left unwrapped (per user decision)"
  - "Denormalization handled inside complex_mel_to_waveform when stats dict provided"

patterns-established:
  - "ISTFT reconstruction: denorm -> split channels -> expm1+sqrt -> IF*pi -> cumsum -> InverseMelScale(CPU) -> complex STFT -> istft"

requirements-completed: [RECON-01, RECON-02]

# Metrics
duration: 3min
completed: 2026-02-27
---

# Phase 15 Plan 01: ISTFT Reconstruction Summary

**ISTFT waveform reconstruction from 2-channel magnitude+IF spectrograms via cumulative-sum phase and InverseMelScale inversion**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-27T18:31:24Z
- **Completed:** 2026-02-27T18:33:50Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Implemented `complex_mel_to_waveform` method on ComplexSpectrogram for exact phase reconstruction via ISTFT
- Full 9-step pipeline: denormalize, split channels, undo log1p, undo IF normalization, cumsum phase, InverseMelScale, complex STFT, ISTFT, reshape
- 7 comprehensive tests covering shape, NaN/Inf safety, amplitude, round-trip quality, normalization round-trip, white noise, and batch processing

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for ISTFT reconstruction** - `f27ef8d` (test)
2. **Task 2: Implement complex_mel_to_waveform on ComplexSpectrogram** - `72d4ff9` (feat)

## Files Created/Modified
- `tests/test_istft_reconstruction.py` - 7 test cases for ISTFT reconstruction pipeline
- `src/distill/audio/spectrogram.py` - Added complex_mel_to_waveform method and InverseMelScale instance attribute

## Decisions Made
- InverseMelScale applied to both magnitude and phase for mel-to-linear inversion (least-squares is reasonable for phase projection too)
- Denormalization happens inside `complex_mel_to_waveform` when `stats` dict is provided; when `stats=None`, input is assumed raw
- Phase left unwrapped after cumulative sum (no wrapping to [-pi, pi]), starting at zero -- per user decision from 15-CONTEXT.md

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `complex_mel_to_waveform` is ready for use in the training preview pipeline
- Phase 15 Plan 02 (Griffin-Lim removal) can proceed -- the ISTFT path is now the replacement
- All 7 tests pass confirming reconstruction produces valid audio from 2-channel spectrograms

## Self-Check: PASSED

- FOUND: tests/test_istft_reconstruction.py
- FOUND: src/distill/audio/spectrogram.py
- FOUND: f27ef8d (Task 1 commit)
- FOUND: 72d4ff9 (Task 2 commit)

---
*Phase: 15-istft-reconstruction*
*Completed: 2026-02-27*
