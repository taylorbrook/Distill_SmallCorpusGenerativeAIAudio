---
phase: 14-generation-pipeline-integration
plan: 01
subsystem: inference
tags: [bigvgan, vocoder, generation, kaiser-resampler, oom-fallback]

# Dependency graph
requires:
  - phase: 12-vocoder-foundation
    provides: "BigVGANVocoder, VocoderBase, get_vocoder factory, MelAdapter"
  - phase: 13-model-persistence-v2
    provides: "vocoder_state persistence in .distillgan format"
provides:
  - "GenerationPipeline with vocoder injection and auto-creation fallback"
  - "_vocoder_with_fallback GPU OOM -> CPU fallback helper"
  - "Kaiser-windowed sinc resampler (lowpass_filter_width=64)"
  - "All chunking functions wired to vocoder.mel_to_waveform"
  - "Default export sample rate 44100 Hz (BigVGAN native)"
affects: [14-02, 15-ui-vocoder-controls, 16-hifigan-training]

# Tech tracking
tech-stack:
  added: []
  patterns: ["vocoder injection via constructor with lazy auto-creation", "OOM fallback pattern for GPU inference"]

key-files:
  created: []
  modified:
    - "src/distill/inference/generation.py"
    - "src/distill/inference/chunking.py"

key-decisions:
  - "Internal sample rate derived from vocoder.sample_rate instead of hardcoded 48000"
  - "chunk_samples still uses spectrogram.config.sample_rate for mel shape computation (spectrogram operates at 48kHz)"
  - "Kaiser-windowed sinc interpolation with lowpass_filter_width=64 for resampling"
  - "Default GenerationConfig.sample_rate changed from 48000 to 44100"

patterns-established:
  - "Vocoder injection: pass vocoder through pipeline to all mel-to-waveform call sites"
  - "OOM fallback: try GPU, catch RuntimeError('out of memory'), empty cache, fallback to CPU, restore device in finally"

requirements-completed: [GEN-01, GEN-02]

# Metrics
duration: 4min
completed: 2026-02-27
---

# Phase 14 Plan 01: Generation Pipeline Integration Summary

**BigVGAN vocoder wired through GenerationPipeline and all three chunking paths, replacing Griffin-Lim with Kaiser-windowed sinc resampler and GPU OOM fallback**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-27T17:54:35Z
- **Completed:** 2026-02-27T17:58:56Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Injected vocoder into GenerationPipeline with auto-creation fallback via get_vocoder factory
- Replaced all spectrogram.mel_to_waveform calls with vocoder-based inference across all three chunking paths (crossfade, latent_interp, slider)
- Added GPU OOM fallback helper that gracefully moves to CPU on memory exhaustion
- Upgraded resampler from default sinc to Kaiser-windowed sinc with lowpass_filter_width=64
- Changed default export sample rate to 44100 Hz (BigVGAN native rate)

## Task Commits

Each task was committed atomically:

1. **Task 1: Inject vocoder into GenerationPipeline and update internal sample rate** - `6273f35` (feat)
2. **Task 2: Update chunking functions to use vocoder instead of spectrogram.mel_to_waveform** - `df5e968` (feat)

**Plan metadata:** `b74524c` (docs: complete plan)

## Files Created/Modified
- `src/distill/inference/generation.py` - GenerationPipeline with vocoder injection, _vocoder_with_fallback OOM helper, Kaiser resampler, 44100 default sample rate
- `src/distill/inference/chunking.py` - generate_chunks_crossfade and generate_chunks_latent_interp using vocoder for mel-to-waveform

## Decisions Made
- Internal sample rate now derived from vocoder.sample_rate (44100 for BigVGAN) instead of hardcoded 48000
- chunk_samples computation still uses spectrogram.config.sample_rate (48kHz) because that feeds mel shape computation via spectrogram.get_mel_shape -- the vocoder produces audio at its own native rate from those mel frames
- Kaiser-windowed sinc interpolation chosen for highest quality resampling (lowpass_filter_width=64)
- GenerationConfig.sample_rate defaults to 44100 to match BigVGAN native output

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Python path picking up wrong `distill` package from another project; resolved by using PYTHONPATH override for verification

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All generation paths now use vocoder for mel-to-waveform conversion
- Ready for Plan 02 (error handling, logging, test coverage)
- MPS compatibility for BigVGAN Snake activations still unverified (deferred from Phase 12)

## Self-Check: PASSED

- FOUND: src/distill/inference/generation.py
- FOUND: src/distill/inference/chunking.py
- FOUND: 14-01-SUMMARY.md
- FOUND: commit 6273f35
- FOUND: commit df5e968

---
*Phase: 14-generation-pipeline-integration*
*Completed: 2026-02-27*
