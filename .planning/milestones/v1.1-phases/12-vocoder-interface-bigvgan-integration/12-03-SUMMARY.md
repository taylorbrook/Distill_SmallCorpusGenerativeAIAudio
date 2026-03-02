---
phase: 12-vocoder-interface-bigvgan-integration
plan: 03
subsystem: vocoder
tags: [mel-adapter, griffin-lim, bigvgan, resampling, waveform-round-trip, torchaudio]

# Dependency graph
requires:
  - phase: 12-02
    provides: BigVGANVocoder class, get_vocoder() factory, weight manager
provides:
  - MelAdapter class converting VAE log1p-HTK mels to BigVGAN log-clamp-Slaney mels
  - End-to-end pipeline: VAE mel [B, 1, 128, T] -> MelAdapter -> BigVGAN -> waveform [B, 1, samples]
  - Updated BigVGANVocoder.mel_to_waveform accepting VAE-format mels directly
affects: [phase-14, phase-16]

# Tech tracking
tech-stack:
  added: [torchaudio.transforms.Resample]
  patterns: [waveform-round-trip-mel-conversion, lazy-mel-adapter-initialization]

key-files:
  created:
    - src/distill/vocoder/mel_adapter.py
  modified:
    - src/distill/vocoder/bigvgan_vocoder.py
    - src/distill/vocoder/__init__.py

key-decisions:
  - "Waveform round-trip (Griffin-Lim) for mel conversion instead of transfer matrix approach -- simpler, guaranteed correct BigVGAN mels"
  - "Griffin-Lim quality loss accepted as stopgap -- Phase 16 per-model HiFi-GAN will eliminate this entirely"
  - "Peak normalization to 0.95 after resample to match BigVGAN training data distribution"
  - "MelAdapter lazy-initialized on first mel_to_waveform call to defer AudioSpectrogram loading"

patterns-established:
  - "Mel format bridge: waveform round-trip via Griffin-Lim when source and target mel configs are incompatible"
  - "Lazy adapter initialization: heavy intermediary loaded only when actually invoked"

requirements-completed: [VOC-02]

# Metrics
duration: 3min
completed: 2026-02-22
---

# Phase 12 Plan 03: Mel Adapter & Audio Quality Verification Summary

**MelAdapter bridges VAE log1p-HTK mels to BigVGAN log-clamp-Slaney mels via waveform round-trip (Griffin-Lim -> resample 48kHz->44.1kHz -> BigVGAN mel computation), completing the end-to-end vocoder pipeline**

## Performance

- **Duration:** 3 min (execution) + human verification
- **Started:** 2026-02-22T00:34:00Z
- **Completed:** 2026-02-22T01:14:35Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- MelAdapter converts VAE mel [B, 1, 128, T] to BigVGAN mel [B, 128, T'] using waveform round-trip: Griffin-Lim reconstruction at 48kHz, torchaudio resample to 44.1kHz, then BigVGAN's own mel_spectrogram() for correct format
- BigVGANVocoder.mel_to_waveform() now accepts VAE-format mels directly, with MelAdapter initialized lazily on first call
- End-to-end pipeline verified: VAE mel -> MelAdapter -> BigVGAN -> audible waveform output
- Human-verified audio quality with known limitations documented and accepted

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement MelAdapter and integrate into BigVGANVocoder** - `f47141b` (feat)
2. **Task 2: Verify vocoder audio quality** - N/A (human-verify checkpoint, no code changes)

## Files Created/Modified
- `src/distill/vocoder/mel_adapter.py` - MelAdapter class with waveform round-trip conversion (Griffin-Lim -> resample -> BigVGAN mel computation)
- `src/distill/vocoder/bigvgan_vocoder.py` - Updated mel_to_waveform to accept VAE-format mels, lazy MelAdapter initialization
- `src/distill/vocoder/__init__.py` - Added MelAdapter to public API exports

## Decisions Made
- **Waveform round-trip over transfer matrix:** Chose the simpler Approach A (Griffin-Lim round-trip) over the more complex Approach C (hybrid transfer matrix). Approach A is guaranteed to produce correct BigVGAN mels because it uses BigVGAN's own mel_spectrogram() on a real waveform signal. The quality tradeoff is acceptable as a stopgap.
- **Lazy MelAdapter initialization:** MelAdapter created on first `mel_to_waveform` call rather than in BigVGANVocoder constructor, so AudioSpectrogram infrastructure is only loaded when actually needed.
- **Peak normalization to 0.95:** After resampling, waveform is peak-normalized to 0.95 to match BigVGAN's training data distribution (avoids clipping artifacts).

## Deviations from Plan

None - plan executed exactly as written.

## Known Audio Quality Limitations

These are inherent to the current approach and accepted as a stopgap until Phase 16:

**1. BigVGAN quality on non-speech content**
- BigVGAN-v2 was trained primarily on speech data (LibriTTS + AudioSet)
- Non-speech audio (instruments, guitar, etc.) sounds wobbly/unstable because it is out of distribution for the model
- This is a limitation of the pre-trained universal model, not of the integration

**2. Griffin-Lim round-trip distortion (InverseMelScale bottleneck)**
- The MelAdapter uses Griffin-Lim via AudioSpectrogram.mel_to_waveform() as an intermediate step
- InverseMelScale is the bottleneck -- it cannot perfectly reconstruct the full STFT from 128 mel bins
- Additional Griffin-Lim iterations do not improve quality (the mel inversion is the limiting factor)
- This adds consistent, audible distortion to the output

**3. Phase 16 resolution path**
- Per-model HiFi-GAN V2 training (Phase 16) will eliminate both issues:
  - Trained on the exact audio data for each model (no out-of-distribution problem)
  - Trained on the exact mel parameters used by the VAE (no mel format conversion needed)
  - No Griffin-Lim intermediate step (direct mel-to-waveform neural conversion)

## Issues Encountered
None - the quality limitations above were anticipated in the research phase and confirmed during human verification.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 12 complete: VocoderBase interface, BigVGAN vocoder wrapper, weight manager, and mel adapter all working
- Ready for Phase 13 (Model Persistence v2) to add optional vocoder state to .distill format
- Ready for Phase 14 (Generation Pipeline Integration) to wire vocoder through all generation paths
- The 44.1kHz -> 48kHz output resample needed by the generation pipeline will be handled in Phase 14

## Self-Check: PASSED

All files verified present, all commits verified in git log.

---
*Phase: 12-vocoder-interface-bigvgan-integration*
*Completed: 2026-02-22*
