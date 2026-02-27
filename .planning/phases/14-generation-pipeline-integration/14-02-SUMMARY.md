---
phase: 14-generation-pipeline-integration
plan: 02
subsystem: inference
tags: [bigvgan, vocoder, generation, training-preview, cli, ui, blending]

# Dependency graph
requires:
  - phase: 14-generation-pipeline-integration
    plan: 01
    provides: "GenerationPipeline with vocoder injection, _vocoder_with_fallback OOM helper"
  - phase: 12-vocoder-foundation
    provides: "BigVGANVocoder, VocoderBase, get_vocoder factory"
provides:
  - "All training preview functions wired to neural vocoder (no Griffin-Lim)"
  - "CLI generate defaults to 44100 Hz sample rate"
  - "UI export sample rate dropdown defaults to 44100"
  - "All 5 GenerationPipeline call sites pass vocoder explicitly"
  - "Training loop creates vocoder once and passes to preview generation"
affects: [15-ui-vocoder-controls, 16-hifigan-training]

# Tech tracking
tech-stack:
  added: []
  patterns: ["vocoder parameter threading through training preview pipeline"]

key-files:
  created: []
  modified:
    - "src/distill/training/preview.py"
    - "src/distill/training/loop.py"
    - "src/distill/cli/generate.py"
    - "src/distill/ui/tabs/generate_tab.py"
    - "src/distill/ui/tabs/library_tab.py"
    - "src/distill/inference/blending.py"

key-decisions:
  - "Training preview functions take vocoder as required parameter (not optional) -- no Griffin-Lim fallback"
  - "Preview sample_rate derived from vocoder.sample_rate in training loop PreviewEvent"
  - "All vocoder imports lazy (inside function bodies) consistent with project pattern"

patterns-established:
  - "Vocoder created once per entry point (train setup, CLI command, UI load) and threaded through"

requirements-completed: [GEN-01, GEN-03]

# Metrics
duration: 4min
completed: 2026-02-27
---

# Phase 14 Plan 02: Vocoder Wiring Across All Call Sites Summary

**Neural vocoder wired through training previews, CLI, UI, and blending -- all mel-to-waveform paths use BigVGAN, all defaults at 44.1kHz**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-27T18:01:45Z
- **Completed:** 2026-02-27T18:05:24Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Replaced all Griffin-Lim mel_to_waveform calls in training previews with neural vocoder using _vocoder_with_fallback OOM helper
- Changed all sample rate defaults from 48000 to 44100 across CLI, UI, and training preview functions
- Passed vocoder explicitly to all 5 GenerationPipeline constructor call sites (CLI, library_tab, 3x blending)
- Wired vocoder creation into training loop setup so it is created once and reused for all preview generations

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire vocoder into training previews and update sample rates** - `faad09b` (feat)
2. **Task 2: Update all GenerationPipeline callers and sample rate defaults** - `9f3fd01` (feat)
3. **Task 3: Wire vocoder into training loop for preview generation** - `c05de07` (feat)

**Plan metadata:** (pending)

## Files Created/Modified
- `src/distill/training/preview.py` - Both preview functions now require vocoder parameter, use _vocoder_with_fallback, default 44100 Hz
- `src/distill/training/loop.py` - Creates BigVGAN vocoder once at training setup, passes to generate_preview, PreviewEvent uses vocoder.sample_rate
- `src/distill/cli/generate.py` - --sample-rate defaults to 44100, passes vocoder to GenerationPipeline
- `src/distill/ui/tabs/generate_tab.py` - Export sample rate dropdown defaults to "44100"
- `src/distill/ui/tabs/library_tab.py` - Passes vocoder to GenerationPipeline when loading model
- `src/distill/inference/blending.py` - All 3 GenerationPipeline call sites pass vocoder

## Decisions Made
- Training preview functions take vocoder as required parameter (not optional) -- calling without vocoder raises TypeError, intentional per user decision to fully remove Griffin-Lim
- Preview sample_rate in PreviewEvent derived from vocoder.sample_rate (44100) instead of spec_config.sample_rate (48000)
- All vocoder imports are lazy (inside function bodies) consistent with project pattern to keep module import lightweight

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All generation, preview, blending, CLI, and UI paths now use the neural vocoder exclusively
- No Griffin-Lim code path remains in any generation or preview path
- The only remaining spectrogram.mel_to_waveform calls are in vocoder/mel_adapter.py (internal) and controls/analyzer.py (analysis, not generation)
- Phase 14 integration is complete -- ready for Phase 15 (UI vocoder controls) and Phase 16 (HiFi-GAN training)

## Self-Check: PASSED

- FOUND: src/distill/training/preview.py
- FOUND: src/distill/training/loop.py
- FOUND: src/distill/cli/generate.py
- FOUND: src/distill/ui/tabs/generate_tab.py
- FOUND: src/distill/ui/tabs/library_tab.py
- FOUND: src/distill/inference/blending.py
- FOUND: commit faad09b
- FOUND: commit 9f3fd01
- FOUND: commit c05de07

---
*Phase: 14-generation-pipeline-integration*
*Completed: 2026-02-27*
