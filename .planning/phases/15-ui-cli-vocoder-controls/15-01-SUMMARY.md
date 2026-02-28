---
phase: 15-ui-cli-vocoder-controls
plan: 01
subsystem: ui
tags: [gradio, vocoder, bigvgan, download-progress, accordion, lazy-loading]

# Dependency graph
requires:
  - phase: 14-generation-pipeline
    provides: GenerationPipeline with vocoder parameter, BigVGAN integration
  - phase: 12-bigvgan-integration
    provides: BigVGANVocoder, weight_manager, VocoderBase
provides:
  - resolve_vocoder() shared function with auto/bigvgan/hifigan selection logic
  - tqdm_class parameter through vocoder download chain for custom progress display
  - Vocoder Settings accordion in Generate tab with dropdown and status text
  - Deferred vocoder creation pattern (lazy download on first generate)
affects: [15-02, 16-hifigan-training]

# Tech tracking
tech-stack:
  added: []
  patterns: [generator-based-gradio-handlers, lazy-vocoder-resolution, deferred-pipeline-creation]

key-files:
  created: []
  modified:
    - src/distill/vocoder/__init__.py
    - src/distill/vocoder/weight_manager.py
    - src/distill/vocoder/bigvgan_vocoder.py
    - src/distill/ui/tabs/generate_tab.py
    - src/distill/ui/tabs/library_tab.py

key-decisions:
  - "Deferred pipeline creation: app_state.pipeline = None at model load, created at generate time with resolved vocoder"
  - "Generator-based _generate_audio: yields intermediate button-disable update during download, then final results"
  - "tqdm_class forwarded through entire vocoder chain for UI/CLI progress customization"

patterns-established:
  - "Lazy vocoder resolution: vocoder downloaded/created on first generate, not model load"
  - "Generator handler pattern: yield intermediate UI updates for long operations, then final results"
  - "Error-in-accordion: inline error message + retry button for download failures (not Gradio toast)"

requirements-completed: [UI-01, UI-02]

# Metrics
duration: 5min
completed: 2026-02-28
---

# Phase 15 Plan 01: UI Vocoder Controls Summary

**resolve_vocoder() auto-selection function, tqdm_class-aware download chain, and Vocoder Settings accordion in Generate tab with lazy BigVGAN download on first generate**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-28T00:17:27Z
- **Completed:** 2026-02-28T00:22:48Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Added `resolve_vocoder()` function with auto/bigvgan/hifigan selection and info dict return
- Added `tqdm_class` parameter through entire vocoder download chain (get_vocoder -> BigVGANVocoder -> ensure_bigvgan_weights)
- Added Vocoder Settings accordion to Generate tab with dropdown, status text, progress display, and retry button
- Converted `_generate_audio()` to generator for button disable/enable during vocoder download
- Deferred vocoder creation from model load time to generate time (lazy download pattern)
- Quality badge now includes vocoder name after generation

## Task Commits

Each task was committed atomically:

1. **Task 1: Add resolve_vocoder() and tqdm_class support to vocoder package** - `fd1049f` (feat)
2. **Task 2: Add Vocoder Settings accordion to Generate tab and defer vocoder creation** - `6bdfa19` (feat)

**Plan metadata:** (pending)

## Files Created/Modified
- `src/distill/vocoder/__init__.py` - Added resolve_vocoder(), tqdm_class to get_vocoder(), TYPE_CHECKING import for LoadedModel
- `src/distill/vocoder/weight_manager.py` - Added tqdm_class parameter to ensure_bigvgan_weights()
- `src/distill/vocoder/bigvgan_vocoder.py` - Added tqdm_class parameter to BigVGANVocoder.__init__(), forwarded to ensure_bigvgan_weights()
- `src/distill/ui/tabs/generate_tab.py` - Added Vocoder Settings accordion, converted _generate_audio to generator, added vocoder resolution at generate time, vocoder info in quality badge
- `src/distill/ui/tabs/library_tab.py` - Removed vocoder creation at model load, set pipeline=None for deferred creation

## Decisions Made
- **Deferred pipeline creation:** Rather than modifying GenerationPipeline to accept vocoder=None without fallback, set app_state.pipeline = None at model load and create the pipeline in _generate_audio after vocoder resolution. This avoids modifying the GenerationPipeline constructor while achieving the lazy download goal.
- **Generator-based handler:** Converted _generate_audio to a Python generator (yield) to support intermediate UI updates (disabling generate button during download, showing progress) before final results.
- **Error-in-accordion pattern:** Download failures show inline error message + Retry Download button inside the Vocoder Settings accordion, following the CONTEXT.md locked decision (not Gradio toast).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- resolve_vocoder() ready for CLI integration in Plan 02 (--vocoder flag, Rich progress bar)
- tqdm_class parameter ready for Rich progress bar integration
- Vocoder dropdown dynamically updates choices when model with vocoder_state is loaded
- Per-model HiFi-GAN option prepared (disabled until Phase 16)

## Self-Check: PASSED

- All 5 source files verified present
- Commit fd1049f (Task 1) verified in git log
- Commit 6bdfa19 (Task 2) verified in git log
- SUMMARY.md verified at expected path

---
*Phase: 15-ui-cli-vocoder-controls*
*Completed: 2026-02-28*
