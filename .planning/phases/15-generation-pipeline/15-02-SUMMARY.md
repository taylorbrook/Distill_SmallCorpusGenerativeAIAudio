---
phase: 15-generation-pipeline
plan: 02
subsystem: ui
tags: [gradio, generate-tab, prior, sampling, temperature, top-k, top-p, vq-vae]

# Dependency graph
requires:
  - phase: 15-generation-pipeline
    provides: generate_audio_from_prior() sampling engine and end-to-end pipeline
provides:
  - Prior-based generate tab with temperature, top-k, top-p, duration, crossfade, seed controls
  - _generate_prior_audio() UI handler calling generate_audio_from_prior()
  - _update_generate_tab_for_model() visibility helper for model-type switching
  - loaded_vq_model field on AppState for cross-tab VQ-VAE model access
affects: [15-03, 16-code-editor, library-tab-model-loading]

# Tech tracking
tech-stack:
  added: []
  patterns: [dual-control-section-visibility, prior-vs-v1-model-detection]

key-files:
  created: []
  modified:
    - src/distill/ui/tabs/generate_tab.py
    - src/distill/ui/state.py

key-decisions:
  - "Dual control sections: prior_controls_section and controls_section toggled by model type"
  - "Prior export uses simplified inline handler (no sidecar JSON, no metadata fields)"
  - "Overlap in ms converted to samples at 48kHz (48 samples/ms) in handler"

patterns-established:
  - "Model-type visibility pattern: _update_generate_tab_for_model() returns [empty_msg, controls, prior_controls] updates"
  - "Prior controls section alongside v1.0 controls (no deletion of existing UI)"

requirements-completed: [UI-04]

# Metrics
duration: 3min
completed: 2026-02-27
---

# Phase 15 Plan 02: Prior-Based Generate Tab Summary

**Prior sampling controls UI with temperature/top-k/top-p sliders, duration/crossfade controls, seed input, and model-type-aware visibility toggling**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-27T17:56:35Z
- **Completed:** 2026-02-27T17:59:44Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Prior-based generate tab UI with all sampling controls (temperature 0.1-2.0, top-p 0-1.0, top-k 0-512, duration 1-30s, crossfade 0-200ms)
- loaded_vq_model field on AppState for cross-tab access to VQ-VAE models with priors
- Model-type visibility helper that toggles between v1.0 slider controls and prior sampling controls
- Generate button wired to _generate_prior_audio() handler with gr.Progress callback for chunk counter

## Task Commits

Each task was committed atomically:

1. **Task 1: Add loaded_vq_model to AppState and create prior generation handler** - `739311c` (feat)
2. **Task 2: Build prior-based generate tab UI section with sampling controls** - `b47d9b3` (feat)

## Files Created/Modified
- `src/distill/ui/state.py` - Added loaded_vq_model: Optional[LoadedVQModel] field and LoadedVQModel TYPE_CHECKING import
- `src/distill/ui/tabs/generate_tab.py` - Added prior_controls_section with sampling controls, _generate_prior_audio handler, _update_generate_tab_for_model visibility helper, prior event wiring, prior export handler

## Decisions Made
- Dual control sections approach: prior_controls_section and controls_section coexist, toggled by loaded model type (no deletion of v1.0 UI)
- Prior export uses a simplified inline handler rather than the full v1.0 export pipeline (no metadata fields, no sidecar JSON)
- Overlap converted from ms to samples in the handler (48 samples per ms at 48kHz) matching the generate_audio_from_prior API

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Generate tab ready for visual verification with a trained VQ-VAE model with prior
- _update_generate_tab_for_model() ready for cross-tab wiring from library tab model loading
- prior_controls_section exposed in component dict for external visibility control
- Plan 03 (CLI generate command) can proceed independently

## Self-Check: PASSED

- src/distill/ui/tabs/generate_tab.py exists: FOUND
- src/distill/ui/state.py exists: FOUND
- Task 1 commit 739311c: FOUND
- Task 2 commit b47d9b3: FOUND
- All imports verified working

---
*Phase: 15-generation-pipeline*
*Completed: 2026-02-27*
