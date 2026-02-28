---
phase: 16-encode-decode-code-visualization
plan: 02
subsystem: ui
tags: [gradio, codes-tab, encode, decode, vqvae, interactive-grid, audio-preview, cross-tab-wiring]

# Dependency graph
requires:
  - phase: 16-encode-decode-code-visualization
    plan: 01
    provides: "encode_audio_file, decode_code_grid, preview_single_code, preview_time_slice, play_row_audio, render_code_grid, get_level_labels"
  - phase: 13-vqvae-training
    provides: "Model persistence (LoadedVQModel, load_model_v2)"
provides:
  - "build_codes_tab() complete Codes tab with model selector, encode/decode, grid, audio players"
  - "Codes tab registered in 5-tab app layout (Data, Train, Generate, Codes, Library)"
  - "Library model load chain refreshes Codes tab model dropdown"
affects: [17-code-editing, ui-app]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Module-level _current_encode state for encode result persistence across handlers"
    - "VQ-VAE model dropdown scanning via torch.load peek (version + model_type filter)"
    - "Cross-tab wiring via .then() chain from Library load to Codes dropdown refresh"

key-files:
  created:
    - "src/distill/ui/tabs/codes_tab.py"
  modified:
    - "src/distill/ui/app.py"

key-decisions:
  - "MAX_LEVEL_LABELS=4 covers all supported quantizer configurations (2-4 levels)"
  - "Auto-decode on encode for immediate A/B comparison (no extra click needed)"
  - "Model dropdown scans .distill files directly with torch.load peek (no library catalog dependency)"
  - "Codes tab placed between Generate and Library in tab order"

patterns-established:
  - "Module-level state dict pattern for encode results (_current_encode, _current_labels)"
  - "build_codes_tab() returns dict with model_dropdown, grid_html, status for cross-tab wiring"

requirements-completed: [CODE-01, CODE-02, CODE-03, CODE-07, CODE-09]

# Metrics
duration: 3min
completed: 2026-02-28
---

# Phase 16 Plan 02: Codes Tab UI & App Registration Summary

**Interactive Codes tab with VQ-VAE model selector, audio encode/decode, grid click previews, level label editing, and cross-tab Library sync**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-28T00:25:06Z
- **Completed:** 2026-02-28T00:28:04Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Complete Codes tab with model selector (VQ-VAE v2 only), audio upload, encode/decode buttons, interactive grid, and three audio players
- Cell/column/row click dispatching to preview_single_code, preview_time_slice, and play_row_audio with autoplay
- Cross-tab wiring so Library model load refreshes Codes tab model dropdown

## Task Commits

Each task was committed atomically:

1. **Task 1: Create codes_tab.py with full Codes tab layout and event handlers** - `02fa4d9` (feat)
2. **Task 2: Register Codes tab in app.py and wire cross-tab events** - `ce72c1c` (feat)

## Files Created/Modified
- `src/distill/ui/tabs/codes_tab.py` - Complete Codes tab builder with model selector, encode/decode handlers, cell click dispatch, level label editor, and audio players
- `src/distill/ui/app.py` - Codes tab registered in 5-tab layout, cross-tab wiring from Library load chain to Codes dropdown refresh

## Decisions Made
- MAX_LEVEL_LABELS set to 4 to cover all supported RVQ configurations (2-4 quantizer levels)
- Auto-decode on encode provides immediate A/B comparison without requiring a separate Decode click
- Model dropdown uses direct .distill file scanning with torch.load peek rather than ModelLibrary catalog (simpler, no index dependency)
- Codes tab positioned between Generate and Library tabs in the layout order (logical flow: generate -> inspect codes -> manage library)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Codes tab fully wired and ready for user interaction
- Grid click events bridge JS to Python handlers for real-time code previews
- Foundation ready for Phase 17 code editing (grid cells are selectable, labels editable)

## Self-Check: PASSED

All created files verified on disk. All commit hashes found in git log.

---
*Phase: 16-encode-decode-code-visualization*
*Completed: 2026-02-28*
