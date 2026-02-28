---
phase: 16-encode-decode-code-visualization
plan: 01
subsystem: inference, ui
tags: [vqvae, codes, encode, decode, preview, html-grid, css-grid, javascript, matplotlib]

# Dependency graph
requires:
  - phase: 12-vqvae-architecture
    provides: "ConvVQVAE model with encode/quantize/decode/codes_to_embeddings"
  - phase: 13-vqvae-training
    provides: "Model persistence (LoadedVQModel) and training pipeline"
provides:
  - "encode_audio_file() for extracting VQ-VAE code indices from audio"
  - "decode_code_grid() for reconstructing audio from code indices"
  - "preview_single_code(), preview_time_slice(), play_row_audio() for granular code audio preview"
  - "render_code_grid() for interactive HTML visualization of VQ-VAE codes"
  - "DEFAULT_LEVEL_LABELS cascading scheme for quantizer level naming"
affects: [16-02 codes-tab, ui-components]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "JS bridge onclick dispatch to hidden Gradio textbox via nativeSet pattern"
    - "CSS grid with sticky row labels for horizontal scroll"
    - "Playhead CSS animation with JS start/stop control"
    - "Lazy imports in inference functions following generation.py pattern"

key-files:
  created:
    - "src/distill/inference/codes.py"
    - "src/distill/ui/components/code_grid.py"
  modified:
    - "src/distill/inference/__init__.py"

key-decisions:
  - "tab20 colormap (20 distinct colors cycling) for code cell coloring -- visually distinct without being overwhelming"
  - "Luminance-based text color (ITU-R BT.709) for readability on any cell background"
  - "Playhead animation via CSS @keyframes with JS class toggle -- no timer-based JS needed"

patterns-established:
  - "code-grid-cell-clicked: JS dispatch target elem_id for grid click events"
  - "cell,{level},{pos} / col,{pos} / row,{level}: event value format for code grid interactions"

requirements-completed: [CODE-01, CODE-02, CODE-03, CODE-07, CODE-09]

# Metrics
duration: 5min
completed: 2026-02-28
---

# Phase 16 Plan 01: Encode/Decode Pipeline & Code Grid Renderer Summary

**VQ-VAE encode/decode/preview pipeline with interactive HTML grid renderer using CSS grid, tab20 coloring, JS click bridge, and playhead animation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-28T00:17:33Z
- **Completed:** 2026-02-28T00:22:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Five pipeline functions (encode, decode, preview_single_code, preview_time_slice, play_row_audio) for complete code inspection workflow
- Interactive HTML grid renderer with per-cell coloring, JS click bridge, playhead animation, and sticky row labels
- Cascading level label scheme (Structure/Detail for 2L, Structure/Timbre/Detail for 3L, Structure/Timbre/Texture/Detail for 4L)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create inference/codes.py with encode/decode pipeline and public API exports** - `dd58e10` (feat)
2. **Task 2: Create code_grid.py HTML renderer with level labels, cell coloring, JS bridge, and playhead** - `1fd01f0` (feat)

## Files Created/Modified
- `src/distill/inference/codes.py` - Encode/decode/preview pipeline (encode_audio_file, decode_code_grid, preview_single_code, preview_time_slice, play_row_audio)
- `src/distill/ui/components/code_grid.py` - Interactive HTML grid renderer with CSS grid, JS bridge, playhead, sticky labels
- `src/distill/inference/__init__.py` - Added exports for all five codes.py functions

## Decisions Made
- Used matplotlib tab20 colormap (20 distinct colors cycling via modulo) for visually distinguishing code indices without excessive palette size
- Applied ITU-R BT.709 luminance formula for white/black text color selection on colored cell backgrounds
- Implemented playhead via pure CSS @keyframes animation with JS class toggle (no setInterval timers needed)
- Column width 26px per data column with 80px sticky label column -- balances readability with information density

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Pipeline functions and grid renderer ready for Plan 02 (Codes tab) to wire together
- JS bridge events (cell, col, row) ready for Gradio event handlers
- Playhead start/stop JS functions available for audio playback sync

## Self-Check: PASSED

All created files verified on disk. All commit hashes found in git log.

---
*Phase: 16-encode-decode-code-visualization*
*Completed: 2026-02-28*
