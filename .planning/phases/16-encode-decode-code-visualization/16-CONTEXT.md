# Phase 16: Encode/Decode + Code Visualization - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Encode audio files into discrete code representations, view them as a labeled timeline grid, preview individual codebook entries as audio, and decode codes back to audio. No code editing (Phase 17), no diagnostics/heatmaps (Phase 18), no CLI encode/decode (Phase 18).

</domain>

<decisions>
## Implementation Decisions

### Grid visualization
- Cell coloring and content display: Claude's discretion (optimize for readability at typical grid sizes)
- Grid scrolls horizontally for audio longer than one screen width; all quantizer levels remain visible vertically
- Playback position indicator: a vertical playhead line sweeps across the grid during audio playback, showing which codes are currently sounding

### Codebook preview
- Click a cell = instant auto-play of that single code decoded through the VQ-VAE decoder (the "pure sound" of that codebook entry)
- Click a column header = instant auto-play of the full time-slice (all levels at that position decoded together)
- Clicked cell gets a visible selection highlight in the grid
- "Play row" button per level row: plays all codes in that row sequentially along the timeline, so the user can hear a single level's contribution

### Level labeling
- Fixed default labels, but user can rename them
- Cascading detail scheme based on number of RVQ levels:
  - 2 levels: Structure / Detail
  - 3 levels: Structure / Timbre / Detail
  - 4 levels: Structure / Timbre / Texture / Detail
- Labels appear as row headers on the left side of the grid (always visible)
- Row ordering (coarsest top vs bottom): Claude's discretion

### Tab & layout
- New top-level "Codes" tab alongside Train, Generate, etc. (Phase 17 editing will extend this tab)
- Layout: controls at top (upload audio, model selector dropdown, encode/decode buttons), code grid takes main area below
- Explicit model selector dropdown — no auto-detection, user picks a trained VQ-VAE model
- Side-by-side audio players: original audio on one side, decoded reconstruction on the other, for A/B quality comparison

### Claude's Discretion
- Cell coloring approach (color by code index, by level, or hybrid)
- Cell content display (color block only vs color + code number)
- Row ordering (coarsest at top or bottom)
- Exact grid cell sizing and spacing
- Playhead implementation approach within Gradio constraints
- How single-code audio preview is rendered (duration, windowing)

</decisions>

<specifics>
## Specific Ideas

- The playhead sweeping across the grid during playback creates a direct visual-to-audio mapping — this is the key interaction that makes the code grid feel alive rather than static
- "Play row" enables isolating what each quantizer level contributes, which is critical for building intuition about how RVQ levels work before the user starts editing in Phase 17
- Side-by-side A/B comparison lets the user immediately hear encode/decode quality, building confidence in the representation before they manipulate it

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 16-encode-decode-code-visualization*
*Context gathered: 2026-02-27*
