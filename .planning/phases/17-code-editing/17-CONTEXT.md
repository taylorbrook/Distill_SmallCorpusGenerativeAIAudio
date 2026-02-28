# Phase 17: Code Editing - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Interactive editing of discrete VQ-VAE codes in the Codes tab: changing individual cells, swapping regions between two audio files, blending in embedding space with per-region control, and full undo/redo. Extends the existing Codes tab from Phase 16 (encode/decode + visualization). No diagnostics/heatmaps (Phase 18), no CLI code editing.

</domain>

<decisions>
## Implementation Decisions

### Cell editing interaction
- Double-click a cell to enter inline edit mode — type a new codebook index directly in-place
- Input is constrained to valid codebook indices only (0 to codebook_size-1), impossible to enter out-of-range values
- No auto-play after editing — user manually triggers playback when ready to hear the change
- Multi-select (drag-select region, bulk edit): Claude's discretion on whether the complexity is worth it

### Region swapping
- Two audio files displayed as side-by-side grids, each with their own upload/encode controls
- Click-drag rectangle on a grid to select a region for swapping (time positions x quantizer levels)
- After swap, both grids update in place simultaneously showing the new code values (with brief highlight animation)
- Whether selected region sizes must match exactly between files: Claude's discretion on what produces good audio results

### Embedding blending
- Per-region + global blend control: select a region first, then a slider controls blend ratio for that region; also a global slider for whole-grid blending
- Multiple active regions can have different blend ratios, all adjustable before committing
- Real-time grid updates as the slider moves (cells update continuously during drag)
- Show both the snapped codebook index AND interpolation distance/confidence, so the user knows how close the blend landed to actual codebook entries

### Undo/redo
- Undo/Redo buttons in the UI toolbar plus Ctrl+Z / Ctrl+Y keyboard shortcuts
- Silent operation — no visible history panel or step counter, just the buttons and shortcuts
- Undo history resets when a new audio file is uploaded (new file = fresh start)
- Decode timing (auto vs manual): Claude's discretion on the right pattern

### Claude's Discretion
- Whether multi-select for bulk cell editing is worth the complexity
- Region size matching policy for swaps (exact match vs. truncate/pad)
- Decode trigger pattern (manual button vs. auto-decode after each edit)
- Exact highlight animation for swapped regions
- Keyboard shortcut conflict resolution with Gradio/browser defaults
- Side-by-side grid layout details (sizing, spacing, responsive behavior)

</decisions>

<specifics>
## Specific Ideas

- Embedding blending should expose interpolation distance so the user builds intuition about codebook density — when a blend lands far from any entry, that's informative about the embedding space structure
- Multiple active blend regions enable "painting with different strengths" — e.g., keep structure from File A but gradually blend in File B's texture across different time regions
- Side-by-side grids for swapping creates a visual "two-track editor" feel, making the swap operation intuitive (drag from one grid, apply to the other)
- Real-time blend updates make the slider feel responsive and experimental — the user can "scrub" through the blend space

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 17-code-editing*
*Context gathered: 2026-02-27*
