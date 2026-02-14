# Phase 8: Gradio UI - Context

**Gathered:** 2026-02-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Complete Gradio-based GUI wrapping all features from Phases 1-7. Users interact with dataset import, training, generation (sliders + playback), model management, presets, history, and A/B comparison through a web browser interface. No new capabilities — this phase surfaces existing backend functionality.

</domain>

<decisions>
## Implementation Decisions

### Page/tab layout
- 4 tabs: Data, Train, Generate, Library
- Guided navigation with overrides — tabs highlight the suggested next step, but user can click any tab; unready tabs show what's needed (e.g., "Import a dataset to start training")
- Generate tab includes history and A/B comparison — Claude's discretion on whether inline or collapsible sections, based on Gradio's component model

### Generation controls
- Sliders grouped by category (timbral, temporal, spatial) in 2-3 columns — more compact layout
- Explicit Generate button with spinner/progress; inline audio player appears when done (no auto-play)
- Preset recall approach: Claude's discretion (dropdown vs buttons — pick what fits Gradio best)
- Seed input visibility: Claude's discretion (visible vs advanced toggle — based on workflow prominence)

### Training experience
- Live loss chart (train + val curves) with stats panel (epoch, learning rate, ETA)
- Inline audio players for each preview epoch — user can play any preview to hear model progress over time
- Training config UI: Claude's discretion — optimize for best audio quality outcomes
- Cancel button during training + explicit "Resume Training" button when checkpoint exists for selected dataset

### Dataset import
- Drag-and-drop zone + browse button for file/folder import
- After import: stats panel at top (count, total duration, rate info) + grid of waveform thumbnails below, clickable to play

### Model library
- Dual view: card grid and table/list with a toggle between them
- Card grid as default view — each card shows name, dataset info, training date, sample count
- Table view as alternative — sortable columns, compact and scannable

### Audio export
- Export controls on Generate tab, next to audio player after generation
- Keeps the generate-listen-export flow together in one place

### Claude's Discretion
- Generate tab layout for history/A/B (inline vs collapsible sections)
- Preset recall component choice (dropdown vs button row)
- Seed input prominence (visible field vs advanced toggle)
- Training config UI design (optimized for audio quality outcomes)
- Exact spacing, typography, and visual polish within Gradio's constraints

</decisions>

<specifics>
## Specific Ideas

- Model library should support toggling between card grid and table view — user wants both presentation styles available
- Training previews should show progression — inline audio players per epoch, not just the latest one
- Tabs should feel guided (highlight next step) but never locked — always accessible with helpful empty states

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 08-gradio-ui*
*Context gathered: 2026-02-13*
