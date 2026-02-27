# Phase 15: Generation Pipeline - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the trained autoregressive prior into the generation pipeline — prior samples code sequences, VQ-VAE decodes to mel, vocoder converts to audio. Expose through Gradio Generate tab and CLI with sampling controls and multi-chunk stitching. v1.0 continuous generation is not relevant — this phase is prior-only.

</domain>

<decisions>
## Implementation Decisions

### Sampling defaults & ranges
- Default temperature: 1.0 (neutral)
- Temperature slider range: 0.1 – 2.0
- Default top-p: 0.9 (nucleus sampling) — adapts to context, good for small codebooks (64–256)
- Top-k: off by default
- Top-p slider range: 0 – 1.0 (0 = disabled)
- Top-k slider range: 0 – 512 (0 = disabled)

### Multi-chunk stitching
- User controls output duration via a duration slider (not chunk count)
- Default duration: ~10 seconds (2-3 chunks stitched)
- Maximum duration: 30 seconds
- Crossfade/overlap amount is user-configurable (advanced control exposed in UI)

### Generate tab integration
- Leave v1.0 continuous generation behind — not relevant, will not be used
- Generate tab shows prior-based controls only (temperature, top-k, top-p, duration)
- No sub-tabs or mode switching needed

### Generation feedback
- Progress bar with chunk counter: "Generating chunk 2/4..."
- Simple and clear, no chunk-by-chunk audio preview

### Claude's Discretion
- Crossfade default overlap amount and range
- Error state handling (no prior trained, generation fails mid-chunk)
- Exact slider step sizes and UI layout

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-generation-pipeline*
*Context gathered: 2026-02-27*
