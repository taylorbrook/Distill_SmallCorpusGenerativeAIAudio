# Phase 13: VQ-VAE Training Pipeline - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

End-to-end VQ-VAE training pipeline: wire the RVQ-VAE model (Phase 12) into the training engine, display codebook health during training, add VQ-VAE controls to the training UI, support CLI training with VQ flags, warn on low utilization, and save/load trained models in v2 .sda format with full metadata.

</domain>

<decisions>
## Implementation Decisions

### Codebook health display
- Inline per-level rows in the training metrics panel — each RVQ level gets its own row showing utilization %, perplexity, and dead code count
- CLI training output shows the same per-level detail (not a compact summary)
- Warning presentation when utilization drops below 30%: Claude's discretion on format (highlight, toast, or both)
- Update frequency for stats: Claude's discretion (balance responsiveness with performance)

### Training UI controls
- Layout/grouping of VQ-VAE controls in the training tab: Claude's discretion based on existing Gradio layout
- Codebook sizing is auto-only — no manual override in the UI. Size auto-scales from dataset size (64/128/256)
- RVQ levels: slider constrained to 2, 3, or 4 with descriptive labels ("Fewer levels = faster, more levels = finer detail")
- Commitment weight: pre-filled sensible default (e.g., 0.25) with slider/input to adjust — most users leave it alone

### Model save format (v2)
- Same .sda extension for v2 — differentiated internally by file header/magic bytes
- v1 models are left behind — do not support loading v1 .sda files at all
- v2 metadata includes: training config (codebook size, RVQ levels, commitment weight, codebook dim) + final codebook health snapshot per level + training loss curve history
- Codebook health display on load: Claude's discretion

### CLI training flags
- Codebook size: auto-determined by default, but --codebook-size flag available for power user override
- --rvq-levels and --commitment-weight flags with sensible defaults
- --model-name flag for custom naming; auto-generates timestamp-based name if omitted
- Output: tqdm-style progress bar per epoch with per-level codebook health stats at each update interval
- End-of-training: full summary printed — final loss, per-level codebook health, model save path, and any warnings

### Claude's Discretion
- Warning presentation format for low codebook utilization (30% threshold)
- Codebook health update frequency during training
- VQ-VAE controls layout/grouping in Gradio training tab
- Codebook health display behavior on model load
- Exact commitment weight default value
- Progress bar implementation details

</decisions>

<specifics>
## Specific Ideas

- CLI should feel like a power-user complement to the UI — same information, more control (codebook size override)
- Training summary at end should be a complete training report, not just a save path
- Codebook health is the key differentiator from v1 training — make it visible and prominent during training

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-vq-vae-training-pipeline*
*Context gathered: 2026-02-21*
