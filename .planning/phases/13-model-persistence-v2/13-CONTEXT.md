# Phase 13: Model Persistence v2 - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Update the model file format to support optional per-model vocoder state. The new `.distillgan` format replaces `.distill` entirely — no backward compatibility with v1. The model catalog is updated to reflect vocoder presence and training stats.

</domain>

<decisions>
## Implementation Decisions

### Migration behavior
- **No backward compatibility** — PERS-02 is dropped entirely
- v1 `.distill` files are completely ignored by the application (not recognized, not loaded, not scanned)
- Users must retrain models with the new code; no conversion tooling provided
- When someone attempts to load a v1 file directly (e.g., via CLI path argument), raise a clear error: "This model was saved in v1 format which is no longer supported. Please retrain your model."

### File extension & format marker
- **New extension:** `.distillgan` (replaces `.distill`)
- **New format marker:** `distillgan_model` (replaces `distill_model`)
- **Version:** starts at `1` for the new format (clean slate, not `2` of the old format)
- Training checkpoint files stay as `.pt` — only the user-facing saved model format changes
- Old `.distill` files are invisible to the app — treated like any non-model file

### Vocoder state bundling
- **Single file** — per-model vocoder weights are stored inside the `.distillgan` file alongside model weights, config, analysis, and metadata
- Store vocoder state_dict, model config (architecture params), AND training metadata (epochs, final loss, training date)
- File size will increase significantly when vocoder is bundled (~50-100MB+ vs ~6MB without)

### Catalog vocoder display
- Show vocoder training stats (epochs, loss) in the catalog — not just presence/absence
- No vocoder-based filtering in this phase — keep search as-is
- Vocoder stats visible alongside existing model metrics

### Claude's Discretion
- Whether to use null marker or omit vocoder key when no vocoder is present
- Whether to allow stripping vocoder state on re-save
- Catalog display approach (badge, column, or icon) — pick what fits existing UI patterns
- Nested sub-object vs flat fields in the JSON catalog index — pick the better data modeling approach

</decisions>

<specifics>
## Specific Ideas

- User explicitly chose `.distillgan` as the extension — it's a deliberate rebrand, not just a version bump
- The clean break from v1 is intentional: no migration code, no dual-format support, no conversion tools
- "Just retrain" is the migration path — models are cheap to retrain

</specifics>

<deferred>
## Deferred Ideas

- Vocoder-based filtering in model catalog — can add later if model count warrants it
- Batch conversion tool for v1→v2 — explicitly rejected, but could revisit if users complain

</deferred>

---

*Phase: 13-model-persistence-v2*
*Context gathered: 2026-02-21*
