# Phase 14: Autoregressive Prior - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Build and train an autoregressive prior model on frozen VQ-VAE code sequences. Includes the prior architecture, training loop with memorization detection, best-checkpoint tracking, model bundling into .sda files, and CLI training support. Generation UI/controls and sampling parameters are Phase 15.

</domain>

<decisions>
## Implementation Decisions

### Training trigger & flow
- User selects a saved .sda VQ-VAE model from a dropdown to train the prior on
- Prior training automatically uses the same dataset the VQ-VAE was trained on (encodes it to get code sequences)
- No separate dataset selection needed

### Memorization response
- Memorization detection uses relaxed sensitivity — only warn when memorization is very likely (small datasets naturally have low perplexity)
- On detection: show prominent warning with a "Stop and use best checkpoint" button, but don't force early stop
- System automatically tracks best checkpoint (lowest validation perplexity) throughout training — user can always roll back to pre-memorization weights

### Model bundling behavior
- Prior model state is bundled into the .sda file with a `has_prior` flag and metadata (epochs trained, final perplexity, training date)
- Loading a model with a prior auto-loads both VQ-VAE and prior — ready to generate immediately

### User-facing parameters
- Moderate complexity: epochs, hidden size, number of layers, number of attention heads
- Defaults auto-scale based on dataset size (smaller datasets get smaller priors to prevent overfitting)
- CLI flags mirror UI knobs exactly: `--epochs`, `--hidden-size`, `--layers`, `--heads`

### Claude's Discretion
- Training initiation UX (whether second stage in training tab or separate action)
- Training progress display (same chart with new lines vs separate section)
- Memorization threshold approach (fixed vs adaptive to dataset size)
- Model save behavior (update in-place vs create copy)
- Retrain prior behavior (overwrite vs warn-then-overwrite)
- Prior architecture choice (transformer vs LSTM vs other)
- Learning rate, optimizer, and training hyperparameters
- Adaptive default scaling curves for architecture params

</decisions>

<specifics>
## Specific Ideas

- Dataset-adaptive defaults should follow the same pattern as codebook sizing (64/128/256 tiers based on file count) — prior architecture scales similarly
- CLI prior training should work by pointing at a saved .sda model file (consistent with existing patterns)
- Best-checkpoint tracking means the user always has a safety net against memorization

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-autoregressive-prior*
*Context gathered: 2026-02-21*
