# Phase 12: RVQ-VAE Core Architecture - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the ConvVQVAE model with residual vector quantization that encodes mel spectrograms to discrete codes and decodes back. Includes dataset-adaptive codebook sizing, k-means initialization, EMA updates, dead code reset, and commitment loss. Training pipeline, UI, CLI, and prior are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Temporal resolution
- Claude's discretion on exact compression ratio — user prefers not to configure this
- Fixed at model level (not a user-configurable training parameter)
- Should balance region-level code editing (primary use) with occasional fine edits
- Must handle mixed clip lengths (1-30+ seconds) — chunking helps for longer clips

### Default model configuration
- Default RVQ levels: 3 (Structure / Timbre / Detail)
- Configurable range: 2-4 levels (per requirements)
- Default codebook dimension: 128
- Codebook size auto-scales with dataset size (64/128/256 per requirements)
- Commitment loss weight: Claude's discretion (user trusts best practice for small-dataset + EMA)

### Architecture reuse
- Fresh design — do NOT reuse v1.0 encoder/decoder architecture
- Clean break matches v1.1 philosophy (discrete paradigm, not continuous)
- Replace v1.0 model code entirely — remove SpectrogramVAE, replace with ConvVQVAE
- Old .sda model files will not load — confirmed acceptable (no backward compatibility)

### Reconstruction fidelity
- Quality-first architecture — deeper model, longer training is acceptable
- Fidelity priority depends on use: generation (prior) is more tolerant of artifacts, encode/edit/decode needs better reconstruction
- Multi-scale spectral loss for reconstruction (multiple STFT resolutions)
- Combined with commitment loss (single weight parameter, no KL divergence)

### Claude's Discretion
- Exact temporal compression ratio (guided by: medium resolution, region-level editing focus, mixed clip lengths)
- Commitment loss weight default
- Reconstruction fidelity balance for non-edited regions in encode/edit/decode
- Encoder/decoder depth and layer configuration
- Dead code reset thresholds and EMA decay rate

</decisions>

<specifics>
## Specific Ideas

- User wants this to feel like a clean break from v1.0 — new paradigm, new architecture, new model format
- Region-level code editing is the primary editing use case, with occasional fine edits
- Mixed clip lengths (1-30+ seconds) are typical — architecture must handle this via chunking
- 3-level RVQ (Structure/Timbre/Detail) as the default mental model for how levels are understood

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-rvq-vae-core-architecture*
*Context gathered: 2026-02-21*
