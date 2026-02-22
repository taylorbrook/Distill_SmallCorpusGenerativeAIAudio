# Phase 13: 2-Channel VAE Architecture - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Adapt the ConvVAE to accept 2-channel spectrograms (magnitude + instantaneous frequency) and produce 2-channel output with per-channel decoder activations. Default latent dimension changes to 128 (configurable). This is a clean break from v1.0 — old single-channel models are not supported. v1.0 code paths are removed.

</domain>

<decisions>
## Implementation Decisions

### IF output activation
- Decoder uses **Tanh** for the IF channel, bounding output to [-1, 1] to match IF normalization range
- Magnitude channel keeps **Softplus** (non-negative)
- Split-apply vs separate output heads: Claude's discretion
- Strict tanh vs scaled tanh: Claude's discretion

### Backward compatibility
- **Clean break** — v2.0 is 2-channel only, no v1.0 model loading support
- **Hard-code 2 channels** — `in_channels` is not a parameter; the VAE always expects 2 channels
- Old v1.0 models fail naturally on load (no special error handling)
- **Remove v1.0 single-channel code paths** — delete old waveform-to-mel training paths, single-channel dataset code, and the `complex_spectrogram` toggle (it's always 2-channel now)

### Channel interaction and network architecture
- **Fully shared convolutions** — both channels feed into the same conv stack from layer 1; the network learns cross-channel relationships
- **Scaled-up channel progression**: 2→64→128→256→512 (encoder), 512→256→128→64→2 (decoder)
- **5 conv layers** (up from 4) — 5 stride-2 layers = 32x spatial reduction; inputs padded to multiples of 32
- Approximately ~12M+ parameters (up from ~3M), managed by existing dropout/weight-decay overfitting presets

### Latent space sizing
- **Configurable latent_dim, default 128** — `latent_dim` remains a parameter in TrainingConfig, default changes from 64 to 128

### Claude's Discretion
- Split-apply activation layer vs separate output heads for per-channel activations
- Strict tanh vs scaled tanh (e.g., 1.05 * tanh) for IF channel
- Unified vs structured latent space (whether to impose any magnitude/IF separation in latent dims)
- Keep lazy linear init vs compute spatial dims from config
- Whether to save latent_dim in checkpoint metadata for validation on load
- 5th conv layer channel count (e.g., 512→1024 or keep at 512)
- IF activation configurability (hard-code tanh vs config option)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches within the decisions above.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-2-channel-vae-architecture*
*Context gathered: 2026-02-21*
