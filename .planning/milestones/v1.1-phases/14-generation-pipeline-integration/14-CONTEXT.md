# Phase 14: Generation Pipeline Integration - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the neural vocoder through all five generation code paths (single chunk, crossfade, latent interpolation, preview, reconstruction) so every audio output uses BigVGAN. Remove Griffin-Lim reconstruction entirely. Handle sample rate differences with optional resampling at the export boundary.

</domain>

<decisions>
## Implementation Decisions

### Resampling strategy
- High quality resampling (sinc/Kaiser) — no cheap linear interpolation
- Resampling is **optional**, not automatic — vocoder returns native 44.1kHz
- Resampling happens at the **export boundary**, not after vocoder inference
- Default export sample rate is **44.1kHz** (BigVGAN native)
- Simple 44.1kHz / 48kHz toggle for users who want 48kHz — no multi-rate dropdown

### Preview generation
- Preview uses the **full BigVGAN vocoder** — same path as final generation, no lighter alternative
- What you preview is what you get

### Crossfade & interpolation domain
- Crossfade blending happens in **mel space** (before vocoder) — blend mel spectrograms, then run vocoder once on the blended result
- Latent interpolation stays in **latent space** — interpolate between latent vectors, decode each to mel, then vocoder
- Reconstruction path uses the **full vocoder** — consistent quality, reconstruction metric reflects what users actually hear
- Crossfade overlap region size is **configurable** by the user

### Griffin-Lim removal
- Griffin-Lim is **removed entirely in Phase 14** — not deferred to Phase 16
- No fallback, no hidden code, no legacy path
- Neural vocoder is the only reconstruction method after this phase

### Fallback & error handling
- If BigVGAN weights aren't downloaded: **auto-download with progress**, then generate — no blocking error
- GPU inference failure (e.g., OOM): **warn the user**, then fall back to CPU inference
- Vocoder inference failure handling: Claude's discretion

### Claude's Discretion
- Preview output path (raw vocoder vs. through export pipeline)
- Preview latency UX (loading indicator threshold)
- Preview caching strategy (cache vocoder output vs. always regenerate)
- Vocoder inference error recovery strategy (chunk-and-retry vs. fail with message)
- Exact resampling library choice (torchaudio, soxr, etc.)

</decisions>

<specifics>
## Specific Ideas

- The system already has 48kHz as its standard — the switch to 44.1kHz default reflects BigVGAN's native rate and avoids unnecessary processing
- Mel-space crossfade was chosen for cleaner transitions (vocoder sees coherent mel frames rather than blended waveforms)
- User wants full vocoder quality everywhere — no compromises on preview or diagnostic paths

</specifics>

<deferred>
## Deferred Ideas

- Griffin-Lim removal was originally Phase 16 scope — pulling it into Phase 14 means Phase 16 can focus entirely on per-model HiFi-GAN training and auto-selection

</deferred>

---

*Phase: 14-generation-pipeline-integration*
*Context gathered: 2026-02-27*
