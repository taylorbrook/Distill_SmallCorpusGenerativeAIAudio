# Phase 12: Vocoder Interface & BigVGAN Integration - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Abstract vocoder layer with BigVGAN-v2 as the default mel-to-waveform reconstruction. Every existing model gets dramatically better audio with zero additional training. This phase builds the vocoder infrastructure (interface, BigVGAN integration, mel adapter, auto-download); wiring it through all generation paths is Phase 14, UI/CLI controls are Phase 15.

</domain>

<decisions>
## Implementation Decisions

### Vendoring strategy
- Full BigVGAN package copy into `vendor/bigvgan/` (top-level vendor directory, separated from project source)
- All model variants, training code, and configs included — nothing stripped
- BigVGAN's LICENSE file preserved as-is in the vendored copy
- Version pinning approach: Claude's discretion (commit hash file or submodule)

### Weight caching & download
- Cache location: Claude's discretion (user-global like `~/.cache/distill/` or HuggingFace cache — pick what works best)
- Download UX: Claude's discretion (blocking with progress vs background — pick the simpler approach)
- Cache management tooling: Claude's discretion
- Fully offline after initial download — no network connectivity required once weights are cached, app never phones home or checks for updates
- Model: `bigvgan_v2_44khz_128band_512x` (the one best model, per REQUIREMENTS.md)

### Transition behavior
- No Griffin-Lim fallback needed — if BigVGAN isn't available, it's an error not a graceful degradation
- Default vocoder timing: Claude's discretion (Phase 12 vs Phase 14 switchover)
- 44.1kHz → 48kHz resampling ownership: Claude's discretion (vocoder layer vs generation pipeline)
- No audio quality comparison mechanism — trust BigVGAN's established quality; if mel adapter is correct, output will be better

### Claude's Discretion
- Version pinning mechanism (commit hash file vs git submodule)
- Weight cache location (user-global vs HuggingFace cache)
- Download UX pattern (blocking with progress vs background)
- Cache management CLI commands (if any)
- Default vocoder activation timing (Phase 12 immediate vs Phase 14 wiring)
- Resampling layer ownership (vocoder returns 48kHz vs pipeline handles it)
- Vocoder interface abstraction depth (enough for BigVGAN + HiFi-GAN)
- Device selection logic (auto-detect best available: CUDA > MPS > CPU)
- Mel adapter implementation details (log1p → log-clamp conversion)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. User trusts BigVGAN's quality and wants a clean, full vendor copy with offline-capable weight caching.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 12-vocoder-interface-bigvgan-integration*
*Context gathered: 2026-02-21*
