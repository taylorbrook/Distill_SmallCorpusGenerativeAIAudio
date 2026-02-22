# Phase 14: Multi-Resolution Loss - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Training uses perceptually grounded loss combining multi-resolution STFT loss (auraloss) with per-channel reconstruction loss weighted by magnitude. Loss term weights are configurable via training config. This phase adds the loss function and wires it into the existing training loop — it does not change the model architecture, data pipeline, or reconstruction path.

</domain>

<decisions>
## Implementation Decisions

### Loss weight configuration
- Grouped config section (nested), not flat top-level keys: `loss: {stft: {weight: ...}, reconstruction: {magnitude_weight: ..., if_weight: ...}, kl: {weight: ...}}`
- Per-channel reconstruction uses L1 (mean absolute error) loss
- STFT window sizes: Claude's discretion on which sizes and whether to expose in config

### Default balance priorities
- Default weights should favor spectral quality (auraloss STFT loss) over pixel-perfect spectrogram reconstruction
- STFT loss is the primary perceptual signal; per-channel reconstruction supports it
- KL divergence annealing and warmup scheduling: Claude's discretion based on what works for small-dataset VAEs
- Magnitude vs IF channel weight ratio: Claude's discretion based on literature/experiments

### Training observability
- Log a config summary at training start showing all loss weights and settings
- Add new loss component info alongside existing training progress display — don't replace the existing format
- Loss logging granularity: Claude's discretion on per-component vs grouped

### Stability handling
- Log a prominent warning if total loss increases for N consecutive epochs (divergence detection)
- NaN handling, gradient clipping, epsilon guards: Claude's discretion on safest approach for small-dataset training

### Claude's Discretion
- Multi-resolution STFT window sizes and whether they're configurable
- Magnitude weighting approach for IF loss (soft vs hard mask)
- KL annealing schedule vs fixed weight
- Loss warmup (reconstruction-only period before adding STFT loss)
- Gradient clipping behavior (new or keep existing)
- NaN detection and recovery strategy
- Epsilon guards in loss computation
- Loss logging granularity (per-component vs grouped)
- Default magnitude-to-IF weight ratio

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. The user's priority is that the loss produces perceptually good audio, with spectral quality taking precedence over spectrogram-level accuracy.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-multi-resolution-loss*
*Context gathered: 2026-02-21*
