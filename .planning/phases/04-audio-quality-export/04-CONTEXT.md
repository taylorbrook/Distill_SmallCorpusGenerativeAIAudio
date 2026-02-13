# Phase 4: Audio Quality & Export - Context

**Gathered:** 2026-02-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate high-fidelity 48kHz/24-bit audio from trained VAE models without aliasing artifacts, and export as professional-quality WAV files with configurable sample rate, bit depth, and channel mode. This phase builds the generation pipeline, anti-aliasing, export system, and quality metrics. Full UI integration is Phase 8.

</domain>

<decisions>
## Implementation Decisions

### Generation duration & variations
- Freeform duration input (slider or numeric), not fixed presets
- Initial max duration: 60 seconds (architecture should support longer in the future)
- Chunk duration is configurable (currently 1.0s default), not locked
- For multi-chunk generation (>1 chunk), support BOTH concatenation modes:
  - Crossfade between chunks (reliable default)
  - Latent interpolation for evolving, continuous sound (experimental/smooth option)
- User selects which concatenation mode to use per generation

### Stereo output behavior
- Default output is mono
- User can opt into stereo per generation
- When stereo is selected, user chooses the method:
  - Simple stereo widening (mid-side / Haas effect)
  - Dual generation (two different seeds for L/R channels)

### Export defaults & configuration
- Default format: 48kHz / 24-bit WAV (professional production standard)
- One-click export with defaults; advanced settings (sample rate, bit depth, channel mode) available but tucked away
- Sidecar JSON alongside each exported .wav with full generation details (model name, parameters, seed, timestamp)
- No embedded WAV metadata tags (sidecar JSON is the metadata store)

### Quality feedback
- Spectral analysis (spectrogram / frequency plot) available on demand, not always shown
- Audio preview player supports both waveform and spectrogram views (toggle between them)
- Automatic quality score shown after each generation, based on:
  - Signal-to-noise ratio
  - Clipping detection
- No spectral coverage or flatness metrics (keep it focused)

### Claude's Discretion
- Stereo width control implementation (slider vs presets)
- Export file destination (project output folder vs save-as)
- Exact quality score presentation (numeric, letter grade, visual indicator)
- Number of variations in batch generation (if implemented)
- Anti-aliasing filter implementation details
- Crossfade overlap duration for chunk concatenation

</decisions>

<specifics>
## Specific Ideas

- User wants the option to eventually generate hours-long audio (capture in architecture, but 60s limit for v1)
- Both concatenation modes (crossfade + latent interpolation) available to the user, not just one
- Quality score should be practical: SNR + clipping detection, not academic metrics
- Stereo method is a user choice per generation, not a global setting

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 04-audio-quality-export*
*Context gathered: 2026-02-12*
