# Phase 5: Musically Meaningful Controls - Context

**Gathered:** 2026-02-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Map latent space dimensions to perceptually meaningful musical parameters so users control generation via intuitive sliders instead of opaque latent vectors. Covers PCA-based discovery, parameter labeling, slider interface logic, and analysis persistence. UI rendering is Phase 8; model save/load is Phase 6.

</domain>

<decisions>
## Implementation Decisions

### Parameter mapping approach
- PCA discovery on trained latent space (not predefined targets)
- Adaptive slider count — expose as many PCA components as exceed a variance threshold (could be 3 or 12 depending on model)
- Neutral axis labels by default ("Axis 1", "Axis 2") with suggested labels based on audio feature correlation; user can accept or rename
- User-triggered analysis (not automatic after training) — user decides when to run "Analyze latent space"

### Slider behavior & ranges
- Stepped sliders (discrete positions, not continuous) for repeatability
- Soft warning zone at extreme values — visual indicator but slider still allows full range; no hard clamp
- "Randomize all" button to set all sliders to random positions within safe bounds
- No per-slider randomize (global only)

### Parameter categories
- All parameter families weighted equally — timbre, harmony, temporal, spatial all matter; no priority
- Acoustic terms for labels when accurate (spectral centroid, RMS energy, zero-crossing rate) — precision over producer jargon
- Fully independent sliders — each maps to exactly one PCA component, no coupling
- Global "reset to center" button (returns all sliders to latent space mean)
- No per-slider reset

### Discovery & calibration
- Analysis data: training data encodings + random prior samples combined for full coverage
- Graceful degradation — show whatever dimensions exist, even if only 1-2; warn user that more data might improve variety
- Analysis results (PCA mapping, labels, safe ranges) saved with the model checkpoint — instant slider restoration on load

### Claude's Discretion
- Slider visual feedback approach (label + value only, or label + descriptor, etc.)
- Whether to expose variance-explained percentages per slider
- Exact variance threshold for determining "meaningful" PCA components
- Compression algorithm for analysis data within checkpoint

</decisions>

<specifics>
## Specific Ideas

- User expressed preference for acoustic accuracy over musical abstraction in labeling — if "spectral centroid" is what a slider controls, call it that
- Exploration is key: randomize button + soft warning zones = encourage users to find edges of the space
- The tool should work even with poorly-trained models (1-2 meaningful dims) — degrade gracefully, don't gate access

</specifics>

<deferred>
## Deferred Ideas

- Coupled/correlated parameters (moving one slider subtly influences others for more natural musical control) — revisit after v1
- Per-slider randomize buttons — could add later if users want targeted exploration
- Per-slider reset-to-center — could add alongside per-slider randomize

</deferred>

---

*Phase: 05-musically-meaningful-controls*
*Context gathered: 2026-02-13*
