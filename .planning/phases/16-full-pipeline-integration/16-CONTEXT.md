# Phase 16: Full Pipeline Integration - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the new 2-channel ISTFT pipeline (magnitude + IF) through all user-facing workflows: generation, export, training previews, PCA-based latent space analysis, Gradio UI, and CLI. All v1.0 code paths are removed — this is a clean break, not backward-compatible.

</domain>

<decisions>
## Implementation Decisions

### v1.0 Model Handling
- Clean break: v1.0 models will not load at all
- Simple error message when incompatible model is detected: "Incompatible model format. Please retrain with current version."
- Delete all v1.0-specific code paths — no dead code, no disabled branches
- Old model files on disk can be deleted; no cleanup utility needed

### PCA Slider Behavior
- PCA analysis requires manual trigger — same workflow as v1.0 (user explicitly runs analysis)
- Slider labels use technical + musical style: e.g., "PC1 (Brightness)", "Mag-Texture"
- Labels should reflect what PCA components actually correlate with in the 2-channel space

### Training Previews
- Preview audio files saved to same location and naming scheme as v1.0
- Add IF coherence / spectral quality metrics to console log output alongside loss values
- Audio-only preview files (no metrics files saved alongside)

### Migration Experience
- No onboarding notification or first-run message — just works (or errors on old models)
- Clean break on training config format — new schema, old v1.0 configs won't load
- Old cached preprocessed spectrograms (1-channel) are ignored naturally — pipeline recomputes 2-channel format, old cache files sit unused
- No migration utility or version detection on configs

### Claude's Discretion
- Number of PCA sliders to expose (based on explained variance in 128-dim space)
- Training preview generation frequency (may increase since ISTFT is faster than Griffin-Lim)
- Specific IF/spectral quality metrics to log (phase coherence, spectral convergence, etc.)
- Export default settings (whether ISTFT pipeline benefits from different sample rate or bit depth)
- Exact PCA slider labels (determine from what components actually correlate with sonically)

</decisions>

<specifics>
## Specific Ideas

- Slider labels should combine technical and musical: show the PCA component AND a descriptive name based on sonic effect
- Console-logged metrics during training should appear alongside existing loss values — same log format, just additional fields

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 16-full-pipeline-integration*
*Context gathered: 2026-02-27*
