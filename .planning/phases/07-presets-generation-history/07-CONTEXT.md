# Phase 7: Presets & Generation History - Context

**Gathered:** 2026-02-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Save slider configurations as named presets, browse generation history with parameter snapshots, and A/B compare generations. Presets are model-specific (tied to the model they were created with). The Gradio UI for these features is Phase 8 — this phase builds the data layer and API.

</domain>

<decisions>
## Implementation Decisions

### Preset organization
- Presets are model-specific — each preset belongs to one model (PCA axes differ per model)
- User-defined folders for organizing presets (e.g., "Pads", "Textures", "Percussion")
- Presets store slider positions + seed (no duration — duration is set per-generation)
- No audio preview on save — presets are parameter snapshots only, user loads and generates to hear

### History browsing
- Reverse-chronological list, most recent first
- Each entry shows: waveform thumbnail, preset name used (or "custom"), and timestamp
- Expand/click for full parameter details (slider values, seed, model, duration)
- Unlimited history retention — user manually deletes entries they don't want
- History entries store both the audio file (WAV) and full parameter snapshot — instant replay plus exact settings

### A/B comparison
- Toggle A/B button — single play control, toggle switches between A and B at the same playback position
- Audio-only comparison — no visual parameter diff (parameters visible in each entry's details separately)
- Default mode: compare current/latest generation against one history entry
- Also supports picking any two entries from history to compare
- "Keep this one" action after comparing — saves the winner's parameters as a preset

### Claude's Discretion
- Preset file format and storage structure
- History index implementation
- Waveform thumbnail generation approach for history entries
- Folder management UX details (create, rename, delete folders)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-presets-generation-history*
*Context gathered: 2026-02-13*
