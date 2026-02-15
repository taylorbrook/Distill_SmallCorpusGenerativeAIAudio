# Phase 10: Multi-Format Export & Spatial Audio - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Export audio in multiple formats (MP3, FLAC, OGG) alongside existing WAV, generate spatial audio with stereo field and binaural output, and blend multiple models simultaneously. The existing stereo width parameter (Phase 4) is replaced by the new spatial control system.

</domain>

<decisions>
## Implementation Decisions

### Export format defaults
- All three formats (MP3, FLAC, OGG) are first-class citizens with equal priority
- MP3 default: 320 kbps CBR (maximum quality)
- FLAC compression: level 8 (max compression, smallest files)
- OGG quality: Claude's discretion for sensible default
- Format selection available everywhere audio is exported (generate tab, history, CLI) — not a separate export step

### Spatial audio controls
- Replaces the existing Phase 4 stereo width parameter (0.0-1.5) entirely
- Two control dimensions: width + depth (front-back)
- Output mode selector: stereo, binaural, or mono — spatial controls adapt to selected mode
- Binaural target: immersive headphone experience with full HRTF-based spatialization

### Multi-model blending
- Up to 4 models loaded simultaneously
- Individual weight slider per model (0-100%), auto-normalized to sum to 100%
- User toggle between latent-space blending and audio-domain blending
- Union of all sliders from loaded models — each slider maps to whichever models have that parameter

### Export metadata
- Key info embedded in audio file tags: model name, seed, preset name
- Full provenance retained in sidecar JSON (complements, doesn't replace Phase 4 pattern)
- Default tags: Artist = "SDA Generator", Album = model name
- Metadata fields are editable before export — user can override any tag

### Claude's Discretion
- OGG quality/bitrate default
- HRTF dataset selection for binaural rendering
- How union sliders handle models that don't share a parameter (zero-fill, skip, interpolate)
- Exact normalization behavior for blend weight sliders

</decisions>

<specifics>
## Specific Ideas

- User wants toggle between latent-space and audio-domain blending rather than a fixed approach — implies both should produce meaningfully different results
- "App-branded" metadata (Artist: SDA Generator) suggests the tool should have recognizable identity in exported files
- Replacing stereo width entirely (not extending) means Phase 4's stereo_width parameter should be migrated to the new spatial system

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 10-multi-format-export-spatial-audio*
*Context gathered: 2026-02-14*
