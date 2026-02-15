---
phase: 10-multi-format-export-spatial-audio
plan: 01
subsystem: audio-export
tags: [mp3, flac, ogg, lameenc, mutagen, id3, vorbis-comments, metadata]

# Dependency graph
requires:
  - phase: 04-generation-inference-pipeline
    provides: "WAV export with sidecar JSON, soundfile I/O pattern"
provides:
  - "ExportFormat enum (WAV, MP3, FLAC, OGG)"
  - "FORMAT_EXTENSIONS mapping"
  - "export_mp3 (lameenc 320kbps CBR)"
  - "export_flac (soundfile PCM_24, level 8)"
  - "export_ogg (soundfile Vorbis, quality 0.6)"
  - "export_audio unified dispatcher with metadata embedding"
  - "embed_metadata for ID3 (MP3) and Vorbis Comments (FLAC/OGG)"
  - "build_export_metadata helper for standard SDA metadata dicts"
affects: [10-02, 10-03, 10-04, 10-05, ui-export, cli-export, history-reexport]

# Tech tracking
tech-stack:
  added: [lameenc>=1.7, mutagen>=1.47]
  patterns: [format-dispatch via enum, lazy mutagen imports, ID3/Vorbis tag embedding]

key-files:
  created:
    - src/small_dataset_audio/audio/metadata.py
  modified:
    - src/small_dataset_audio/inference/export.py
    - src/small_dataset_audio/inference/__init__.py
    - pyproject.toml
    - uv.lock

key-decisions:
  - "OGG Vorbis quality default 0.6 (~192 kbps VBR) for good quality/size balance"
  - "lameenc quality=2 (highest encoding quality) for MP3"
  - "FLAC at PCM_24 subtype for professional 24-bit lossless"
  - "WAV embed_metadata is no-op (sidecar JSON only, per Phase 4 pattern)"
  - "Custom TXXX frames for SDA provenance in MP3 (SDA_SEED, SDA_MODEL, SDA_PRESET)"

patterns-established:
  - "Format dispatch via ExportFormat enum and export_audio dispatcher"
  - "Metadata embedding as post-encoding step (encode first, tag second)"
  - "build_export_metadata for standard SDA-branded metadata dict construction"

# Metrics
duration: 3min
completed: 2026-02-15
---

# Phase 10 Plan 01: Multi-Format Export Engine Summary

**MP3/FLAC/OGG export encoders with ExportFormat dispatch and mutagen-based metadata embedding (ID3 + Vorbis Comments)**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-15T02:07:01Z
- **Completed:** 2026-02-15T02:10:20Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- ExportFormat enum with WAV/MP3/FLAC/OGG and unified export_audio dispatcher
- MP3 encoding via lameenc at 320 kbps CBR with int16 PCM conversion and stereo interleaving
- FLAC export via soundfile at PCM_24 with level 8 compression
- OGG Vorbis export via soundfile at quality 0.6 (~192 kbps VBR)
- Metadata embedding: ID3 tags for MP3, Vorbis Comments for FLAC/OGG, no-op for WAV
- build_export_metadata helper with "SDA Generator" artist branding and user-overridable fields

## Task Commits

Each task was committed atomically:

1. **Task 1: Add dependencies and extend export.py with ExportFormat enum and MP3/FLAC/OGG encoders** - `0991545` (feat)
2. **Task 2: Create metadata.py for format-aware tag embedding via mutagen** - `f81b36b` (feat)

## Files Created/Modified
- `src/small_dataset_audio/inference/export.py` - ExportFormat enum, FORMAT_EXTENSIONS, export_mp3, export_flac, export_ogg, export_audio dispatcher
- `src/small_dataset_audio/audio/metadata.py` - embed_metadata (ID3/Vorbis), build_export_metadata helper, DEFAULT_METADATA
- `src/small_dataset_audio/inference/__init__.py` - Re-export ExportFormat, FORMAT_EXTENSIONS, export_audio
- `pyproject.toml` - Added lameenc>=1.7 and mutagen>=1.47 dependencies
- `uv.lock` - Updated lockfile with new dependencies

## Decisions Made
- OGG Vorbis quality default set to 0.6 (~192 kbps VBR) -- good balance of quality and file size for generated audio
- lameenc encoder quality set to 2 (highest quality encoding, slower but better output)
- FLAC uses PCM_24 subtype to match the project's 24-bit professional standard
- WAV metadata embedding is a no-op (preserves Phase 4 sidecar-only pattern)
- MP3 provenance uses custom TXXX frames (SDA_SEED, SDA_MODEL, SDA_PRESET) for non-standard metadata

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Export engine complete and ready for integration into UI format selector (plan 10-04/10-05)
- Metadata embedding ready for CLI --format flag and history re-export
- Existing export_wav unchanged, full backward compatibility maintained

## Self-Check: PASSED

- FOUND: src/small_dataset_audio/inference/export.py
- FOUND: src/small_dataset_audio/audio/metadata.py
- FOUND: commit 0991545 (Task 1)
- FOUND: commit f81b36b (Task 2)

---
*Phase: 10-multi-format-export-spatial-audio*
*Completed: 2026-02-15*
