---
phase: 07-presets-generation-history
plan: 01
subsystem: presets
tags: [json, dataclass, uuid, atomic-write, virtual-folders, preset-management]

# Dependency graph
requires:
  - phase: 06-model-persistence-management
    provides: "JSON index + atomic write pattern (library/catalog.py), UUID-based IDs"
  - phase: 05-musically-meaningful-controls
    provides: "SliderState dataclass from controls/mapping.py"
provides:
  - "PresetEntry dataclass for model-scoped slider configuration presets"
  - "PresetManager with full CRUD (save/load/get/list/rename/update/delete)"
  - "Virtual folder management (create/rename/delete/list)"
  - "Config defaults for presets and history data paths"
affects: [07-02, 07-03, 08-gradio-interface]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Model-scoped JSON index (one presets.json per model_id directory)"
    - "Virtual folders via string field instead of filesystem directories"
    - "Seed None <-> JSON null explicit round-trip handling"

key-files:
  created:
    - src/small_dataset_audio/presets/manager.py
    - src/small_dataset_audio/presets/__init__.py
  modified:
    - src/small_dataset_audio/config/defaults.py

key-decisions:
  - "Copied _write_index_atomic pattern locally instead of importing from library/catalog.py (module independence)"
  - "Virtual folders stored as list of dicts with name+created (not just name strings) for future metadata"
  - "Case-insensitive folder name duplicate check to prevent confusing duplicates"

patterns-established:
  - "Model-scoped data directory: presets_dir / model_id / index.json"
  - "Virtual folder pattern: folder field on entries + folders list in index"

# Metrics
duration: 2min
completed: 2026-02-14
---

# Phase 7 Plan 1: Preset Management Summary

**Model-scoped PresetManager with full CRUD, virtual folder organization, and atomic JSON persistence following catalog.py pattern**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-14T03:06:05Z
- **Completed:** 2026-02-14T03:08:17Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- PresetEntry dataclass storing slider_positions + seed (no duration per locked decision)
- PresetManager with 8 preset operations (save/load/get/list/rename/update/delete/count) and 4 folder operations (create/rename/delete/list)
- Atomic JSON index writes with temp file + os.replace + .bak backup
- Seed None round-trips correctly through JSON null serialization
- Config defaults updated with presets and history data paths

## Task Commits

Each task was committed atomically:

1. **Task 1: Create PresetEntry dataclass and PresetManager with CRUD and folder management** - `8d40ad5` (feat)
2. **Task 2: Create presets __init__.py and update config defaults** - `10d2b92` (feat)

## Files Created/Modified
- `src/small_dataset_audio/presets/manager.py` - PresetEntry dataclass and PresetManager class with full CRUD + folder management
- `src/small_dataset_audio/presets/__init__.py` - Public API exports for preset module
- `src/small_dataset_audio/config/defaults.py` - Added presets and history path defaults

## Decisions Made
- Copied `_write_index_atomic` pattern locally rather than importing from `library/catalog.py` to maintain module independence
- Virtual folders stored as list of dicts with `name` + `created` fields (not just name strings) for future extensibility
- Case-insensitive folder name duplicate check prevents confusing near-duplicate folders

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- PresetManager API ready for consumption by Plan 07-02 (generation history) and Plan 07-03 (A/B comparison)
- Config defaults ready for both presets and history data directories
- Phase 8 (Gradio UI) can build preset management UI directly on this API

## Self-Check: PASSED

All files verified present on disk. All commit hashes verified in git log.

---
*Phase: 07-presets-generation-history*
*Completed: 2026-02-14*
