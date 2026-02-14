---
phase: 09-cli-interface
plan: 02
subsystem: cli
tags: [typer, cli, generate, model-management, batch, presets, rich-tables]

# Dependency graph
requires:
  - phase: 09-cli-interface
    provides: "Typer CLI skeleton with bootstrap(), subcommand registration"
  - phase: 04-generation-pipeline
    provides: "GenerationPipeline.generate() and .export() for audio generation"
  - phase: 06-model-persistence-management
    provides: "load_model, delete_model, ModelLibrary for model resolution"
  - phase: 07-presets-generation-history
    provides: "PresetManager for model-scoped preset loading"
  - phase: 05-musically-meaningful-controls
    provides: "SliderState, sliders_to_latent for preset-to-latent conversion"
provides:
  - "sda generate command with model resolution, batch generation, preset support"
  - "sda model list/info/delete commands with Rich tables and JSON output"
  - "resolve_model helper for name/ID/file path model lookup"
affects: [09-03-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns: ["resolve_model helper with 3-tier resolution (file path, UUID, name search)", "Console(stderr=True) for all Rich output, print() for machine-readable stdout", "Shared _find_model_entry helper for model commands"]

key-files:
  created:
    - "src/small_dataset_audio/cli/generate.py"
    - "src/small_dataset_audio/cli/model.py"
  modified:
    - "src/small_dataset_audio/cli/__init__.py"

key-decisions:
  - "resolve_model uses 3-tier lookup: .sda file path, UUID, then name search with ambiguity detection"
  - "Preset loading converts slider positions to latent vector via SliderState + sliders_to_latent"
  - "delete command uses persistence.delete_model (handles both file + index removal)"
  - "All Rich console output to stderr; stdout reserved for file paths or JSON (enables piping)"

patterns-established:
  - "CLI model resolution pattern: try file path, then ID, then name search with BadParameter on ambiguity"
  - "CLI output convention: stderr for Rich (progress/status), stdout for data (paths/JSON)"
  - "_find_model_entry shared helper across model subcommands for consistent resolution"

# Metrics
duration: 2min
completed: 2026-02-14
---

# Phase 9 Plan 2: Generate and Model Commands Summary

**sda generate with batch/preset support and sda model list/info/delete with Rich tables and JSON output**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-14T22:16:27Z
- **Completed:** 2026-02-14T22:19:04Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `sda generate MODEL` generates audio with full model resolution (name, ID, or .sda file path)
- Batch generation via `--count` with auto-incrementing seeds and preset support via `--preset`
- `sda model list` shows Rich table; `sda model info` shows detailed panel; `sda model delete` with `--force`
- All commands support `--json` for machine-readable stdout output

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement the generate command with model resolution, batch, and preset support** - `3724311` (feat)
2. **Task 2: Implement model management commands (list, info, delete)** - `b08586e` (feat)

## Files Created/Modified
- `src/small_dataset_audio/cli/generate.py` - sda generate command with resolve_model, batch loop, preset loading
- `src/small_dataset_audio/cli/model.py` - sda model list/info/delete with Rich tables and JSON output
- `src/small_dataset_audio/cli/__init__.py` - Removed try/except guards for generate and model imports

## Decisions Made
- Used 3-tier model resolution (file path -> UUID -> name search) for flexible model referencing
- Used persistence.delete_model() for delete command (handles both file deletion and index removal)
- Preset loading converts slider positions to latent vector via SliderState + sliders_to_latent pipeline
- All Rich output goes to stderr via Console(stderr=True), enabling `sda generate model | xargs` piping

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- generate and model commands complete; only train command remaining (Plan 03)
- CLI now covers the core headless workflow: find a model, generate audio
- All subcommands visible in `sda --help`: ui, generate, train, model

## Self-Check: PASSED

All files verified present. All commit hashes confirmed in git log.

---
*Phase: 09-cli-interface*
*Completed: 2026-02-14*
