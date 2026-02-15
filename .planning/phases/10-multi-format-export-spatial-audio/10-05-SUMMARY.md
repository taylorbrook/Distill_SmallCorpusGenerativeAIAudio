---
phase: 10-multi-format-export-spatial-audio
plan: 05
subsystem: cli
tags: [cli, typer, format-export, spatial-audio, blending, metadata, backward-compat]

# Dependency graph
requires:
  - phase: 10-01-multi-format-export
    provides: "ExportFormat enum, export_audio dispatcher, metadata embedding"
  - phase: 10-02-spatial-audio
    provides: "SpatialMode, SpatialConfig, apply_spatial, migrate_stereo_config"
  - phase: 10-03-multi-model-blending
    provides: "BlendEngine, ModelSlot, BlendMode"
provides:
  - "CLI --format/-f option for wav/mp3/flac/ogg export"
  - "CLI --spatial-mode (mono/stereo/binaural) with --spatial-width and --spatial-depth"
  - "CLI --blend/-b for multi-model blending via MODEL:WEIGHT pairs"
  - "CLI --artist/--album/--title metadata override options"
  - "Backward-compatible --stereo flag with deprecation warning"
  - "GenerationPipeline.export() updated for multi-format with metadata"
  - "GenerationConfig.spatial field with get_spatial_config() migration"
  - "Full Phase 10 public API re-exports (export_mp3, export_flac, export_ogg)"
affects: [ui-export, scripting-workflows, history-reexport]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CLI spatial post-processing: generate mono, apply spatial after"
    - "Blend argument parsing with MODEL:WEIGHT split on last colon"
    - "Deprecated flag migration via migrate_stereo_config in CLI"
    - "Export helper function for format-agnostic file output"

key-files:
  modified:
    - src/small_dataset_audio/cli/generate.py
    - src/small_dataset_audio/inference/__init__.py

key-decisions:
  - "Spatial processing applied post-generation (generate mono, then apply_spatial) for clean separation"
  - "Deprecated --stereo flag maps to --spatial-mode only when spatial_mode is still default ('mono')"
  - "Blend primary model gets weight=50.0 by default alongside user-specified blend models"
  - "Export helper writes sidecar JSON before audio file (research pitfall #6 pattern)"
  - "JSON output includes 'format' field for script consumers"

patterns-established:
  - "CLI blend argument parsing: split on last colon for MODEL:WEIGHT"
  - "CLI spatial post-processing: mono generation + spatial post-step"
  - "Deprecation pattern: old flag works silently, prints [yellow]Warning[/yellow]"

# Metrics
duration: 5min
completed: 2026-02-15
---

# Phase 10 Plan 05: CLI Integration for Multi-Format Export, Spatial Audio, and Blending Summary

**CLI generate command with --format (wav/mp3/flac/ogg), --spatial-mode (mono/stereo/binaural), --blend (MODEL:WEIGHT), and metadata override options with backward-compatible --stereo deprecation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-15T02:14:18Z
- **Completed:** 2026-02-15T02:19:07Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- CLI generate command supports all Phase 10 features: format selection, spatial audio, multi-model blending, and metadata overrides
- Backward-compatible --stereo flag with deprecation warning, migrating to new SpatialConfig via migrate_stereo_config
- Full public API re-exports including export_mp3, export_flac, export_ogg in inference/__init__.py
- Auto-generated filenames use correct extension per export format (gen_{timestamp}_seed{seed}.{ext})

## Task Commits

Each task was committed atomically:

1. **Task 1: Update sda generate with format, spatial, and blend CLI options** - `19eb47b` (feat)
2. **Task 2: Final integration verification and public API cleanup** - `47fc84a` (feat)

## Files Created/Modified
- `src/small_dataset_audio/cli/generate.py` - Added --format, --spatial-mode, --spatial-width, --spatial-depth, --blend, --artist, --album, --title options; deprecated --stereo; added _parse_blend_arg, _apply_spatial_post, _export_result helpers
- `src/small_dataset_audio/inference/__init__.py` - Added export_mp3, export_flac, export_ogg to public API re-exports and __all__

## Decisions Made
- Spatial processing applied post-generation rather than inline: the CLI generates mono audio via the pipeline, then applies spatial processing separately. This clean separation avoids modifying the generation pipeline's internal stereo handling for CLI use.
- Deprecated --stereo flag only maps to --spatial-mode when spatial_mode is still at its default value ("mono"). If user provides both, --spatial-mode takes precedence.
- Primary model in blend mode gets weight=50.0 by default, providing reasonable balance with user-specified blend models.
- JSON output changed key from "wav" to "file" and added "format" field to support multi-format awareness in scripting workflows.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Updated GenerationPipeline for spatial and multi-format support**
- **Found during:** Task 1 (CLI integration)
- **Issue:** GenerationPipeline.generate() still used old stereo_mode/stereo_width fields and export() only produced WAV. CLI integration required the pipeline itself to understand SpatialConfig and ExportFormat.
- **Fix:** Already addressed by Plan 10-04 (commit e036241). The generation.py was updated with SpatialConfig integration, get_spatial_config() migration, _generate_right_channel helper, and multi-format export() method. These changes were already committed before this plan started.
- **Files modified:** src/small_dataset_audio/inference/generation.py (pre-existing from 10-04)
- **Verification:** All backward compatibility tests pass; new spatial and format features verified
- **Committed in:** e036241 (prior plan)

---

**Total deviations:** 0 new (1 pre-existing dependency from 10-04 was already committed)
**Impact on plan:** No scope creep. All changes align with plan objectives.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All Phase 10 features are now accessible via both CLI and programmatic API
- CLI supports piping workflows: `sda generate model --format mp3 -n 5 | xargs ls -la`
- JSON output mode includes format information for script consumers
- Full backward compatibility maintained with Phases 4-9 functionality
- Phase 10 (final phase) is now complete across all 5 plans

## Self-Check: PASSED

- FOUND: src/small_dataset_audio/cli/generate.py
- FOUND: src/small_dataset_audio/inference/__init__.py
- FOUND: commit 19eb47b (Task 1)
- FOUND: commit 47fc84a (Task 2)

---
*Phase: 10-multi-format-export-spatial-audio*
*Completed: 2026-02-15*
