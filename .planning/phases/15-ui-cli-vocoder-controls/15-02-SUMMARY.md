---
phase: 15-ui-cli-vocoder-controls
plan: 02
subsystem: cli
tags: [typer, vocoder, cli, rich-progress, tqdm, json-output]

# Dependency graph
requires:
  - phase: 15-ui-cli-vocoder-controls
    plan: 01
    provides: resolve_vocoder() function, tqdm_class parameter through vocoder chain
  - phase: 14-generation-pipeline
    provides: GenerationPipeline with vocoder parameter
provides:
  - --vocoder CLI flag (auto/bigvgan/hifigan) on distill generate command
  - Vocoder status line printed to stderr on every generate call
  - Rich download progress bar for BigVGAN first-time download via tqdm_rich
  - JSON output vocoder field with name and selection
  - Clean error exit for --vocoder hifigan on models without per-model vocoder
affects: [16-hifigan-training]

# Tech tracking
tech-stack:
  added: []
  patterns: [tqdm-rich-for-cli-downloads, vocoder-status-stderr, json-vocoder-field]

key-files:
  created: []
  modified:
    - src/distill/cli/generate.py

key-decisions:
  - "Blend mode vocoder resolution for status/JSON only: BlendEngine creates its own vocoder internally, so resolve_vocoder in CLI is for status line and JSON field only"
  - "Rich progress disabled in JSON output mode: tqdm_cls=None when --json flag active to avoid polluting machine-readable output"
  - "TqdmExperimentalWarning suppressed globally via warnings.filterwarnings for clean CLI output"

patterns-established:
  - "Vocoder status line pattern: always print vocoder selection info to stderr with label + reason"
  - "CLI download progress: use tqdm_rich for Rich-styled progress bars, with ImportError fallback"

requirements-completed: [CLI-01, CLI-03]

# Metrics
duration: 2min
completed: 2026-02-28
---

# Phase 15 Plan 02: CLI Vocoder Controls Summary

**--vocoder flag with auto/bigvgan/hifigan selection, Rich-styled download progress via tqdm_rich, vocoder status line to stderr, and JSON vocoder field on distill generate command**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-28T00:25:42Z
- **Completed:** 2026-02-28T00:28:07Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added `--vocoder` option (auto/bigvgan/hifigan) to `distill generate` command with auto as default
- Replaced hardcoded `get_vocoder("bigvgan")` with `resolve_vocoder()` for user-selectable vocoder
- Vocoder status line printed to stderr on every generate call: `Vocoder: BigVGAN Universal (auto -- no per-model vocoder)`
- Rich-styled download progress via `tqdm_rich` for BigVGAN first-time download (disabled in JSON mode)
- JSON output includes `"vocoder": {"name": "bigvgan_universal", "selection": "auto"}` field
- Clean error exit via `typer.BadParameter` when `--vocoder hifigan` used on model without per-model vocoder
- Blend mode path also resolves vocoder for status line and JSON field consistency

## Task Commits

Each task was committed atomically:

1. **Task 1: Add --vocoder flag with resolve_vocoder, Rich progress, and JSON field** - `1434c3a` (feat)

**Plan metadata:** (pending)

## Files Created/Modified
- `src/distill/cli/generate.py` - Added --vocoder flag, resolve_vocoder() integration, tqdm_rich download progress, vocoder status line, JSON vocoder field, blend mode vocoder resolution

## Decisions Made
- **Blend mode vocoder resolution scope:** BlendEngine creates its own vocoder internally via `get_vocoder("bigvgan")`. Rather than refactoring the blend engine (out of scope for this plan), resolve_vocoder is called in the CLI blend path for the status line and JSON field only. The actual vocoder used by the blend engine remains its internal one. Refactoring BlendEngine to accept an external vocoder is Phase 16 territory.
- **Rich progress disabled in JSON mode:** When `--json` flag is active, `tqdm_cls` is set to None to avoid Rich progress output polluting machine-readable JSON stdout.
- **Warning suppression:** `TqdmExperimentalWarning` from `tqdm.rich.tqdm_rich` suppressed via `warnings.filterwarnings` for clean CLI experience.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- **Stale editable install:** The `distill` package was editable-installed from a different project directory (`Distill-vqvae`), causing `inspect.signature()` to see a different version at runtime. Verified using AST source analysis and PYTHONPATH override instead. Not a code issue, just an environment configuration matter.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 15 complete: both UI and CLI vocoder controls implemented
- resolve_vocoder() used by both UI (Plan 01) and CLI (Plan 02)
- tqdm_class parameter flows through entire vocoder chain for custom progress in any context
- Ready for Phase 16: HiFi-GAN per-model training will activate the "hifigan" selection path

## Self-Check: PASSED

- Source file src/distill/cli/generate.py verified present
- Commit 1434c3a (Task 1) verified in git log
- SUMMARY.md verified at expected path

---
*Phase: 15-ui-cli-vocoder-controls*
*Completed: 2026-02-28*
