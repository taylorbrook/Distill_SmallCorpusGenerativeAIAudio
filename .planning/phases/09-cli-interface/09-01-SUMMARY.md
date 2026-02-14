---
phase: 09-cli-interface
plan: 01
subsystem: cli
tags: [typer, cli, entry-point, bootstrap]

# Dependency graph
requires:
  - phase: 08-gradio-ui
    provides: "launch_ui/create_app accepting optional config/device"
  - phase: 01-project-foundation
    provides: "config.settings, hardware.device, validation.startup, app.first_run_setup"
provides:
  - "Typer CLI app with sda command and ui subcommand"
  - "bootstrap() function for shared config+device loading"
  - "Updated entry point (pyproject.toml -> cli:main)"
  - "Graceful subcommand registration with try/except for generate/train/model"
affects: [09-02-PLAN, 09-03-PLAN]

# Tech tracking
tech-stack:
  added: ["typer>=0.23,<1.0"]
  patterns: ["Typer callback with invoke_without_command=True for default GUI launch", "Lazy imports inside command functions for fast --help", "Module-level _cli_state dict for passing global options to subcommands"]

key-files:
  created:
    - "src/small_dataset_audio/cli/__init__.py"
    - "src/small_dataset_audio/cli/ui.py"
  modified:
    - "pyproject.toml"
    - "uv.lock"
    - "src/small_dataset_audio/__main__.py"

key-decisions:
  - "Typer with no_args_is_help=False and invoke_without_command=True callback for backward-compatible bare sda"
  - "Module-level _cli_state dict to pass --device/--verbose/--config from callback to subcommands"
  - "try/except ImportError for generate/train/model sub-typers so plan 01 works before plans 02/03"

patterns-established:
  - "CLI subcommand pattern: separate Typer app per module, registered via add_typer in __init__.py"
  - "bootstrap() returns (config, torch_device, config_path) tuple for any command needing config+device"
  - "All heavy imports (torch, hardware, ui) lazy inside function bodies for fast sda --help"

# Metrics
duration: 2min
completed: 2026-02-14
---

# Phase 9 Plan 1: CLI Skeleton Summary

**Typer CLI skeleton with bootstrap(), ui subcommand, and backward-compatible bare sda launching Gradio GUI**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-14T22:11:52Z
- **Completed:** 2026-02-14T22:13:52Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Typer installed and CLI framework functional with `sda --help` completing in <0.1s
- Bare `sda` (no args) launches Gradio GUI with full legacy flow (first-run, validation, device)
- `sda ui` subcommand launches GUI explicitly
- `bootstrap()` function provides shared config+device loading for all future commands
- Entry point updated in pyproject.toml and __main__.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Install Typer and create CLI package with bootstrap and app scaffold** - `d626994` (feat)
2. **Task 2: Update app.py and __main__.py for new CLI entry point** - `13ec2ac` (feat)

## Files Created/Modified
- `src/small_dataset_audio/cli/__init__.py` - Typer app, bootstrap(), callback, subcommand registration
- `src/small_dataset_audio/cli/ui.py` - sda ui command that launches Gradio via _launch_gui
- `pyproject.toml` - Added typer dependency, updated entry point to cli:main
- `uv.lock` - Lock file updated with typer dependency tree
- `src/small_dataset_audio/__main__.py` - Import changed from app:main to cli:main

## Decisions Made
- Used module-level `_cli_state` dict to share callback options with subcommands (simplest Typer pattern)
- Used `callback(invoke_without_command=True)` to detect bare `sda` and launch GUI (backward compatible)
- Used `try/except ImportError` for generate/train/model sub-typer registration so CLI works incrementally

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CLI skeleton ready for Plans 02 (generate/train commands) and 03 (model management commands)
- generate/train/model modules will be auto-registered when they exist (try/except pattern)
- bootstrap() available for all future commands to share config+device loading

## Self-Check: PASSED

All files verified present. All commit hashes confirmed in git log.

---
*Phase: 09-cli-interface*
*Completed: 2026-02-14*
