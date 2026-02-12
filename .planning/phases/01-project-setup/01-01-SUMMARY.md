---
phase: 01-project-setup
plan: 01
subsystem: infra
tags: [python, pyproject, uv, toml, hatchling, pytorch, makefile]

# Dependency graph
requires: []
provides:
  - "pip-installable package layout (src/small_dataset_audio) with hatchling build"
  - "pyproject.toml with torch/torchaudio/tomli-w/rich/psutil dependencies"
  - "Makefile with setup/run/test/lint/format/clean/benchmark targets"
  - "TOML configuration system (load_config, save_config, get_config_path, resolve_path)"
  - "DEFAULT_CONFIG with general, paths, hardware sections"
  - "All subpackage __init__.py stubs (config, hardware, models, training, audio, inference, ui, validation)"
  - "uv.lock for reproducible dependency resolution"
affects: [01-02, 01-03, 02-audio, 03-models, 03-training, 04-inference, 08-ui]

# Tech tracking
tech-stack:
  added: [torch 2.10.0, torchaudio 2.10.0, tomli-w 1.2.0, rich 14.3.2, psutil 7.2.2, hatchling, uv, ruff 0.15.0, pytest 9.0.2]
  patterns: [src-layout packaging, TOML config with deep merge defaults, Makefile task runner]

key-files:
  created:
    - pyproject.toml
    - Makefile
    - .gitignore
    - .python-version
    - uv.lock
    - src/small_dataset_audio/__init__.py
    - src/small_dataset_audio/config/defaults.py
    - src/small_dataset_audio/config/settings.py
  modified: []

key-decisions:
  - "Used dependency-groups.dev instead of deprecated tool.uv.dev-dependencies"
  - "TOML config uses 0/0.0 instead of None for unset numeric fields (TOML has no null type)"
  - "Config module has zero PyTorch dependency for error reporting when torch is broken"
  - "Deep merge strategy for config forward compatibility (new keys get defaults automatically)"

patterns-established:
  - "src layout: all code under src/small_dataset_audio/ with subpackage domains"
  - "Config pattern: load_config() returns deep-merged defaults + file, save_config() writes TOML"
  - "Path resolution: resolve_path() handles ~, relative, and absolute paths against project root"
  - "Makefile convention: PYTHON := uv run python, MODULE := small_dataset_audio"

# Metrics
duration: 3min
completed: 2026-02-12
---

# Phase 1 Plan 1: Project Scaffolding Summary

**pip-installable Python package with uv/hatchling build, PyTorch 2.10.0 deps, Makefile targets, and TOML configuration system with deep-merge defaults**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-12T22:26:54Z
- **Completed:** 2026-02-12T22:29:39Z
- **Tasks:** 2
- **Files modified:** 20

## Accomplishments
- Complete src-layout package with all 8 domain subpackages stubbed out
- pyproject.toml with PyTorch 2.10.0, TorchAudio 2.10.0, and platform-specific uv index config for CUDA/MPS
- TOML configuration system that loads defaults when no file exists, saves/reloads correctly, and deep-merges for forward compatibility
- Makefile with 8 targets (help, setup, run, run-verbose, test, lint, format, clean, benchmark)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create project scaffolding and packaging** - `6bddbe2` (feat)
2. **Task 2: Implement configuration system with TOML support** - `f69ad73` (feat)
3. **uv.lock committed for reproducibility** - `95157fb` (chore)

## Files Created/Modified
- `pyproject.toml` - Package metadata, dependencies, entry points, uv PyTorch index config
- `Makefile` - Common operations (setup, run, test, lint, format, clean, benchmark)
- `.gitignore` - Python, caches, user config, user data exclusions
- `.python-version` - Pinned to 3.11
- `uv.lock` - Lockfile for reproducible dependency resolution
- `src/small_dataset_audio/__init__.py` - Package root with __version__
- `src/small_dataset_audio/config/__init__.py` - Config subpackage
- `src/small_dataset_audio/config/defaults.py` - DEFAULT_CONFIG with general/paths/hardware sections
- `src/small_dataset_audio/config/settings.py` - load_config, save_config, get_config_path, resolve_path
- `src/small_dataset_audio/hardware/__init__.py` - Hardware subpackage stub
- `src/small_dataset_audio/models/__init__.py` - Models subpackage stub
- `src/small_dataset_audio/training/__init__.py` - Training subpackage stub
- `src/small_dataset_audio/audio/__init__.py` - Audio subpackage stub
- `src/small_dataset_audio/inference/__init__.py` - Inference subpackage stub
- `src/small_dataset_audio/ui/__init__.py` - UI subpackage stub
- `src/small_dataset_audio/validation/__init__.py` - Validation subpackage stub
- `tests/__init__.py` - Test package init
- `data/datasets/.gitkeep` - Dataset directory marker
- `data/models/.gitkeep` - Models directory marker
- `data/generated/.gitkeep` - Generated audio directory marker

## Decisions Made
- Used `dependency-groups.dev` instead of deprecated `tool.uv.dev-dependencies` (uv 0.7+ migration)
- TOML config uses `0` and `0.0` instead of `None` for unset numeric fields since TOML has no null type
- Config module intentionally avoids importing torch so config errors can be reported even when PyTorch is broken
- Deep merge strategy for config loading ensures forward compatibility when new config keys are added in future versions

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed deprecated tool.uv.dev-dependencies in pyproject.toml**
- **Found during:** Task 2 verification (deprecation warning from uv)
- **Issue:** `[tool.uv] dev-dependencies` is deprecated in uv 0.7+, produces warning on every uv command
- **Fix:** Migrated to `[dependency-groups] dev` (the current standard)
- **Files modified:** pyproject.toml
- **Verification:** uv commands run without deprecation warning
- **Committed in:** f69ad73 (Task 2 commit)

**2. [Rule 3 - Blocking] Committed uv.lock for reproducibility**
- **Found during:** Post-task verification
- **Issue:** uv.lock was generated but untracked; plan explicitly requires it be committed
- **Fix:** Added uv.lock to git
- **Files modified:** uv.lock
- **Verification:** File tracked in git, not in .gitignore
- **Committed in:** 95157fb (separate chore commit)

---

**Total deviations:** 2 auto-fixed (1 bug fix, 1 blocking)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Package structure ready for hardware detection (Plan 01-02) and environment validation (Plan 01-03)
- Config system ready to store benchmark results and device preferences
- All subpackage stubs in place for Phases 2-10 implementation
- `uv sync` resolves all dependencies cleanly on macOS (MPS via PyPI defaults)

## Self-Check: PASSED

All 20 created files verified present on disk. All 3 task commits (6bddbe2, f69ad73, 95157fb) verified in git history.

---
*Phase: 01-project-setup*
*Completed: 2026-02-12*
