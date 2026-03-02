---
phase: 16-per-model-hifigan-training-griffin-lim-removal
plan: 05
subsystem: cli
tags: [hifigan, cli, typer, rich, vocoder-training, sigint, resume]

# Dependency graph
requires:
  - phase: 16-per-model-hifigan-training-griffin-lim-removal
    plan: 03
    provides: "VocoderTrainer class, event dataclasses, HiFiGANConfig"
provides:
  - "distill train-vocoder CLI command with Rich live progress table"
  - "SIGINT graceful cancellation with checkpoint save for vocoder training"
  - "Resume from checkpoint with interactive user prompt"
  - "Preview WAV file saving during headless/SSH training"
  - "JSON output mode for automated workflows"
affects: [cli, automation, scripting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Rich Live(Table) for in-place training metrics display (epoch, gen/disc/mel loss, LR, ETA)"
    - "CLI mirrors existing distill train command patterns: lazy imports, SIGINT handler, stderr console"
    - "Preview WAVs saved to disk alongside model file during training for headless monitoring"

key-files:
  created:
    - src/distill/cli/train_vocoder.py
  modified:
    - src/distill/cli/__init__.py

key-decisions:
  - "Command is distill train-vocoder (top-level subcommand, not nested under train) -- distinct operation from VAE training"
  - "CLI saves periodic audio preview WAVs to disk alongside model file for headless/SSH training review"
  - "Resume via --resume flag; if model has checkpoint and user doesn't pass --resume, prompts interactively"
  - "--force flag skips replacement confirmation for scripted/automated workflows"
  - "checkpoint_interval CLI param maps to preview_interval trainer param (trainer handles checkpoint saves on cancel/completion internally)"
  - "Full vocoder_state dict passed to trainer for resume (trainer extracts checkpoint internally)"

patterns-established:
  - "Vocoder CLI command follows same lazy-import + SIGINT + Rich pattern as VAE train command"
  - "Preview WAV files written as vocoder_preview_epochNNNN.wav in model parent directory"

requirements-completed: [CLI-02]

# Metrics
duration: 3min
completed: 2026-02-28
---

# Phase 16 Plan 05: Train-Vocoder CLI Command Summary

**distill train-vocoder CLI command with Rich live table progress, SIGINT checkpoint save, resume prompts, and preview WAV output for headless training**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-28T23:09:41Z
- **Completed:** 2026-02-28T23:12:15Z
- **Tasks:** 1
- **Files created:** 1
- **Files modified:** 1

## Accomplishments
- Created distill train-vocoder CLI command with all UI-mirrored parameters: --epochs, --lr, --batch-size, --checkpoint-interval
- Rich Live table display updating in-place showing epoch, gen/disc/mel loss, learning rate, and ETA
- SIGINT handler saves checkpoint gracefully with "Checkpoint saved at epoch N. Resume anytime." confirmation
- Interactive resume prompt when model has existing checkpoint (resume vs fresh choice)
- Replacement confirmation when model already has completed vocoder (bypassed with --force)
- Preview WAV files saved to disk during training for headless/SSH monitoring
- JSON output mode (--json) for machine-readable results in automated pipelines
- Registered as top-level distill train-vocoder subcommand alongside train, generate, model, ui

## Task Commits

Each task was committed atomically:

1. **Task 1: Create train-vocoder CLI command with Rich progress** - `e90f774` (feat)

## Files Created/Modified
- `src/distill/cli/train_vocoder.py` - Full CLI command with Typer, Rich Live table, SIGINT handling, resume logic, preview WAV saving, JSON output
- `src/distill/cli/__init__.py` - Registered train-vocoder subcommand via app.add_typer

## Decisions Made
- Command is `distill train-vocoder` (top-level, not nested under `train`) because vocoder training is a distinct operation from VAE model training
- CLI saves preview WAVs to disk alongside model file -- useful for headless/SSH training where user can check audio quality later
- Resume via `--resume` flag; if model has checkpoint and user doesn't pass --resume, prompt interactively for choice
- `--force` flag skips all confirmation prompts for scripted/automated workflows
- Adapted plan's `max_epochs` and `checkpoint_interval` parameters to match actual trainer API (`epochs` and `preview_interval`)
- Full vocoder_state dict passed to trainer for resume (not just the inner checkpoint dict) to match trainer's extraction pattern

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed trainer API parameter mismatch**
- **Found during:** Task 1 (CLI command implementation)
- **Issue:** Plan code called trainer.train() with `max_epochs=epochs` and `checkpoint_interval=checkpoint_interval`, but actual VocoderTrainer.train() uses `epochs=` and `preview_interval=` parameters
- **Fix:** Changed to `epochs=epochs` and `preview_interval=checkpoint_interval` to match actual trainer API
- **Files modified:** src/distill/cli/train_vocoder.py
- **Verification:** AST parse confirms function exists; help output shows all expected options
- **Committed in:** e90f774 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed checkpoint data extraction for resume**
- **Found during:** Task 1 (CLI command implementation)
- **Issue:** Plan code extracted inner checkpoint dict (`loaded.vocoder_state.get("checkpoint")`), but trainer expects full vocoder_state dict and extracts checkpoint internally via `checkpoint.get("checkpoint")`
- **Fix:** Pass `loaded.vocoder_state` directly as the checkpoint parameter
- **Files modified:** src/distill/cli/train_vocoder.py
- **Verification:** Code matches trainer.train() checkpoint handling pattern
- **Committed in:** e90f774 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correct API integration. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CLI command ready for use: `distill train-vocoder MODEL.distillgan AUDIO_DIR --epochs 200 --lr 0.0002`
- All 5 Phase 16 plans complete: HiFi-GAN V2 architecture, mel adapter, training loop, UI integration, and CLI command
- Per-model vocoder training available via both Gradio UI (Plan 04) and CLI (Plan 05)

## Self-Check: PASSED

All files verified present. Task commit (e90f774) confirmed in git log.

---
*Phase: 16-per-model-hifigan-training-griffin-lim-removal*
*Completed: 2026-02-28*
