---
phase: 03-core-training-engine
plan: 03
subsystem: training
tags: [checkpoint, audio-preview, torch-save, griffin-lim, wav, soundfile]

# Dependency graph
requires:
  - phase: 03-01
    provides: "ConvVAE model with encode/decode/sample methods, AudioSpectrogram with mel_to_waveform"
provides:
  - "save_checkpoint / load_checkpoint for full training state persistence"
  - "manage_checkpoints retention policy (3 recent + 1 best = 4 max)"
  - "JSON sidecar files for fast checkpoint metadata scanning"
  - "generate_preview for random-sample WAV generation from VAE decoder"
  - "generate_reconstruction_preview for original vs decoded comparison"
  - "list_previews for epoch-sorted preview timeline"
affects: [03-04, 04-training-ui, 05-generation]

# Tech tracking
tech-stack:
  added: []
  patterns: ["JSON sidecar for lightweight metadata alongside heavy .pt files", "peak normalization before WAV export"]

key-files:
  created:
    - src/small_dataset_audio/training/checkpoint.py
    - src/small_dataset_audio/training/preview.py
  modified: []

key-decisions:
  - "JSON sidecar (.meta.json) alongside .pt checkpoints for fast scanning without loading full state dicts"
  - "Retention policy keeps 3 recent + 1 best-val-loss; when best overlaps recent, total is 3 not 4"
  - "Peak normalization before 16-bit WAV output prevents clipping from untrained decoder"
  - "Reconstruction previews limited to 2 items per epoch to control disk usage"

patterns-established:
  - "JSON sidecar pattern: lightweight metadata file alongside heavy binary for fast scanning"
  - "Model mode save/restore: eval mode for generation, restore original training state after"
  - "Per-file error isolation in preview generation (one failure does not stop training)"

# Metrics
duration: 2min
completed: 2026-02-13
---

# Phase 3 Plan 3: Checkpoint & Preview Summary

**Checkpoint persistence with JSON sidecar metadata scanning and WAV preview generation from VAE decoder via InverseMelScale + GriffinLim**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-13T00:57:09Z
- **Completed:** 2026-02-13T00:59:47Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Full training state checkpoint system with save/load/manage lifecycle and version compatibility
- JSON sidecar files enable fast checkpoint scanning without loading multi-megabyte .pt files
- WAV preview generation from random latent vectors and reconstruction comparison pairs
- Epoch-sorted preview listing supporting scrollable timeline UI

## Task Commits

Each task was committed atomically:

1. **Task 1: Checkpoint save, load, and retention management** - `cac1377` (feat)
2. **Task 2: Audio preview generation from VAE decoder** - `a166feb` (feat)

**Plan metadata:** (pending final commit)

## Files Created/Modified
- `src/small_dataset_audio/training/checkpoint.py` - Save/load/manage checkpoints with JSON sidecar metadata, version validation, and retention policy
- `src/small_dataset_audio/training/preview.py` - Generate WAV previews from random latent samples and reconstruction comparisons, with epoch-sorted listing

## Decisions Made
- **JSON sidecar pattern:** Each `.pt` checkpoint gets a `.pt.meta.json` sidecar containing `{epoch, train_loss, val_loss}`. This avoids loading full state dicts (potentially hundreds of MB) just to compare validation losses during retention management.
- **Retention overlap handling:** When the best-val-loss checkpoint is already among the 3 most recent, the total retained count is 3, not 4. This is correct behavior -- the "4 max" is a ceiling, not a target.
- **Peak normalization:** Untrained VAE decoders produce arbitrary amplitude outputs. Peak normalization (`audio / max(abs(audio))`) before 16-bit WAV export prevents clipping and ensures audible output even from early training epochs.
- **Reconstruction preview limit:** Only 2 items per epoch to balance monitoring quality vs disk usage over long training runs.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Checkpoint and preview modules are ready for integration into the training loop (Plan 04)
- Training loop can call `save_checkpoint` after each epoch and `manage_checkpoints` to enforce retention
- Preview generation can be triggered every N epochs via `generate_preview`
- `list_previews` and `list_checkpoints` provide metadata for UI display in Phase 4

## Self-Check: PASSED

- FOUND: src/small_dataset_audio/training/checkpoint.py
- FOUND: src/small_dataset_audio/training/preview.py
- FOUND: .planning/phases/03-core-training-engine/03-03-SUMMARY.md
- FOUND: cac1377 (Task 1 commit)
- FOUND: a166feb (Task 2 commit)

---
*Phase: 03-core-training-engine*
*Completed: 2026-02-13*
