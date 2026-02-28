---
phase: 16-per-model-hifigan-training-griffin-lim-removal
plan: 03
subsystem: vocoder
tags: [hifigan, gan-training, vocoder-trainer, inference, auto-selection, data-augmentation, cancel-resume]

# Dependency graph
requires:
  - phase: 16-per-model-hifigan-training-griffin-lim-removal
    plan: 01
    provides: "HiFiGAN V2 generator, discriminators, loss functions, config"
provides:
  - "VocoderTrainer class with full GAN training loop, cancel/resume, data augmentation"
  - "HiFiGANVocoder inference wrapper implementing VocoderBase"
  - "Auto-selection: per-model HiFi-GAN preferred over BigVGAN when vocoder_state exists"
  - "Low-epoch warning when vocoder trained < 20 epochs"
  - "VocoderEpochMetrics, VocoderPreviewEvent, VocoderTrainingCompleteEvent events"
affects: [16-04, 16-05, vocoder-pipeline, training-ui]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GAN alternating training: disc step (augmented inputs) then gen step (mel loss weight 45 + feature matching)"
    - "Discriminator input augmentation: random gain +/-3dB, noise injection SNR 30-50dB"
    - "Cancel-safe checkpoint: full training state saved into .distillgan on cancel_event"
    - "VocoderBase inference wrapper with log1p->expm1 mel conversion"

key-files:
  created:
    - src/distill/vocoder/hifigan/trainer.py
    - src/distill/vocoder/hifigan/vocoder.py
  modified:
    - src/distill/vocoder/hifigan/__init__.py
    - src/distill/vocoder/__init__.py

key-decisions:
  - "Discriminator LR set to 0.5x generator LR to prevent discriminator overfitting on small datasets"
  - "Mel loss weight 45 (matching original HiFi-GAN) for strong reconstruction signal"
  - "HiFiGANVocoder undoes log1p via expm1 before generator forward -- no MelAdapter needed"
  - "Auto-selection prefers per-model HiFi-GAN over BigVGAN universal when vocoder_state exists"
  - "Low-epoch warning threshold at 20 epochs (per CONTEXT.md locked decision)"
  - "Pre-load all audio into memory for small datasets (5-50 files) to avoid I/O bottleneck"

patterns-established:
  - "Vocoder trainer saves/loads checkpoint inside .distillgan via torch.load/torch.save round-trip"
  - "Preview audio emitted via callback at configurable interval (default every 20 epochs)"
  - "resolve_vocoder auto-selection priority: per-model HiFi-GAN > BigVGAN universal"

requirements-completed: [TRAIN-03, TRAIN-04, TRAIN-05, VOC-05]

# Metrics
duration: 4min
completed: 2026-02-28
---

# Phase 16 Plan 03: Training Loop & Inference Summary

**VocoderTrainer with alternating GAN training loop, discriminator augmentation, cancel-safe checkpoint persistence, and HiFiGANVocoder inference wrapper with auto-selection over BigVGAN**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-28T23:02:50Z
- **Completed:** 2026-02-28T23:06:57Z
- **Tasks:** 2
- **Files created:** 2
- **Files modified:** 2

## Accomplishments
- VocoderTrainer with complete GAN training loop: alternating discriminator/generator steps, mel loss (weight 45), feature matching, ExponentialLR schedulers
- Discriminator input augmentation (random gain +/-3dB, noise injection SNR 30-50dB) to prevent overfitting on small datasets
- Cancel-safe checkpoint saving with full training state (generator, MPD, MSD, optimizers, schedulers, epoch) into .distillgan files
- Resume from checkpoint restoring exact training state for continuation
- HiFiGANVocoder inference wrapper accepting VAE mels directly (expm1 to undo log1p, no MelAdapter needed)
- Auto-selection in resolve_vocoder: per-model HiFi-GAN preferred when vocoder_state exists, with low-epoch warning when < 20 epochs
- Removed all NotImplementedError from hifigan paths in vocoder package

## Task Commits

Each task was committed atomically:

1. **Task 1: Create VocoderTrainer with training loop, data pipeline, and cancel/resume** - `224ca69` (feat)
2. **Task 2: Create HiFiGANVocoder inference wrapper and update auto-selection** - `b5807c5` (feat)

## Files Created/Modified
- `src/distill/vocoder/hifigan/trainer.py` - VocoderTrainer class with GAN training loop, data pipeline, augmentation, cancel/resume, event dataclasses
- `src/distill/vocoder/hifigan/vocoder.py` - HiFiGANVocoder(VocoderBase) inference wrapper with expm1 mel conversion
- `src/distill/vocoder/hifigan/__init__.py` - Updated exports: VocoderTrainer, event classes, HiFiGANVocoder with lazy imports
- `src/distill/vocoder/__init__.py` - Updated get_vocoder (hifigan support), resolve_vocoder (auto-selection, low-epoch warning), removed NotImplementedError

## Decisions Made
- Discriminator LR at 0.5x generator LR (config.learning_rate * 0.5) -- prevents discriminator from winning too easily on small datasets
- Mel loss weight 45 (matching original HiFi-GAN paper) -- provides strong reconstruction signal alongside GAN losses
- HiFiGANVocoder applies expm1(clamp(mel, min=0)) to undo log1p normalization before generator -- trained on this format natively, no MelAdapter needed
- Auto-selection in resolve_vocoder returns per_model_hifigan over bigvgan_universal when vocoder_state is present
- Low-epoch warning threshold = 20 (per project CONTEXT.md locked decision)
- Audio files pre-loaded into memory by _VocoderDataset -- appropriate for small datasets (5-50 files), avoids disk I/O during training
- get_vocoder now accepts vocoder_state parameter for hifigan type construction

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Training loop ready for UI integration (Plan 04: training dashboard/CLI)
- HiFiGANVocoder plugs directly into generation pipeline via resolve_vocoder auto-selection
- Callback events (VocoderEpochMetrics, VocoderPreviewEvent, VocoderTrainingCompleteEvent) ready for training progress display
- All weight_norm operations MPS-compatible (no CUDA kernel dependencies)

## Self-Check: PASSED

All 4 source files and 1 summary file verified present. Both task commits (224ca69, b5807c5) confirmed in git log.

---
*Phase: 16-per-model-hifigan-training-griffin-lim-removal*
*Completed: 2026-02-28*
