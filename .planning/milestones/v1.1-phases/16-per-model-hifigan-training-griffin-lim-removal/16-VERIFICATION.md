---
phase: 16-per-model-hifigan-training-griffin-lim-removal
verified: 2026-02-28T23:45:00Z
status: passed
score: 15/15 must-haves verified
re_verification: false
---

# Phase 16: Per-Model HiFi-GAN Training & Griffin-Lim Removal Verification Report

**Phase Goal:** Users who want maximum fidelity can train a small per-model HiFi-GAN V2 vocoder on their specific audio, the system auto-selects the best available vocoder, and the legacy Griffin-Lim path is fully removed
**Verified:** 2026-02-28T23:45:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | HiFi-GAN V2 generator converts 128-band 48kHz mel spectrograms to waveforms | VERIFIED | `HiFiGANGenerator` with upsample_rates=[8,8,4,2] (product=512=hop_size), ConvTranspose1d stages, ResBlock1 with weight_norm, tanh output |
| 2  | MPD (periods [2,3,5,7,11]) and MSD (3 scales) discriminators classify real vs generated audio | VERIFIED | `MultiPeriodDiscriminator` wraps 5 `PeriodDiscriminator` instances; `MultiScaleDiscriminator` wraps 3 `ScaleDiscriminator` instances at raw/2x/4x downsampled scales |
| 3  | Loss functions compute adversarial, mel reconstruction (weight 45), and feature matching losses | VERIFIED | `generator_loss`, `discriminator_loss`, `feature_loss` pure functions in `losses.py`; mel weight 45 applied at `trainer.py:519` |
| 4  | AudioSpectrogram no longer has InverseMelScale or GriffinLim transforms | VERIFIED | `spectrogram.py` imports only `MelSpectrogram`; no `inverse_mel`, `griffin_lim`, or `mel_to_waveform` attributes; `grep` across entire `src/distill` returns zero GriffinLim/InverseMelScale hits |
| 5  | MelAdapter uses a direct mel-domain conversion instead of waveform round-trip through Griffin-Lim | VERIFIED | `mel_adapter.py` builds Tikhonov-regularized transfer matrix (HTK->Slaney) and uses `torch.matmul` + `F.interpolate` — no waveform or Griffin-Lim |
| 6  | No import of GriffinLim or InverseMelScale exists anywhere in the codebase | VERIFIED | `grep -r "GriffinLim\|InverseMelScale" src/distill` returns zero matches |
| 7  | BigVGAN vocoder path still works for models without per-model vocoder | VERIFIED | `MelAdapter.convert()` provides mel conversion; `resolve_vocoder("auto", model_without_vocoder_state)` returns BigVGAN with reason="no per-model vocoder" |
| 8  | User can train a HiFi-GAN vocoder given a model path and audio directory, with progress callbacks | VERIFIED | `VocoderTrainer.train()` accepts `model_path`, `audio_dir`, `callback`, `cancel_event`, `checkpoint`, `preview_interval`; emits `VocoderEpochMetrics`, `VocoderPreviewEvent`, `VocoderTrainingCompleteEvent` |
| 9  | Training applies data augmentation to discriminator inputs to prevent overfitting on small datasets | VERIFIED | `_augment_disc_input()` applies random gain +/-3dB and noise injection SNR 30-50dB; called on `wav_real_aug` and `wav_fake_aug` before discriminator forward passes |
| 10 | Cancel event triggers immediate checkpoint save inside the .distillgan model file | VERIFIED | `cancel_event.is_set()` checked after each batch and at epoch boundary; `_save_vocoder_state()` loads the `.distillgan` file via `torch.load`, injects `vocoder_state`, writes back via `torch.save` |
| 11 | Resume from checkpoint restores generator, discriminator, optimizers, schedulers to exact training state | VERIFIED | `trainer.py:422-433` restores `generator`, `mpd`, `msd`, `optim_g`, `optim_d`, `sched_g`, `sched_d` state dicts plus `start_epoch = ckpt["epoch"] + 1` |
| 12 | Trained vocoder weights are saved into the model's vocoder_state slot in .distillgan file | VERIFIED | `_build_vocoder_state()` builds dict with `generator_state_dict`, `config`, `training_metadata`; `_save_vocoder_state()` persists into model file |
| 13 | Auto-selection prefers per-model HiFi-GAN over BigVGAN when vocoder_state exists | VERIFIED | `resolve_vocoder("auto")` at `vocoder/__init__.py:161-181` creates `HiFiGANVocoder` when `has_per_model=True`, returns info with `name="per_model_hifigan"` |
| 14 | HiFiGANVocoder loads from vocoder_state and produces waveforms from VAE mels | VERIFIED | `HiFiGANVocoder.__init__` extracts config+weights from `vocoder_state`, `mel_to_waveform` squeezes channel dim, applies `expm1(clamp(mel, min=0))`, forwards through generator |
| 15 | User can run a CLI command to train a per-model HiFi-GAN vocoder | VERIFIED | `distill train-vocoder MODEL_PATH AUDIO_DIR` registered in `cli/__init__.py:205-207`; `train_vocoder_cmd` with `--epochs`, `--lr`, `--batch-size`, `--checkpoint-interval`, SIGINT handler, Rich Live table |

**Score:** 15/15 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/distill/vocoder/hifigan/config.py` | HiFiGANConfig dataclass with 128-band 48kHz defaults | VERIFIED | `HiFiGANConfig` dataclass, `__post_init__` validates upsample product = hop_size, all expected fields present |
| `src/distill/vocoder/hifigan/generator.py` | HiFi-GAN V2 generator with upsample_rates product=512 | VERIFIED | `class HiFiGANGenerator(nn.Module)`, `ResBlock1`, 4-stage transposed conv upsampling, weight_norm, `remove_weight_norm()` method |
| `src/distill/vocoder/hifigan/discriminator.py` | MPD and MSD discriminator implementations | VERIFIED | `class MultiPeriodDiscriminator`, `class MultiScaleDiscriminator`, both return 4-tuple `(real_out, fake_out, real_fmaps, fake_fmaps)` |
| `src/distill/vocoder/hifigan/losses.py` | generator_loss, discriminator_loss, feature_loss | VERIFIED | All three pure functions present, least-squares formulas, feature_loss multiplied by 2 |
| `src/distill/vocoder/hifigan/trainer.py` | VocoderTrainer with full GAN training loop | VERIFIED | 781 lines, full GAN alternation loop, discriminator augmentation, cancel/resume, checkpoint save, event dataclasses |
| `src/distill/vocoder/hifigan/vocoder.py` | HiFiGANVocoder(VocoderBase) inference wrapper | VERIFIED | `class HiFiGANVocoder(VocoderBase)`, `mel_to_waveform`, `sample_rate`, `to()` all implemented |
| `src/distill/vocoder/__init__.py` | Updated resolve_vocoder with auto-selection | VERIFIED | `resolve_vocoder` has auto-selection logic, `HiFiGANVocoder` in `__all__`, `get_vocoder("hifigan")` works with `vocoder_state` param |
| `src/distill/audio/spectrogram.py` | Forward-only AudioSpectrogram | VERIFIED | Only `MelSpectrogram` import; `waveform_to_mel` present; no `griffin_lim`, `inverse_mel`, or `mel_to_waveform` |
| `src/distill/vocoder/mel_adapter.py` | MelAdapter with direct filterbank transfer | VERIFIED | Tikhonov-regularized pseudo-inverse transfer matrix, `torch.matmul`, `F.interpolate` for time resampling, no waveform intermediate |
| `src/distill/ui/tabs/train_tab.py` | Vocoder training section with controls and chart | VERIFIED | `gr.Accordion("Vocoder Training (Per-Model HiFi-GAN)")`, all parameter controls, cancel/resume/replace-confirm handlers, Timer-polled dashboard |
| `src/distill/ui/components/loss_chart.py` | Dual-loss chart builder for GAN training | VERIFIED | `build_vocoder_loss_chart()` with dual y-axes (generator blue left, discriminator red right), mel loss dashed |
| `src/distill/ui/state.py` | AppState vocoder training fields | VERIFIED | `vocoder_trainer`, `vocoder_metrics_buffer`, `vocoder_cancel_event`, `vocoder_training_active` fields; `reset_vocoder_metrics_buffer()` function |
| `src/distill/cli/train_vocoder.py` | distill train-vocoder CLI command | VERIFIED | `train_vocoder_cmd` with all required options, Rich `Live(Table)`, SIGINT handler, `cancel_event.set()` |
| `src/distill/cli/__init__.py` | train-vocoder subcommand registered | VERIFIED | `app.add_typer(train_vocoder_app, name="train-vocoder", ...)` at line 207 |

---

## Key Link Verification

### Plan 01 Key Links

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `generator.py` | `config.py` | Generator reads `upsample_rates`, `kernel_sizes`, `resblock` config from `HiFiGANConfig` | WIRED | `HiFiGANConfig` imported in `generator.py`; constructor reads `config.upsample_rates`, `config.upsample_kernel_sizes`, `config.resblock_kernel_sizes`, `config.resblock_dilation_sizes`, `config.num_mels`, `config.upsample_initial_channel` |
| `discriminator.py` | `config.py` | Discriminator reads `mpd_periods` from `HiFiGANConfig` | WIRED | `MultiPeriodDiscriminator.__init__` uses `config.mpd_periods` to build `PeriodDiscriminator` list |

### Plan 02 Key Links

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `bigvgan_vocoder.py` | `mel_adapter.py` | BigVGANVocoder uses `MelAdapter.convert()` which no longer calls Griffin-Lim | WIRED | `MelAdapter` imported in `bigvgan_vocoder.py`; `MelAdapter.convert()` uses direct filterbank transfer matrix only |

### Plan 03 Key Links

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `hifigan/trainer.py` | `models/persistence.py` | Trainer saves `vocoder_state` into `.distillgan` via `torch.save` | WIRED | `_save_vocoder_state()` uses `torch.load` + injects `saved["vocoder_state"]` + `torch.save` |
| `vocoder/__init__.py` | `hifigan/vocoder.py` | `resolve_vocoder` creates `HiFiGANVocoder` when auto-select finds `vocoder_state` | WIRED | `from distill.vocoder.hifigan.vocoder import HiFiGANVocoder` at `vocoder/__init__.py:81`; used at lines 142, 163 in `resolve_vocoder` |
| `hifigan/vocoder.py` | `vocoder/base.py` | `HiFiGANVocoder` implements `VocoderBase` abstract methods | WIRED | `class HiFiGANVocoder(VocoderBase)` imports `VocoderBase` directly at line 25; implements `mel_to_waveform`, `sample_rate`, `to()` |
| `vocoder/__init__.py` | `cli/generate.py` | `resolve_vocoder` returns info dict with `name="per_model_hifigan"` consumed by generate CLI | WIRED | `cli/generate.py:397,604` branch on `vocoder_info["name"] == "bigvgan_universal"` — handles `per_model_hifigan` as the else case |

### Plan 04 Key Links

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `ui/tabs/train_tab.py` | `hifigan/trainer.py` | `VocoderTrainer.train()` called from background thread with metrics callback | WIRED | `from distill.vocoder.hifigan.trainer import VocoderTrainer` at line 861; `threading.Thread(target=trainer.train, ...)` at line 898 |
| `ui/tabs/train_tab.py` | `ui/state.py` | Vocoder training metrics stored in `app_state` for Timer-polled UI updates | WIRED | `app_state.vocoder_metrics_buffer`, `app_state.vocoder_trainer`, `app_state.vocoder_cancel_event` all used in handlers; `vocoder_timer.tick()` reads buffer |

### Plan 05 Key Links

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `cli/train_vocoder.py` | `hifigan/trainer.py` | CLI calls `VocoderTrainer.train()` with Rich callback | WIRED | `from distill.vocoder.hifigan import VocoderTrainer` at line 77; `trainer.train(model_path=model_path, audio_dir=audio_dir, ...)` in `with live_context` block |
| `cli/__init__.py` | `cli/train_vocoder.py` | `app.add_typer` registration for `train-vocoder` subcommand | WIRED | `from distill.cli.train_vocoder import app as train_vocoder_app` then `app.add_typer(train_vocoder_app, name="train-vocoder", ...)` at lines 205-207 |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| TRAIN-01 | 16-01 | User can train HiFi-GAN V2 vocoder on model's training audio | SATISFIED | `VocoderTrainer.train()` accepts `model_path` + `audio_dir`, loads model's `AudioSpectrogram`, trains HiFi-GAN |
| TRAIN-02 | 16-01 | Training uses adversarial loss (MPD+MSD) + mel reconstruction loss + feature matching loss | SATISFIED | `loss_gen = loss_gen_mpd + loss_gen_msd + loss_fm_mpd + loss_fm_msd + loss_mel * 45` at `trainer.py:514-520` |
| TRAIN-03 | 16-03 | Data augmentation applied during training | SATISFIED | `_augment_disc_input()` applies random gain +/-3dB and noise injection SNR 30-50dB to discriminator inputs |
| TRAIN-04 | 16-03 | Trained vocoder weights bundled into .distillgan model file | SATISFIED | `_save_vocoder_state()` injects `vocoder_state` (containing `generator_state_dict`) into `.distillgan` via `torch.load/torch.save` |
| TRAIN-05 | 16-03 | Training supports cancel with checkpoint save and resume | SATISFIED | `cancel_event.is_set()` triggers `_build_vocoder_state()` with full checkpoint + `_save_vocoder_state()`; resume restores all 7 state dicts + `start_epoch` |
| VOC-05 | 16-03 | Vocoder auto-selects best available: per-model HiFi-GAN > BigVGAN universal | SATISFIED | `resolve_vocoder("auto")` prefers `HiFiGANVocoder` when `loaded_model.vocoder_state is not None` |
| UI-03 | 16-04 | Train tab has "Train Vocoder" option for models with completed VAE training | SATISFIED | `gr.Accordion("Vocoder Training (Per-Model HiFi-GAN)")` with enable/disable based on `meta.training_epochs > 0` |
| UI-04 | 16-04 | Vocoder training shows loss curve and progress in UI | SATISFIED | `build_vocoder_loss_chart()` renders dual-axis gen+disc chart; Timer polls `vocoder_metrics_buffer` every 2s for live updates |
| CLI-02 | 16-05 | `train-vocoder` CLI command trains per-model HiFi-GAN vocoder | SATISFIED | `distill train-vocoder MODEL_PATH AUDIO_DIR --epochs 200 --lr 0.0002 --batch-size 8 --checkpoint-interval 50` |
| GEN-04 | 16-02 | Griffin-Lim reconstruction code fully removed | SATISFIED | Zero `GriffinLim`, `InverseMelScale`, or `griffin_lim` references in `src/distill`; `AudioSpectrogram` is forward-only |

All 10 requirements for Phase 16 are SATISFIED.

---

## Anti-Patterns Found

No blockers or warnings found. Scanned all 14 modified/created files.

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `train_tab.py:215,682` | `placeholder=` strings in `gr.Textbox` | INFO | Normal Gradio usage — not stub code, just UI hint text |

The two `placeholder=` matches are standard Gradio textbox hint text (not code stubs) and are fully appropriate in context.

---

## Human Verification Required

The following items cannot be verified programmatically:

### 1. GAN Training Convergence Quality

**Test:** Train a vocoder on a small audio dataset (5-20 files) for 100-200 epochs. Compare generated audio from `HiFiGANVocoder` vs `BigVGANVocoder` on the same mel inputs.
**Expected:** Per-model HiFi-GAN should produce audio that matches the specific timbre/texture of the training material more closely than the universal BigVGAN.
**Why human:** Audio perceptual quality cannot be evaluated programmatically without a reference evaluation dataset and metric setup.

### 2. Dual-Loss Chart Readability

**Test:** Launch Gradio UI, load a trained model, start vocoder training for several epochs, observe the loss chart.
**Expected:** Generator loss (blue, left axis) and discriminator loss (red, right axis) display correctly on dual y-axes; chart updates live every 2 seconds; mel loss shown as dashed blue line.
**Why human:** Chart visual rendering and live-update behavior require UI observation.

### 3. Cancel + Resume Round-Trip

**Test:** Start `distill train-vocoder` or UI training, press Ctrl+C / Cancel after ~10 epochs, confirm checkpoint message, re-run with `--resume`, verify training resumes from correct epoch.
**Expected:** "Checkpoint saved at epoch N. Resume anytime." message appears; second run starts at epoch N+1 with same loss trajectory.
**Why human:** Requires actually running training against a real .distillgan file with audio data.

### 4. Low-Epoch Warning Display

**Test:** Manually inject a `vocoder_state` with `training_metadata.epochs = 5` into a `.distillgan` file, then run `distill generate` with `--vocoder auto`.
**Expected:** Warning message "Trained for 5 epochs -- quality may be limited" is displayed.
**Why human:** Requires model file manipulation and real generation invocation.

---

## Gaps Summary

No gaps found. All 15 must-have truths verified, all 14 artifacts confirmed substantive and wired, all 10 Phase 16 requirements satisfied, zero Griffin-Lim code remaining, and all key links from Plans 01-05 confirmed wired in the codebase.

The phase goal is achieved:
1. Users can train per-model HiFi-GAN V2 vocoders via UI (Train tab Accordion) and CLI (`distill train-vocoder`) with full parameter control, live loss charts, audio previews, and cancel/resume.
2. The system auto-selects per-model HiFi-GAN over BigVGAN universal when `vocoder_state` exists in the model file.
3. Griffin-Lim is fully removed — zero references remain in `src/distill/`. `AudioSpectrogram` is forward-only and `MelAdapter` uses a direct filterbank transfer matrix.

---

_Verified: 2026-02-28T23:45:00Z_
_Verifier: Claude (gsd-verifier)_
