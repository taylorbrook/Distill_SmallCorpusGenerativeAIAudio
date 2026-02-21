# Pitfalls Research: Neural Vocoder Integration

**Domain:** Adding HiFi-GAN / BigVGAN-v2 neural vocoders to existing mel-spectrogram VAE generative audio system
**Researched:** 2026-02-21
**Confidence:** HIGH (verified against BigVGAN source code, project codebase, official documentation, and community reports)

This document catalogs the specific pitfalls of **adding neural vocoders to an existing mel-spectrogram-based generation pipeline**. It focuses on what breaks when integrating BigVGAN-v2 and HiFi-GAN V2 into the Distill project's existing 48 kHz / 128-mel / log1p / torchaudio HTK pipeline.

---

## Critical Pitfalls

These cause garbage audio, silent failures, or rewrites if not addressed.

### Pitfall 1: Mel Filterbank Scale Mismatch (Slaney vs HTK)

**What goes wrong:**
BigVGAN produces white noise, metallic garbage, or unrecognizable audio instead of coherent waveforms. The audio is not subtly degraded -- it is completely unintelligible.

**Why it happens:**
BigVGAN was trained with `librosa.filters.mel()` which uses **Slaney mel scale with area normalization** by default. The project's `AudioSpectrogram` uses `torchaudio.transforms.MelSpectrogram` which uses **HTK mel scale with no normalization** by default. These produce numerically different filterbank matrices. The mel frequency mapping formulas differ (Slaney is linear below 1 kHz and logarithmic above; HTK is purely logarithmic), and Slaney normalization divides each triangular filter by its bandwidth. Feeding an HTK-computed mel spectrogram to a model trained on Slaney mels is feeding the wrong data distribution entirely.

**Specific values in this project:**
- Project: `torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128)` -- uses HTK scale, no normalization, `center=True`
- BigVGAN: `librosa.filters.mel(sr=44100, n_fft=2048, n_mels=128, fmin=0, fmax=None)` -- uses Slaney scale, Slaney area normalization, `center=False`

**How to avoid:**
1. Create a dedicated `BigVGANMelSpectrogram` class that uses `librosa.filters.mel()` for the filterbank matrix and `torch.stft()` with BigVGAN's exact parameters.
2. Never attempt to "adapt" or "convert" between filterbank types at runtime -- the mapping is not invertible from a processed mel spectrogram.
3. For BigVGAN inference: compute a fresh BigVGAN-compatible mel from the waveform (if available) or use the MelAdapter to convert from log1p to log domain (if only the VAE mel is available).
4. For new model training (v1.1+): train the VAE on BigVGAN-compatible mels from the start, so the decoded mel can be fed directly to BigVGAN.
5. For backward compatibility: existing v1.0 models trained on HTK mels will use the MelAdapter path, which converts normalization but cannot fix the filterbank difference. Accept slightly degraded quality for legacy models.

**Warning signs:**
- BigVGAN output sounds like static, white noise, or metallic screeching
- Output waveform has extreme amplitude values (clipping at +/- 1.0 constantly)
- Mel spectrogram visualization looks visually similar but produces completely different audio
- A/B comparison with Griffin-Lim shows Griffin-Lim is better (should never happen with correct mels)

**Phase to address:**
Phase 1 (Vocoder Interface and BigVGAN Integration) -- this must be solved before any vocoder code is written. Build the `BigVGANMelSpectrogram` class first and validate it produces identical output to BigVGAN's own `mel_spectrogram()` function.

**Severity:** CRITICAL -- this is the single most likely cause of a "it doesn't work at all" failure.

---

### Pitfall 2: Log Compression Mismatch (log1p vs log-clamp)

**What goes wrong:**
BigVGAN output sounds muffled, distorted, or has wrong dynamics. Quiet passages are too loud, loud passages are clipped. The audio is recognizable but wrong.

**Why it happens:**
The project normalizes mel spectrograms with `torch.log1p(mel)` which computes `log(1 + x)`. BigVGAN uses `torch.log(torch.clamp(x, min=1e-5))` which computes `log(max(x, 0.00001))`. These produce different dynamic ranges:

```
For mel value 0.0:   log1p(0) = 0.0,       log(clamp(0)) = log(1e-5) = -11.51
For mel value 1.0:   log1p(1) = 0.693,     log(clamp(1)) = 0.0
For mel value 10.0:  log1p(10) = 2.398,    log(clamp(10)) = 2.303
```

The zero-handling is the critical difference: `log1p` maps zero to zero (clean silence), while BigVGAN's `log(clamp)` maps zero to -11.51 (a large negative value). If the VAE's log1p-normalized mel is fed to BigVGAN without conversion, the model sees a completely different value distribution than it was trained on.

**Specific code in this project:**
- `spectrogram.py` line 110: `mel_log = torch.log1p(mel)`
- `spectrogram.py` line 134: `mel = torch.expm1(mel_log.squeeze(1).clamp(min=0))`
- BigVGAN `meldataset.py`: `torch.log(torch.clamp(x, min=1e-5) * C)` where C=1

**How to avoid:**
1. Implement `MelAdapter.vae_to_vocoder()`: undo log1p (expm1), then apply BigVGAN's log(clamp).
2. The conversion is mathematically exact: `mel_linear = torch.expm1(mel_log1p.clamp(min=0))`, then `mel_vocoder = torch.log(torch.clamp(mel_linear, min=1e-5))`.
3. The adapter must be stateless and run at the vocoder boundary -- do not change the VAE's internal normalization.
4. Write a unit test that round-trips: `vae_mel -> adapter -> vocoder_mel -> inverse_adapter -> vae_mel` and verifies numerical equivalence.
5. Clamp the expm1 output at min=0 to prevent negative linear mel values from floating-point imprecision.

**Warning signs:**
- Audio output has "underwater" or muffled quality
- Very quiet or silent passages in generated audio are distorted
- Dynamic range sounds compressed or expanded unnaturally
- Comparing BigVGAN output with and without adapter shows dramatic difference

**Phase to address:**
Phase 1 (Vocoder Interface and BigVGAN Integration) -- the MelAdapter is a prerequisite for any BigVGAN inference.

**Severity:** CRITICAL -- gets the mel values wrong, but less catastrophically than Pitfall 1.

---

### Pitfall 3: Center Padding Mismatch (center=True vs center=False)

**What goes wrong:**
The mel spectrogram has a different number of time frames than BigVGAN expects. The vocoder produces audio with timing artifacts, clicks at the boundaries, or slightly wrong duration.

**Why it happens:**
TorchAudio's `MelSpectrogram` defaults to `center=True`, which pads the input signal by `n_fft // 2` on each side before computing the STFT. This ensures the first and last frames are centered on actual signal content. BigVGAN explicitly uses `center=False` and instead applies its own padding: `torch.nn.functional.pad(y, ((n_fft - hop_size) // 2, (n_fft - hop_size) // 2), mode="reflect")`.

The padding difference means:
- `center=True` (project): input of N samples produces `N // hop_length + 1` frames
- `center=False` with BigVGAN padding: input of N samples produces a different frame count

For the project's 1-second chunks at 48 kHz (48000 samples): `center=True` produces 94 frames, while `center=False` with BigVGAN's padding produces a different count. This mismatch means the mel-to-waveform temporal relationship is wrong.

**How to avoid:**
1. The `BigVGANMelSpectrogram` class must use `center=False` with BigVGAN's explicit reflect padding, not TorchAudio's `center=True`.
2. When computing mels for BigVGAN, use `torch.stft()` directly with `center=False` and manually pad the input first.
3. Update `get_mel_shape()` to account for the different frame count when using BigVGAN-compatible mels.
4. The VAE's mel shape calculation in `synthesize_continuous_mel()` depends on `get_mel_shape(chunk_samples)` -- this must be updated for the new padding behavior.

**Warning signs:**
- Mel spectrogram has 1-2 more/fewer frames than expected
- Audio output is slightly time-stretched or compressed
- Clicks or artifacts at chunk boundaries in continuous generation
- `synthesize_continuous_mel` overlap-add produces visible seams in the mel

**Phase to address:**
Phase 1 (BigVGAN Integration) -- part of the `BigVGANMelSpectrogram` implementation.

**Severity:** HIGH -- produces subtle but audible artifacts that are hard to diagnose.

---

### Pitfall 4: Sample Rate Mismatch Without Proper Resampling (48 kHz vs 44.1 kHz)

**What goes wrong:**
Audio is pitch-shifted (sounds higher or lower than expected), time-stretched (plays slower or faster), or has aliasing artifacts in the high frequencies. Alternatively, the audio sounds correct but is at the wrong sample rate for the export pipeline.

**Why it happens:**
The project operates at 48 kHz internally. BigVGAN's best model (`bigvgan_v2_44khz_128band_512x`) operates at 44.1 kHz. The mel-to-waveform upsampling ratio is baked into BigVGAN: each mel frame produces exactly 512 audio samples at 44.1 kHz. If 48 kHz-parameterized audio is fed to BigVGAN without resampling, or if BigVGAN's 44.1 kHz output is treated as 48 kHz, the audio will be wrong.

Two specific failure modes:
1. **Feeding 48 kHz mel to 44.1 kHz model:** The mel frame rate is wrong (48000/512 = 93.75 fps vs 44100/512 = 86.13 fps). Each mel frame represents a different duration of audio. BigVGAN will generate 512 samples per frame at 44.1 kHz timing, but the mel content represents 48 kHz timing. Result: ~8.8% time compression.
2. **Treating 44.1 kHz output as 48 kHz:** The audio plays ~8.8% too fast and is pitch-shifted up by ~1.4 semitones.

**How to avoid:**
1. **Resample input audio to 44.1 kHz before computing BigVGAN-compatible mels.** The project already has `_get_resampler()` in `generation.py` for this.
2. **Let BigVGAN produce 44.1 kHz output, then resample to 48 kHz** before the spatial/export pipeline. Use `torchaudio.transforms.Resample(44100, 48000)` which is already available.
3. Do NOT attempt to "adjust" mel frame rates or "scale" mel spectrograms to compensate for sample rate differences. This produces artifacts.
4. The resampling quality of `torchaudio.transforms.Resample` is excellent for 44.1 kHz to 48 kHz (both are above human hearing Nyquist of ~20 kHz). No audible degradation.
5. Document the internal sample rate change clearly: VAE training data is resampled to 44.1 kHz for mel computation, BigVGAN outputs at 44.1 kHz, output is resampled to 48 kHz for export.

**Warning signs:**
- Generated audio is pitched up or down compared to training data
- Audio duration doesn't match expected duration (off by ~8.8%)
- High-frequency content sounds different between BigVGAN and Griffin-Lim output
- The `GenerationPipeline` reports wrong duration in `GenerationResult`

**Phase to address:**
Phase 1 (BigVGAN Integration) -- the resampling must be wired into the generation pipeline.

**Severity:** HIGH -- produces immediately audible pitch/timing errors.

---

### Pitfall 5: Forgetting to Call remove_weight_norm() Before Inference

**What goes wrong:**
BigVGAN inference is 2-3x slower than expected, uses more memory, and may produce slightly different (lower quality) output compared to the expected behavior.

**Why it happens:**
BigVGAN uses weight normalization during training to stabilize optimization. Weight norm decomposes each weight tensor into magnitude and direction components, requiring extra computation on every forward pass. The `remove_weight_norm()` method fuses these back into a single tensor. If not called, every inference pass recomputes the decomposition unnecessarily.

BigVGAN's `_from_pretrained()` method handles this automatically when loading via `from_pretrained()`, but if the model is loaded manually (e.g., from vendored code), the developer must call `model.remove_weight_norm()` explicitly.

**How to avoid:**
1. Always call `model.remove_weight_norm()` immediately after loading BigVGAN weights and before calling `model.eval()`.
2. Add a defensive check in the `BigVGANVocoder` wrapper: if loading succeeds but weight norm is still present, call `remove_weight_norm()`.
3. The BigVGAN source code catches `ValueError` when weight norm was already removed -- this is safe to call multiple times.
4. Same applies to per-model HiFi-GAN V2: call `remove_weight_norm()` on the generator before inference.

**Warning signs:**
- BigVGAN inference takes 2-3x longer than expected benchmarks
- Memory usage during inference is higher than expected
- Profiling shows time spent in `torch.nn.utils.weight_norm` computations

**Phase to address:**
Phase 1 (BigVGAN Integration) -- part of the `BigVGANVocoder.__init__()` or `_ensure_loaded()`.

**Severity:** MODERATE -- doesn't break functionality, but degrades performance significantly.

---

### Pitfall 6: GAN Discriminator Overfitting on Small Datasets (5-500 files)

**What goes wrong:**
Per-model HiFi-GAN V2 training collapses: the discriminator reaches near-perfect accuracy within a few thousand steps, generator gradients vanish, and the generator stops improving. Alternatively, the generator produces one or two "safe" outputs regardless of input (mode collapse).

**Why it happens:**
GAN training fundamentally requires enough data for the discriminator to learn generalizable features rather than memorizing specific examples. With 5-500 audio files:
- The discriminator sees the same waveforms repeatedly and memorizes them rather than learning generalizable "real vs fake" features.
- With memorized training data, the discriminator trivially rejects everything the generator produces, providing no useful gradient signal.
- Standard HiFi-GAN training assumes thousands of audio files (e.g., LJSpeech has 13,100 clips). At 5-50 files, the discriminator-to-data ratio is severely imbalanced.

Research confirms this: "With smaller datasets (under 1,000 samples), you're more likely to experience overfitting, mode collapse, and training instability." Specific GAN vocoder research (Augmentation-Conditional Discriminator, ICASSP 2024) identifies discriminator overfitting as the primary failure mode for limited-data vocoder training.

**How to avoid:**
1. **Data augmentation is mandatory**, not optional. Apply differentiable augmentations (pitch shift, time stretch, noise injection, SpecAugment) to both real and generated audio fed to the discriminator. Augmentation must be applied to both real and fake samples to prevent the discriminator from using augmentation artifacts as a shortcut.
2. **Reduce discriminator capacity** for small datasets. Use fewer layers or smaller channel widths in the MPD and MSD discriminators. The discriminator should be weaker than the standard HiFi-GAN configuration.
3. **Increase mel spectrogram loss weight** relative to adversarial loss (e.g., mel weight 45.0 + multi-resolution STFT weight 15.0 vs adversarial weight 1.0). This makes the generator less dependent on discriminator feedback for learning.
4. **Discriminator learning rate decay.** Reduce discriminator learning rate faster than generator learning rate to prevent the discriminator from running away.
5. **Early stopping based on mel loss**, not adversarial loss. The mel spectrogram L1 loss is a reliable quality indicator; the adversarial loss oscillates and is unreliable for convergence detection.
6. **Consider fine-tuning from BigVGAN checkpoint** instead of training from scratch. Even though BigVGAN is 112M params (too large for per-model fine-tuning), its discriminator features could initialize a smaller HiFi-GAN discriminator via distillation.
7. **Minimum viable dataset**: document that per-model HiFi-GAN training needs at least 20-50 files for reasonable results. For datasets under 20 files, recommend BigVGAN universal only.

**Warning signs:**
- Discriminator loss drops to near-zero within first 5K steps
- Generator loss increases or plateaus while discriminator loss decreases
- Generated audio samples all sound identical regardless of input mel
- Training metrics show discriminator accuracy > 95% on real samples
- Audio quality stops improving after initial 10K-20K steps despite continued training

**Phase to address:**
Phase 5 (HiFi-GAN V2 Model + Training) -- this is the core risk of the per-model vocoder training feature.

**Severity:** HIGH -- can make per-model HiFi-GAN training completely useless for small datasets.

---

### Pitfall 7: Loading 112M-Parameter BigVGAN Alongside VAE Causes OOM

**What goes wrong:**
Out-of-memory crash when loading BigVGAN for inference, especially on devices with limited VRAM/unified memory (8 GB MacBooks, 4-6 GB CUDA GPUs). The application crashes or becomes extremely slow due to memory pressure.

**Why it happens:**
BigVGAN `bigvgan_v2_44khz_128band_512x` has 122M parameters (~489 MB in float32). The existing VAE model is comparatively small. Loading both simultaneously doubles the GPU memory requirement. On Apple Silicon with 8 GB unified memory, the system must share memory between the OS, the application, the VAE, and BigVGAN.

Additionally, BigVGAN's inference creates large intermediate tensors during the upsampling chain (512x upsampling from mel to waveform involves multiple transposed convolution layers with large channel dimensions).

**How to avoid:**
1. **Lazy loading:** Load BigVGAN only when first needed for inference, not at application startup. The `_ensure_loaded()` pattern already planned in the architecture is correct.
2. **Unload when not needed:** After generation, consider moving BigVGAN to CPU or deleting it entirely if memory is tight. Reload from HuggingFace cache (instant, no network) on next use.
3. **Device-aware loading:** On devices with < 8 GB VRAM, load BigVGAN to CPU and run inference on CPU. Slower (~5-10x realtime) but functional.
4. **Float16 inference:** BigVGAN works in float16 on CUDA, halving memory to ~245 MB. On MPS, float16 may have issues with some operations -- test before enabling.
5. **Chunk long mels:** For very long generations (>10 seconds), process the mel in overlapping chunks through BigVGAN rather than the entire mel at once. This caps peak memory usage.
6. **Memory monitoring:** Before loading BigVGAN, check available memory. Warn the user if memory is low and offer CPU fallback.
7. **Profile actual usage:** Measure peak memory during BigVGAN inference with typical mel sizes (1-60 seconds of audio) on target hardware.

**Warning signs:**
- OOM errors when switching from Griffin-Lim to BigVGAN
- Application becomes unresponsive after loading BigVGAN
- MPS "buffer size exceeded" errors
- Gradio UI freezes during first BigVGAN generation

**Phase to address:**
Phase 1 (BigVGAN Integration) -- the lazy loading and memory management must be built into the BigVGANVocoder class from the start.

**Severity:** HIGH -- OOM crashes are the worst user experience. Users on 8 GB Macs are a key demographic.

---

### Pitfall 8: MPS Incompatibility with torch.stft and Complex Numbers

**What goes wrong:**
BigVGAN mel computation crashes on MPS (Apple Silicon) with errors like `"The operator 'aten::_fft_r2c' is not currently supported on the MPS backend"`. The mel spectrogram cannot be computed on GPU on Macs.

**Why it happens:**
BigVGAN's mel spectrogram function uses `torch.stft()` with `return_complex=True`. The MPS backend has historically lacked support for FFT operations and complex number datatypes. While PyTorch 2.10 has expanded MPS operator coverage significantly, FFT/STFT support on MPS may still be incomplete or have edge cases.

The existing project already handles a similar MPS limitation: `InverseMelScale` is forced to CPU in `spectrogram.py` (line 136: `linear_spec = self.inverse_mel(mel.cpu())`). BigVGAN's mel computation may need the same treatment.

Note: BigVGAN's **inference** (mel-to-waveform) uses standard convolutions and activations that work on MPS. The issue is specifically with the **mel computation** step (STFT), not the vocoder inference itself.

**How to avoid:**
1. **Compute BigVGAN-compatible mels on CPU** when running on MPS. The mel computation is fast (milliseconds) and not the bottleneck. Only move the computed mel tensor to MPS for BigVGAN inference.
2. **Test `torch.stft()` on MPS** with PyTorch 2.10 during early development. If it works, great. If not, fall back to CPU computation.
3. **Use `PYTORCH_ENABLE_MPS_FALLBACK=1`** as a runtime option, but do not rely on it -- explicit CPU fallback for known-unsupported ops is cleaner.
4. **Separate mel computation from vocoder inference** in the architecture. The `BigVGANMelSpectrogram` class computes mels (may need CPU on MPS), and the `BigVGANVocoder` class runs inference (works on MPS).
5. **Document the device flow clearly:** audio on CPU -> mel on CPU -> mel.to(device) -> BigVGAN inference on device -> waveform output.

**Warning signs:**
- `NotImplementedError` or `RuntimeError` mentioning `aten::_fft_r2c` or `aten::_fft_c2c`
- BigVGAN works on CUDA but crashes on MPS
- Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` fixes the crash (confirms MPS op gap)

**Phase to address:**
Phase 1 (BigVGAN Integration) -- test MPS compatibility early, implement CPU fallback for mel computation.

**Severity:** HIGH for Apple Silicon users -- completely blocks vocoder usage if not handled.

---

### Pitfall 9: Breaking Existing Models by Changing Mel Computation

**What goes wrong:**
After switching to BigVGAN-compatible mel parameters, all existing v1.0 `.distill` models produce worse or different audio. The VAE was trained on HTK mels but is now being asked to work with Slaney mels. Latent space meanings shift, slider mappings break, reconstruction quality degrades.

**Why it happens:**
The VAE's encoder learned to encode HTK-normalized mel spectrograms, and the decoder learned to decode them. If the mel computation changes, the encoder/decoder are operating on a different data distribution. The latent space analysis (PCA-based slider mappings) was computed on HTK mel representations and becomes invalid with Slaney mels.

This is the classic "changing the data pipeline invalidates all trained models" pitfall.

**How to avoid:**
1. **Never change mel computation for existing models.** Each `.distill` file stores its `spectrogram_config`. When loading a v1.0 model, use the original HTK mel computation for the VAE.
2. **The vocoder adapter path handles the mismatch.** For v1.0 models: VAE decodes to HTK log1p mel -> MelAdapter converts to vocoder normalization -> BigVGAN produces audio. Accept that this path has slightly lower quality than native BigVGAN mels.
3. **New v1.1 models train on BigVGAN-compatible mels** from the start. Their `spectrogram_config` records the new parameters (Slaney scale, log-clamp normalization, center=False).
4. **Version the mel computation type** in the saved model format. Add a `mel_type: "htk_log1p" | "slaney_logclamp"` field to the model's spectrogram config.
5. **Do NOT offer a "convert existing model" feature.** The VAE weights are coupled to the mel distribution -- conversion requires retraining, not just parameter changes.
6. **Test backward compatibility extensively.** Load every existing v1.0 model with the new code and verify audio quality is unchanged (or acceptably close) via A/B listening tests.

**Warning signs:**
- Existing models produce noticeably different audio after code update
- Slider positions produce different sounds than before
- Users report "my model sounds worse after updating"
- PCA analysis results change for the same model

**Phase to address:**
Phase 2 (Model Persistence Update) -- version the model format and preserve backward compatibility.

**Severity:** CRITICAL -- breaking existing models destroys user trust.

---

### Pitfall 10: HiFi-GAN Upsampling Rate Mismatch with hop_length

**What goes wrong:**
HiFi-GAN V2 produces audio at the wrong sample rate, or crashes with a tensor shape mismatch error. The output waveform has the wrong number of samples per mel frame.

**Why it happens:**
The product of HiFi-GAN's `upsample_rates` must exactly equal the `hop_length`. The standard HiFi-GAN V2 config uses `upsample_rates = [8, 8, 2, 2]` which multiplies to 256 (matching hop_length=256 at 22 kHz). But this project uses `hop_length=512` at 44.1 kHz. The upsampling chain must multiply to 512, not 256.

If `product(upsample_rates) != hop_length`, the generator produces the wrong number of audio samples per mel frame, causing:
- Too few samples: output is time-compressed, pitch-shifted up
- Too many samples: output is time-stretched, pitch-shifted down
- Shape mismatch: crash during training or inference

**How to avoid:**
1. **Set `upsample_rates` to multiply to exactly 512:** options include `[8, 8, 2, 2, 2]` (5 layers) or `[8, 8, 4, 2]` (4 layers) or `[8, 4, 4, 2, 2]` (5 layers).
2. **Update `upsample_kernel_sizes`** to match: each kernel should be 2x its corresponding upsample rate (e.g., rate 8 -> kernel 16, rate 4 -> kernel 8, rate 2 -> kernel 4).
3. **Validate at initialization:** add an assertion in `HiFiGANConfig.__post_init__()` that `product(upsample_rates) == hop_length`.
4. **Also validate `segment_size`:** the training segment size must be divisible by `product(upsample_rates)`. With 512x upsampling, `segment_size` must be a multiple of 512.
5. **The architecture doc recommends `[8, 8, 2, 2, 2]`** giving 512x with 5 upsampling layers. This adds one layer vs standard V2 but is the cleanest factorization.

**Warning signs:**
- `RuntimeError: shape mismatch` during HiFi-GAN forward pass
- Generated audio plays at wrong speed or pitch
- Output sample count != input mel frames * hop_length
- Training loss doesn't decrease (shape mismatch causes meaningless gradients)

**Phase to address:**
Phase 5 (HiFi-GAN V2 Model + Training) -- validate configuration before starting any training.

**Severity:** HIGH -- a single wrong number makes the entire model non-functional.

---

## Moderate Pitfalls

### Pitfall 11: Joint VAE + HiFi-GAN Training Instability

**What goes wrong:**
Attempting to train the VAE and HiFi-GAN vocoder simultaneously causes both to perform worse than training them separately. The HiFi-GAN chases a moving target as the VAE's mel output changes during training.

**Why it happens:**
The VAE decoder produces mel spectrograms that change as the VAE trains. If HiFi-GAN is trained to convert these evolving mels to audio, it learns a mapping that becomes invalid as the VAE improves. The HiFi-GAN discriminator learns to distinguish based on artifacts in early-training VAE mels, which are different from late-training mels. This creates a non-stationary optimization problem that rarely converges.

**How to avoid:**
1. **Train sequentially, never jointly.** Train the VAE first, freeze it, then train HiFi-GAN on the frozen VAE's mel outputs.
2. **The architecture already plans for this** (separate `HiFiGANTrainer` class, runs after VAE training).
3. If users want to retrain the VAE later, they must also retrain the HiFi-GAN vocoder.
4. Document this dependency in the UI: "Retraining the VAE invalidates the per-model vocoder. You will need to retrain the vocoder after retraining the model."

**Warning signs:**
- Both VAE and HiFi-GAN losses oscillate without converging
- Audio quality degrades during training instead of improving
- HiFi-GAN produces good audio for early training checkpoints but bad audio for later ones

**Phase to address:**
Phase 5 (HiFi-GAN Training) -- enforce sequential training in the training pipeline.

**Severity:** MODERATE -- the design already addresses this, but a developer might be tempted to optimize by training jointly.

---

### Pitfall 12: BigVGAN Model File Bloat in .distill Files

**What goes wrong:**
Each `.distill` model file grows from ~5 MB (VAE only) to ~500 MB (VAE + BigVGAN). A user with 10 models uses 5 GB of disk space. Save/load becomes slow. Sharing models becomes impractical.

**Why it happens:**
A developer bundles BigVGAN weights (122M params, ~489 MB) into every `.distill` file alongside the VAE weights. This seems convenient ("everything in one file") but is wasteful since all models use the same BigVGAN weights.

**How to avoid:**
1. **Shared BigVGAN cache.** BigVGAN weights live in the HuggingFace cache (`~/.cache/huggingface/hub`) or a project-level `data/vocoders/` directory. Downloaded once, shared across all models.
2. **Only per-model HiFi-GAN goes in .distill files.** HiFi-GAN V2 generator is ~4 MB (0.92M params). This is negligible.
3. **The .distill format stores a `vocoder_type` field** ("bigvgan" or "hifigan") but NOT BigVGAN weights. If `vocoder_type == "bigvgan"`, the weights are loaded from the shared cache.
4. **Discriminator and optimizer state are never saved in .distill files.** They are checkpoint artifacts only, saved during training and discarded after. (~50-100 MB saved per model.)

**Warning signs:**
- `.distill` files are > 100 MB
- Saving a model takes more than a few seconds
- Users complain about disk space
- Model sharing over network is slow

**Phase to address:**
Phase 2 (Model Persistence Update) -- design the storage strategy correctly from the start.

**Severity:** MODERATE -- doesn't break functionality but creates poor UX and wastes resources.

---

### Pitfall 13: Not Testing BigVGAN with VAE-Generated Mels (Only Ground Truth)

**What goes wrong:**
BigVGAN sounds great when fed ground-truth mel spectrograms computed from real audio, but sounds degraded when fed mel spectrograms reconstructed by the VAE decoder. The developer concludes BigVGAN "works" but ships a product that sounds worse than expected.

**Why it happens:**
VAE-reconstructed mel spectrograms are blurry, smoothed, and may have artifacts compared to ground-truth mels. The VAE's mel output is a lossy reconstruction -- it has been compressed through the latent bottleneck. BigVGAN was trained on clean mel spectrograms, not VAE-reconstructed ones. The distribution shift between "clean mel" and "VAE-decoded mel" causes BigVGAN to produce lower-quality audio.

This is the "vocoder gap" problem well-known in TTS: the spectrogram model produces imperfect mels, and the vocoder handles these imperfections with varying success.

**How to avoid:**
1. **Always test BigVGAN with actual VAE output**, not just ground-truth mels. The quality from VAE-reconstructed mels is the real-world quality.
2. **Measure the quality gap:** compare BigVGAN(ground_truth_mel) vs BigVGAN(vae_reconstructed_mel). If the gap is large, the VAE reconstruction quality is the bottleneck, not the vocoder.
3. **Fine-tuning BigVGAN or training per-model HiFi-GAN on VAE-reconstructed mels** can close this gap. The HiFi-GAN learns to handle the VAE's specific artifacts.
4. **Include this comparison in the quality validation pipeline.** Every model should show: original audio, Griffin-Lim reconstruction, BigVGAN from ground-truth mel, BigVGAN from VAE mel.
5. **This is the primary motivation for per-model HiFi-GAN training:** it trains specifically on the VAE's output distribution, closing the vocoder gap.

**Warning signs:**
- BigVGAN demo with clean mels sounds amazing, but actual pipeline output sounds mediocre
- Per-model HiFi-GAN trained on VAE mels significantly outperforms BigVGAN universal
- Users complain that generated audio doesn't sound as good as the training audio

**Phase to address:**
Phase 3 (Generation Pipeline Integration) -- include ground-truth vs VAE mel comparison in quality validation.

**Severity:** MODERATE -- doesn't break anything but leads to disappointment and wasted debugging time.

---

### Pitfall 14: Resampling at the Wrong Point in the Pipeline

**What goes wrong:**
Audio is resampled multiple times (48 kHz -> 44.1 kHz for BigVGAN -> 48 kHz for spatial processing -> target sample rate for export), each resampling introducing slight quality loss and computational overhead.

**Why it happens:**
The project currently resamples at the end of the pipeline (step 9 in `GenerationPipeline.generate()`). With BigVGAN integration, there's a new resampling point (44.1 kHz BigVGAN output -> 48 kHz). If the export target is also not 48 kHz (e.g., 44.1 kHz), the audio gets resampled twice: 44.1 kHz -> 48 kHz -> 44.1 kHz. This is lossless in theory but introduces floating-point rounding.

**How to avoid:**
1. **Minimize resampling operations.** If BigVGAN outputs at 44.1 kHz and the export target is 44.1 kHz, skip the intermediate 48 kHz step.
2. **Single resampling point:** BigVGAN outputs at 44.1 kHz. If the target is 48 kHz (the default), resample once. If the target is 44.1 kHz, don't resample at all.
3. **Apply spatial processing at BigVGAN's output sample rate** (44.1 kHz) rather than resampling first. The spatial processing (stereo widening, binaural) doesn't depend on a specific sample rate.
4. **Update the `internal_sr` concept** in `GenerationPipeline.generate()`. Currently hardcoded to 48000 (line 422). With BigVGAN, internal processing happens at 44100, and the final resample targets the user's requested sample rate.

**Warning signs:**
- Audio at 44.1 kHz export sounds slightly different than 48 kHz export (should be perceptually identical)
- CPU profiler shows multiple resampling operations per generation
- Very subtle high-frequency loss in generated audio

**Phase to address:**
Phase 3 (Generation Pipeline Integration) -- restructure the sample rate flow.

**Severity:** LOW-MODERATE -- subtle quality impact, but architecturally messy if not handled.

---

### Pitfall 15: HiFi-GAN Training Without Proper Segment Size

**What goes wrong:**
HiFi-GAN training crashes with shape errors, or the model learns poorly because training segments are too short or too long for the dataset.

**Why it happens:**
HiFi-GAN training crops random segments from audio files for each training step. The `segment_size` parameter controls this crop length. If `segment_size` is larger than the shortest audio file in the dataset (converted to samples), training crashes. If it's too small, the model doesn't learn long-range structure. If it's not divisible by `product(upsample_rates)`, shape mismatches occur.

With small datasets (5-500 files), audio file lengths vary widely. A 2-second file at 44.1 kHz is 88,200 samples; `segment_size=16384` (0.37 seconds) is fine, but `segment_size=65536` (1.49 seconds) would fail on a 1-second file.

**How to avoid:**
1. **Validate segment_size against dataset:** scan all training audio files, find the shortest one, and ensure `segment_size < shortest_file_samples`.
2. **Default segment_size to a conservative value:** 16384 samples at 44.1 kHz (~0.37 seconds) is safe for most files.
3. **Assert divisibility:** `segment_size % product(upsample_rates) == 0`. With 512x upsampling, valid sizes are 512, 1024, ..., 16384, 16896, etc.
4. **Log a warning** if any audio file is shorter than 2x segment_size (the file will always be cropped from the same position).
5. **Pad short files** with zeros or reflection padding rather than crashing.

**Warning signs:**
- `IndexError` or `RuntimeError` during HiFi-GAN training data loading
- Training loss spikes or doesn't converge
- All training segments come from the same small set of files

**Phase to address:**
Phase 5 (HiFi-GAN Training) -- validate configuration against dataset before starting training.

**Severity:** MODERATE -- causes training failures that are easy to fix once diagnosed.

---

## Minor Pitfalls

### Pitfall 16: BigVGAN CUDA Kernel Compilation Failure

**What goes wrong:**
Enabling `use_cuda_kernel=True` fails with compilation errors. The user sees `nvcc` or `ninja` errors and concludes BigVGAN is broken.

**How to avoid:**
1. Default to `use_cuda_kernel=False`. The standard PyTorch implementation works everywhere.
2. Only enable CUDA kernel as an opt-in performance optimization for users who have `nvcc` and `ninja` installed.
3. Wrap the kernel loading in a try/except and fall back gracefully.
4. Document the speedup (1.5-3x on NVIDIA GPUs) so users know when it's worth the effort.

**Phase to address:** Phase 4 (UI and CLI Integration) -- make it a toggleable setting.

---

### Pitfall 17: Vendoring BigVGAN Files Without Pinning Version

**What goes wrong:**
BigVGAN repository updates break the vendored code. API changes, new dependencies, or renamed functions cause import errors.

**How to avoid:**
1. Pin to a specific commit hash when vendoring BigVGAN files.
2. Document the exact commit in a comment at the top of each vendored file.
3. Record the commit hash in a `VENDORED_VERSIONS` file or comment.
4. Only update vendored code deliberately, not automatically.

**Phase to address:** Phase 1 (BigVGAN Integration) -- pin when vendoring.

---

### Pitfall 18: HiFi-GAN Training Checkpoint Too Large for .distill

**What goes wrong:**
During HiFi-GAN training, full checkpoints (generator + discriminator + optimizers + scheduler state) are 150+ MB. If these are accidentally saved into the `.distill` model file instead of just the generator weights, file sizes balloon.

**How to avoid:**
1. Training checkpoints live in `data/vocoder_training/` as temporary files.
2. Only the generator state_dict (~4 MB for V2) is transferred to the `.distill` file after training completes.
3. Discriminator, optimizer, and scheduler state are never stored in `.distill` files.
4. Add a validation check: if `vocoder_state_dict` in a `.distill` file is > 50 MB, something is wrong.

**Phase to address:** Phase 5 (HiFi-GAN Training) -- enforce clean separation between training artifacts and model artifacts.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Feed HTK mels to BigVGAN with adapter only | Existing models work immediately, no retraining | Slightly lower quality than native BigVGAN mels; adapter cannot fix filterbank differences | Acceptable for backward compatibility with v1.0 models; new models should use native mels |
| Skip per-model HiFi-GAN, BigVGAN only | No GAN training complexity, simpler codebase | Users cannot get maximum fidelity for their specific audio | Acceptable for MVP; add per-model training later if user demand justifies it |
| CPU-only BigVGAN inference on MPS | Avoids all MPS compatibility issues | 5-10x slower inference; still adequate for non-real-time generation | Acceptable as fallback; test MPS inference and enable if it works |
| Hardcode BigVGAN model ID | Simpler code, no selection UI | Cannot use different BigVGAN variants (e.g., 256x for faster inference) | Acceptable for v1.1; add model selection later |
| No data augmentation for HiFi-GAN training | Simpler training pipeline | Discriminator overfitting on small datasets; poor quality for datasets < 50 files | Never acceptable -- augmentation is essential for small-data GAN training |
| Store BigVGAN weights in every .distill file | Self-contained models, easy sharing | 500 MB per model, massive disk waste | Never acceptable -- use shared cache |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| BigVGAN from_pretrained | Assuming it returns a ready-to-use model | Must call `remove_weight_norm()` and `model.eval()` after loading |
| librosa mel filterbank | Importing librosa at module level | Lazy-import librosa inside the mel computation function (project pattern); librosa import is slow (~1-2 seconds) |
| BigVGAN mel computation | Computing mel on GPU/MPS assuming `torch.stft` works everywhere | Compute mel on CPU, then transfer to GPU/MPS for vocoder inference |
| torchaudio Resample | Creating new Resample instance per call | Use the existing `_resampler_cache` in `generation.py`; Resample allocates internal buffers |
| HiFi-GAN discriminator | Saving discriminator in `.distill` file | Discriminator is training-only; save only generator state_dict in `.distill` |
| BigVGAN output shape | Assuming output is `[B, 1, T]` | BigVGAN outputs `[B, 1, T]` but some paths may squeeze/unsqueeze; validate shape at the boundary |
| HuggingFace Hub download | Not handling offline mode | Use `local_files_only=True` after first download; handle `ConnectionError` gracefully with user message |
| BigVGAN audio normalization | Skipping input normalization | BigVGAN training normalizes audio with `librosa.util.normalize(audio) * 0.95`; inference input mels should match this normalization |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading BigVGAN on every generation call | 3-5 second delay before each generation | Lazy-load once, cache in memory, reuse across generations | Immediately; first generation is slow, subsequent are fine |
| BigVGAN inference without torch.inference_mode | 2x memory usage from gradient tracking | Always wrap BigVGAN forward pass in `torch.inference_mode()` or `torch.no_grad()` | Long mel spectrograms (>10 seconds) |
| Computing BigVGAN mel every time instead of caching | Redundant STFT computation when mel is already available | If VAE mel can be adapted, use adapter; compute BigVGAN mel from waveform only when needed | Batch generation of multiple samples |
| HiFi-GAN training without mixed precision | 2x memory, 2x training time | Use `torch.cuda.amp.autocast()` for generator forward pass; discriminator runs in float32 | Datasets > 100 files where training takes hours |
| Full-length mel through BigVGAN for 60-second generation | Peak memory spike for 60-second mel (~5000 frames) | Process mel in overlapping chunks (e.g., 5-second windows with 0.5-second overlap) | Generations > 15 seconds on 8 GB devices |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No download progress for BigVGAN (~500 MB) | User thinks app froze during first use | Show download progress bar with estimated time; use `huggingface_hub` progress callbacks |
| Silent fallback from BigVGAN to Griffin-Lim | User expects better quality but gets Griffin-Lim without knowing | Show which vocoder is active in the UI; warn if BigVGAN failed to load |
| No A/B comparison between vocoders | User can't tell if BigVGAN is actually better for their audio | Provide a "Compare Vocoders" button that generates same mel with both and shows side-by-side |
| HiFi-GAN training time not estimated | User starts training expecting minutes, waits hours | Show estimated training time based on dataset size before starting; allow cancellation |
| "Train Vocoder" button with no explanation | User doesn't understand when/why to train a per-model vocoder | Tooltip/help text: "BigVGAN works well for most audio. Train a custom vocoder for maximum fidelity with your specific dataset." |
| Vocoder selection shows technical names | "bigvgan_v2_44khz_128band_512x" means nothing to users | Show user-friendly names: "Universal (recommended)", "Custom (trained for this model)", "Legacy (Griffin-Lim)" |

---

## "Looks Done But Isn't" Checklist

- [ ] **BigVGAN produces audio:** Often missing mel normalization adapter -- verify by comparing BigVGAN output from ground-truth mel (via librosa) vs from VAE mel (via adapter). They should sound comparable.
- [ ] **Vocoder works on MPS:** Often missing CPU fallback for torch.stft -- verify by running full generation pipeline on Apple Silicon Mac with no CUDA.
- [ ] **Existing models still work:** Often missing backward compatibility in persistence layer -- verify by loading every v1.0 `.distill` file with new code and comparing audio quality.
- [ ] **BigVGAN downloads correctly:** Often missing offline handling -- verify by disconnecting internet after first download and running generation.
- [ ] **HiFi-GAN V2 trains on small datasets:** Often missing augmentation and discriminator regularization -- verify by training on 10-file and 50-file datasets and comparing quality.
- [ ] **Sample rate is correct throughout:** Often missing resampling step -- verify output sample rate matches requested export sample rate, and pitch/speed matches original audio.
- [ ] **Memory is bounded:** Often missing cleanup after vocoder inference -- verify that generating 10 samples sequentially doesn't increase memory usage.
- [ ] **Vocoder selection persists:** Often missing serialization of vocoder choice -- verify that selected vocoder type survives app restart and model reload.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Mel filterbank mismatch (HTK vs Slaney) | MEDIUM | Implement dedicated BigVGANMelSpectrogram class using librosa filterbank. No model retraining needed -- fix is in the mel computation code. |
| Log compression mismatch | LOW | Implement MelAdapter with exact conversion formulas. Pure math, no model changes. |
| Center padding mismatch | LOW | Update BigVGANMelSpectrogram to use center=False with explicit padding. Update get_mel_shape. |
| Sample rate mismatch | LOW | Add resampling step in generation pipeline using existing _get_resampler. |
| Forgot remove_weight_norm | LOW | Add one line: `model.remove_weight_norm()`. No retraining or data changes. |
| Discriminator overfitting | HIGH | Requires retraining HiFi-GAN with augmentation. May need architecture changes to discriminator capacity. |
| BigVGAN OOM | MEDIUM | Implement lazy loading, CPU fallback, float16 inference. Code changes only, no retraining. |
| MPS STFT crash | LOW | Move mel computation to CPU. Minimal code change, no quality impact. |
| Broke existing models | HIGH | If not caught early: roll back code, re-implement with backward compatibility. If caught early: add version field to model format. |
| Upsampling rate mismatch | LOW | Fix upsample_rates config to multiply to hop_length. Retrain HiFi-GAN with correct config. |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Mel filterbank mismatch (Slaney vs HTK) | Phase 1 (BigVGAN Integration) | BigVGAN output from custom mel matches BigVGAN output from its own mel function (L1 < 1e-6) |
| Log compression mismatch (log1p vs log-clamp) | Phase 1 (BigVGAN Integration) | MelAdapter round-trip test: vae_mel -> vocoder_mel -> vae_mel matches original within 1e-5 |
| Center padding mismatch | Phase 1 (BigVGAN Integration) | Frame count from BigVGANMelSpectrogram matches BigVGAN's own mel_spectrogram() for same input |
| Sample rate mismatch (48 kHz vs 44.1 kHz) | Phase 1 (BigVGAN Integration) | Output audio at correct pitch/speed; duration matches requested duration within 10 ms |
| Forgot remove_weight_norm | Phase 1 (BigVGAN Integration) | BigVGAN inference benchmarks match documented performance (>50x realtime on GPU) |
| Discriminator overfitting | Phase 5 (HiFi-GAN Training) | HiFi-GAN trained on 20-file dataset produces intelligible audio; discriminator accuracy < 85% |
| BigVGAN OOM | Phase 1 (BigVGAN Integration) | Generation succeeds on 8 GB device without OOM; peak memory logged |
| MPS STFT crash | Phase 1 (BigVGAN Integration) | Full pipeline runs on MPS without errors; CPU fallback tested |
| Breaking existing models | Phase 2 (Model Persistence) | All existing v1.0 .distill files load and produce same audio as before (A/B listening test) |
| Upsampling rate mismatch | Phase 5 (HiFi-GAN Training) | Assertion at HiFi-GAN init: product(upsample_rates) == hop_length |
| Joint VAE + HiFi-GAN training | Phase 5 (HiFi-GAN Training) | Training pipeline enforces sequential: VAE fully trained before HiFi-GAN starts |
| Model file bloat | Phase 2 (Model Persistence) | .distill files with HiFi-GAN vocoder are < 50 MB; BigVGAN weights are in shared cache |
| Testing only with ground-truth mels | Phase 3 (Pipeline Integration) | Quality metrics include BigVGAN(vae_mel) comparison alongside BigVGAN(ground_truth_mel) |
| Resampling at wrong point | Phase 3 (Pipeline Integration) | No double-resampling in pipeline; audio resampled at most once |
| HiFi-GAN segment_size | Phase 5 (HiFi-GAN Training) | Pre-training validation checks segment_size < shortest audio file length |
| CUDA kernel compilation | Phase 4 (UI/CLI Integration) | Graceful fallback when use_cuda_kernel=True fails; documented as optional |
| Vendoring without version pin | Phase 1 (BigVGAN Integration) | Vendored files have commit hash in header comment |
| Checkpoint bloat in .distill | Phase 5 (HiFi-GAN Training) | Only generator state_dict stored; discriminator/optimizer/scheduler excluded |

---

## Mel Parameter Compatibility Checklist

A quick-reference checklist for validating mel spectrogram compatibility between components. Every parameter must match between the mel computation and the vocoder that consumes it.

| Parameter | Project v1.0 (HTK) | BigVGAN Required | HiFi-GAN V2 Required | Match? |
|-----------|--------------------|-----------------|-----------------------|--------|
| sample_rate | 48000 | 44100 | 44100 (match BigVGAN) | NO -- resample |
| n_fft | 2048 | 2048 | 2048 | YES |
| hop_length | 512 | 512 | 512 | YES |
| win_length | 2048 | 2048 | 2048 | YES |
| n_mels | 128 | 128 | 128 | YES |
| f_min | 0.0 | 0 | 0 | YES |
| f_max | None (Nyquist=24000) | None (Nyquist=22050) | None (Nyquist=22050) | NO -- different Nyquist |
| Mel scale | HTK | Slaney | Slaney (match BigVGAN) | **NO -- CRITICAL** |
| Mel normalization | None | Slaney area norm | Slaney area norm | **NO -- CRITICAL** |
| Log compression | log1p(x) | log(clamp(x, 1e-5)) | log(clamp(x, 1e-5)) | **NO -- CRITICAL** |
| center | True | False | False | **NO** |
| Pad mode | (torchaudio default) | reflect | reflect | Verify |
| Window | Hann | Hann | Hann | YES |
| Audio normalization | None | librosa.util.normalize * 0.95 | Match BigVGAN | NO |

**5 parameters match, 6 do not.** The mismatches are:
1. sample_rate (resolvable via resampling)
2. f_max/Nyquist (automatic from sample_rate)
3. Mel scale (requires librosa filterbank)
4. Mel normalization (requires librosa filterbank)
5. Log compression (requires MelAdapter)
6. center padding (requires explicit STFT with center=False)

---

## Sources

### High Confidence (Official Source Code, Verified)

- [BigVGAN meldataset.py](https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py) -- exact mel computation: librosa Slaney filterbank, torch.stft center=False, log(clamp(x, 1e-5)) compression. Verified by reading source.
- [BigVGAN bigvgan.py](https://github.com/NVIDIA/BigVGAN/blob/main/bigvgan.py) -- weight norm removal, from_pretrained loading, CUDA kernel option. Verified by reading source.
- [BigVGAN inference.py](https://github.com/NVIDIA/BigVGAN/blob/main/inference.py) -- standard inference workflow: load, remove_weight_norm, eval, inference_mode.
- [BigVGAN-v2 44kHz/128band/512x Model Card](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x) -- config.json with exact parameters.
- [HiFi-GAN Reference Implementation](https://github.com/jik876/hifi-gan) -- V2 config, upsample_rates, training pipeline.
- [librosa.filters.mel Documentation](https://librosa.org/doc/main/generated/librosa.filters.mel.html) -- Slaney norm default, htk option.
- [torchaudio MelSpectrogram vs librosa](https://github.com/pytorch/audio/issues/1058) -- HTK vs Slaney difference documented.
- [PyTorch MPS FFT Issue #78044](https://github.com/pytorch/pytorch/issues/78044) -- FFT operators not supported on MPS backend.

### Medium Confidence (Research Papers, Multiple Sources)

- [Training GAN Vocoder with Limited Data (ICASSP 2024)](https://arxiv.org/abs/2403.16464) -- Augmentation-Conditional Discriminator for small-data vocoder training.
- [Enhancing GAN Vocoders with Contrastive Learning Under Data-Limited Condition](https://arxiv.org/html/2309.09088) -- Contrastive learning to address discriminator overfitting.
- [Training GANs with Limited Data (NVIDIA, NeurIPS 2020)](https://arxiv.org/abs/2006.06676) -- Adaptive discriminator augmentation for limited data.
- [GANs Failure Modes](https://neptune.ai/blog/gan-failure-modes) -- Mode collapse, discriminator overfitting detection.
- [Common GAN Problems - Google ML Guide](https://developers.google.com/machine-learning/gan/problems) -- Vanishing gradients, mode collapse, convergence failure.
- [MPS operator coverage tracking (PyTorch #141287)](https://github.com/pytorch/pytorch/issues/141287) -- Ongoing MPS operator support status.
- [GPT-SoVITS BigVGAN Integration Issues (#2409)](https://github.com/RVC-Boss/GPT-SoVITS/issues/2409) -- Real-world mel parameter mismatch causing artifacts.

### Low Confidence (General Guidance, Needs Validation)

- MPS compatibility for BigVGAN inference with Snake activations -- inferred from standard PyTorch ops, not tested with BigVGAN specifically.
- HiFi-GAN V2 training convergence time on 5-500 file datasets -- extrapolated from standard HiFi-GAN training, not validated empirically.
- BigVGAN inference memory usage on 8 GB Apple Silicon -- estimated from parameter count, needs profiling.

---

*Pitfalls research for: Neural vocoder integration into existing mel-spectrogram VAE generative audio system*
*Researched: 2026-02-21*
*Confidence: HIGH -- critical pitfalls verified against official BigVGAN source code and project codebase; GAN training pitfalls supported by published research*
