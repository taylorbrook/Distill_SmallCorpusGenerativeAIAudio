# Project Research Summary

**Project:** Small Dataset Generative Audio -- v1.1 HiFi-GAN Vocoder Milestone
**Domain:** Neural vocoder integration into existing mel-spectrogram VAE generative audio pipeline
**Researched:** 2026-02-21
**Confidence:** HIGH

## Executive Summary

This milestone replaces Griffin-Lim -- the weakest link in the current generation pipeline -- with neural vocoders that convert mel spectrograms to waveforms using learned models instead of iterative phase estimation. The recommended approach is a two-tier vocoder strategy: BigVGAN-v2 (`bigvgan_v2_44khz_128band_512x`, 122M params) as the universal default that immediately improves every existing model with zero training, and optional per-model HiFi-GAN V2 (0.92M params) fine-tuned on the user's specific audio for maximum fidelity. The architecture introduces a `vocoder/` module with an abstract interface, a mel normalization adapter, and lazy-loaded model management. The integration seam is surgically narrow: five callers of `spectrogram.mel_to_waveform()` all route through the same replacement point.

The single most important technical challenge is mel spectrogram compatibility. The project's VAE produces mels using torchaudio's HTK scale with `log1p` normalization at 48 kHz. BigVGAN was trained on librosa's Slaney scale with `log(clamp)` normalization at 44.1 kHz. Six parameters differ between the two mel computation pipelines, and three of these differences (mel scale, log compression, center padding) are critical enough to produce garbage audio if ignored. The solution is a dedicated `BigVGANMelSpectrogram` class using librosa filterbanks for new models, plus a `MelAdapter` that converts log1p-to-log normalization for backward-compatible inference on existing v1.0 models. BigVGAN's 44.1 kHz output is resampled to 48 kHz using the project's existing resampler infrastructure.

Key risks are: (1) mel filterbank mismatch producing unintelligible audio -- mitigated by computing BigVGAN-compatible mels from scratch rather than adapting existing mels; (2) OOM on 8 GB devices when loading BigVGAN's 122M params alongside the VAE -- mitigated by lazy loading, CPU fallback, and float16 inference; (3) GAN discriminator overfitting during per-model HiFi-GAN training on small datasets (5-500 files) -- mitigated by mandatory data augmentation, reduced discriminator capacity, and high mel-loss weighting; (4) breaking existing v1.0 models by changing the mel pipeline -- mitigated by versioning the model format and preserving backward compatibility through the adapter path.

## Key Findings

### Recommended Stack

The stack additions are minimal. Only one new dependency is required for BigVGAN inference: `librosa>=0.10.0` for Slaney-normalized mel filterbanks that match BigVGAN's training data. BigVGAN itself is vendored (5-6 MIT-licensed source files, ~50KB of code) rather than installed as a package because it is not on PyPI. Model weights (~489 MB) are downloaded once via `huggingface_hub` (already installed at v1.4.1) and cached in the HuggingFace cache directory. For optional per-model HiFi-GAN V2 training, `auraloss>=0.4.0` provides multi-resolution STFT loss. All existing dependencies (PyTorch 2.10.0, torchaudio, numpy, scipy, soundfile) remain unchanged.

**Core technologies:**
- **BigVGAN-v2 44kHz/128band/512x** (vendored): Universal neural vocoder -- 122M params, trained on diverse audio (speech, music, environmental) for 5M steps; n_fft=2048, hop_size=512, n_mels=128 match the project exactly; only sample rate differs (44.1 kHz vs 48 kHz)
- **librosa** (new dependency): Slaney-normalized mel filterbank computation -- required because BigVGAN was trained with librosa filterbanks, which are numerically incompatible with torchaudio's HTK filterbanks
- **HiFi-GAN V2** (custom implementation): Per-model vocoder -- 0.92M params (~4 MB), fast to train on small datasets, small enough to bundle in `.distill` model files; uses MPD+MSD discriminators (simpler and proven vs BigVGAN's CQT discriminator)
- **huggingface_hub** (existing): Model download/caching -- automatic download with progress, offline support via `local_files_only=True`, resume on interrupted downloads

### Expected Features

**Must have (table stakes) -- Phase A:**
- Transparent vocoder swap: press Generate, get better audio, workflow unchanged
- Automatic BigVGAN-v2 download on first use with progress indication
- Audio quality improvement over Griffin-Lim (the entire point of the upgrade)
- Same export pipeline works (WAV/MP3/FLAC/OGG, metadata, spatial audio)
- All hardware targets supported (CUDA, MPS, CPU)
- Vocoder selection in UI (dropdown) and CLI (`--vocoder` flag)
- Graceful Griffin-Lim fallback if BigVGAN download fails

**Should have (differentiators) -- Phase B:**
- Per-model HiFi-GAN V2 training integrated in Train tab and CLI
- Per-model vocoder bundled in `.distill` model file (~4 MB addition)
- Automatic vocoder selection: per-model HiFi-GAN > BigVGAN > Griffin-Lim
- Lazy vocoder loading (load only when first generation requested)

**Defer to v1.2+:**
- Full Griffin-Lim removal (keep as fallback until neural vocoders proven stable)
- CUDA kernel optimization for BigVGAN (optional 1.5-3x speedup, requires nvcc+ninja)
- Vocoder quality comparison metrics in UI
- Vocos investigation (if official 48 kHz model ships)
- Multiple BigVGAN model variants (ship only one: highest quality 44kHz/128band/512x)

### Architecture Approach

The architecture introduces a `src/distill/vocoder/` module with an abstract `Vocoder` base class that accepts VAE-normalized mel (log1p) and handles its own normalization internally. The key design choice is that callers pass the same mel tensor they always did -- the vocoder interface abstracts away all mel format conversion. The integration preserves full backward compatibility: `AudioSpectrogram` gains an optional `vocoder` parameter, and when set, delegates `mel_to_waveform()` to the vocoder instead of Griffin-Lim. BigVGAN weights live in a shared cache (downloaded once, ~489 MB); per-model HiFi-GAN V2 weights are bundled in `.distill` files (~4 MB each). HiFi-GAN training is a separate pipeline from VAE training (never joint), using the frozen VAE's mel outputs as ground truth.

**Major components:**
1. **MelAdapter** -- Stateless converter between VAE's log1p normalization and vocoder's log(clamp) normalization; runs at the vocoder boundary, keeps VAE pipeline unchanged
2. **BigVGANVocoder** -- Wraps BigVGAN-v2; handles HuggingFace download, lazy loading, weight norm removal, mel adaptation, and 44.1 kHz output
3. **HiFiGANVocoder** -- Per-model HiFi-GAN V2 generator (0.92M params); loaded from `.distill` file's optional `vocoder_state_dict`
4. **GriffinLimVocoder** -- Wraps existing `AudioSpectrogram.mel_to_waveform()` as backward-compatible fallback
5. **HiFiGANTrainer** -- Adversarial training loop with MPD+MSD discriminators, mel reconstruction loss, feature matching loss; runs independently after VAE training
6. **VocoderCache** -- Manages BigVGAN download, HuggingFace cache, offline support, progress callbacks

### Critical Pitfalls

1. **Mel filterbank scale mismatch (Slaney vs HTK)** -- BigVGAN produces garbage audio if fed HTK mels. The filterbank matrices are numerically different and the mapping is not invertible. **Avoid by:** creating a dedicated `BigVGANMelSpectrogram` class using `librosa.filters.mel()` with Slaney normalization; never attempt runtime conversion between filterbank types.

2. **Log compression mismatch (log1p vs log-clamp)** -- Feeding log1p-normalized mels to BigVGAN produces muffled, dynamically distorted audio because zero maps to 0.0 in log1p but -11.51 in BigVGAN's log(clamp). **Avoid by:** implementing `MelAdapter.vae_to_vocoder()` with exact conversion: `expm1(mel_log1p)` then `log(clamp(mel, 1e-5))`.

3. **Breaking existing v1.0 models** -- Changing mel computation invalidates all trained models; latent space meanings shift, slider mappings break. **Avoid by:** never changing mel computation for existing models (each `.distill` stores its `spectrogram_config`); new v1.1 models train on BigVGAN-compatible mels from the start; version the mel computation type in saved format.

4. **OOM on 8 GB devices** -- Loading BigVGAN's 122M params alongside the VAE can exhaust memory on MacBooks and small CUDA GPUs. **Avoid by:** lazy loading (load only on first generation), device-aware fallback to CPU, float16 inference on CUDA, chunked processing for long generations.

5. **GAN discriminator overfitting on small datasets** -- With 5-500 files, the discriminator memorizes training data and provides no useful gradient signal. **Avoid by:** mandatory data augmentation on both real and fake samples, reduced discriminator capacity, high mel-loss weight (45.0) vs adversarial weight (1.0), early stopping on mel loss not adversarial loss, minimum 20-50 file requirement for per-model training.

## Implications for Roadmap

Based on combined research, the milestone should be split into 5 phases following the dependency chain identified in both the architecture and features research. The critical path is: mel compatibility first, then BigVGAN integration, then pipeline wiring, then UI/CLI, and finally per-model HiFi-GAN training as the most complex and highest-risk phase.

### Phase 1: Vocoder Interface and BigVGAN Integration

**Rationale:** This is the highest-value, lowest-risk change. BigVGAN replaces Griffin-Lim with zero per-model training. Every existing model immediately sounds better. The mel compatibility layer (adapter + BigVGAN-compatible mel class) is the foundational dependency for everything that follows.

**Delivers:** Abstract `Vocoder` interface, `MelAdapter`, `BigVGANMelSpectrogram` class, `BigVGANVocoder` wrapper with lazy loading and HuggingFace download, `GriffinLimVocoder` fallback wrapper, `VocoderConfig` and `VocoderType` enum, vendored BigVGAN source files.

**Addresses features:** Transparent vocoder swap, automatic download, audio quality improvement, hardware support (CUDA/MPS/CPU).

**Avoids pitfalls:** Mel filterbank mismatch (#1), log compression mismatch (#2), sample rate mismatch (#4), forgot remove_weight_norm (#5), BigVGAN OOM (#7), MPS STFT crash (#8), vendoring without version pin (#17).

**Stack:** BigVGAN (vendored), librosa (new dep), huggingface_hub (existing), torchaudio Resample (existing).

**Key risk:** Mel normalization adapter producing subtly wrong audio. Validate by comparing BigVGAN output from adapter path vs BigVGAN's own mel computation on the same audio.

### Phase 2: Model Persistence Update

**Rationale:** Required before per-model HiFi-GAN training can store results, and before the pipeline can detect whether a model has a trained vocoder. Establishes the `.distill` format v2 with optional vocoder fields.

**Delivers:** `.distill` format v2 with optional `vocoder_state_dict`, `vocoder_config`, and `vocoder_type` fields; backward-compatible loading of v1 files; `mel_type` versioning in spectrogram config; `has_vocoder` indicator in model catalog.

**Addresses features:** Model format updates for vocoder state, backward compatibility.

**Avoids pitfalls:** Breaking existing models (#9), model file bloat (#12, BigVGAN weights stay in shared cache), checkpoint bloat (#18, only generator state_dict in .distill).

**Stack:** Existing persistence infrastructure.

**Key risk:** Low. Backward-compatible version bump follows established patterns in the codebase.

### Phase 3: Generation Pipeline Integration

**Rationale:** Wires the vocoder through the actual generation code paths. After this phase, calling `generate()` uses the neural vocoder. The five identified callers of `mel_to_waveform` all route through the same replacement point.

**Delivers:** Vocoder injection into `AudioSpectrogram` via optional parameter; `GenerationPipeline` carries vocoder reference; all five callers (`generate_chunks_crossfade`, `generate_chunks_latent_interp`, `_generate_chunks_from_vector`, `generate_preview`, `generate_reconstruction_preview`) use vocoder; sample rate handling for 44.1 kHz BigVGAN output to 48 kHz pipeline; single-resampling optimization.

**Addresses features:** Same export pipeline works, CLI generation works with vocoder.

**Avoids pitfalls:** Resampling at wrong point (#14), testing only with ground-truth mels (#13, validate with VAE-reconstructed mels).

**Stack:** Existing generation infrastructure.

**Key risk:** Low-medium. Small changes to well-understood code paths. Main risk is the sample rate flow (ensure single resample, correct pitch/timing).

### Phase 4: UI and CLI Integration

**Rationale:** User-facing changes come after the backend works. Surfaces vocoder selection and download progress to both Gradio UI and Typer CLI.

**Delivers:** Vocoder dropdown in Generate tab ("Auto" / "BigVGAN (Universal)" / "Per-model HiFi-GAN" / "Griffin-Lim (Legacy)"); `--vocoder` CLI flag on generate command; download progress indication; user-friendly vocoder names; vocoder status in AppState.

**Addresses features:** Vocoder selection in UI and CLI, download progress, graceful fallback.

**Avoids pitfalls:** CUDA kernel compilation (#16, default to standard PyTorch), UX pitfalls (no progress for 500 MB download, silent fallback, technical names).

**Stack:** Gradio (existing), Typer (existing), Rich (existing for CLI progress).

**Key risk:** Low. Standard UI/CLI patterns already established in the codebase.

### Phase 5: HiFi-GAN V2 Training

**Rationale:** Most complex phase -- introduces adversarial training (generator + discriminator) which is fundamentally different from existing VAE training. Deferred until BigVGAN universal proves the vocoder infrastructure works. This is an optimization for maximum fidelity, not a requirement for the core upgrade.

**Delivers:** HiFi-GAN V2 generator architecture adapted for 512 hop_length (`upsample_rates=[8,8,2,2,2]`), MPD+MSD discriminators, adversarial training loop with mel reconstruction loss (weight 45.0) + feature matching loss (weight 2.0) + adversarial loss (weight 1.0), data augmentation for discriminator, training UI integration in Train tab, `distill train-vocoder` CLI command, vocoder bundling into `.distill` files, automatic vocoder selection logic.

**Addresses features:** Per-model HiFi-GAN V2 training (UI + CLI), vocoder bundling in .distill, auto-selection.

**Avoids pitfalls:** Discriminator overfitting (#6, mandatory augmentation + reduced capacity), upsampling rate mismatch (#10, assert product=512), joint training instability (#11, sequential only), segment size mismatch (#15, validate against dataset), checkpoint bloat (#18, only generator in .distill).

**Stack:** Custom HiFi-GAN V2 implementation, auraloss (optional dep for multi-resolution STFT loss), existing training infrastructure.

**Key risk:** High. GAN training on 5-500 files is novel territory. Discriminator overfitting is the primary failure mode. Training time estimates (30-60 min fine-tune, 2-9 hours from scratch) are extrapolated, not validated. Minimum viable dataset likely 20-50 files.

### Phase Ordering Rationale

- **Dependency-driven:** Phase 1 creates the vocoder infrastructure everything else depends on. Phase 2 establishes persistence before Phase 5 needs to store results. Phase 3 connects backend to pipeline. Phase 4 connects pipeline to user. Phase 5 is the riskiest and most independent work.
- **Value-first:** Phase 1 alone delivers the core value proposition (better audio from existing models). Phases 2-4 are wiring. Phase 5 is an optimization.
- **Risk-ordered:** Front-loads the mel compatibility challenge (Phase 1) when it is cheapest to fix. Defers GAN training complexity (Phase 5) until the vocoder infrastructure is proven.
- **Backward compatibility preserved throughout:** No phase breaks existing v1.0 models. The adapter path ensures existing models work with BigVGAN from Phase 1. Format versioning in Phase 2 ensures clean upgrade path.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (BigVGAN Integration):** Needs validation of MPS compatibility for BigVGAN inference with Snake activations. Needs empirical testing of mel adapter quality (HTK-trained VAE mels through adapter vs native BigVGAN mels). Memory profiling on 8 GB devices needed.
- **Phase 5 (HiFi-GAN Training):** Needs deeper research into GAN training on small datasets. Augmentation strategies (differentiable augmentations for both real and fake) are critical and partially research-grade. Training time and convergence behavior on 5-500 file datasets is unvalidated. Consider studying ICASSP 2024 Augmentation-Conditional Discriminator paper during phase planning.

Phases with standard patterns (skip research-phase):
- **Phase 2 (Model Persistence):** Backward-compatible format versioning is a well-understood pattern. The codebase already has version checking.
- **Phase 3 (Pipeline Integration):** Five callers clearly identified. Changes are mechanical -- inject vocoder, handle sample rate. Standard refactoring.
- **Phase 4 (UI/CLI):** Gradio dropdown and Typer option are established patterns in the codebase. No novel design decisions.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | BigVGAN model parameters verified from official HuggingFace config.json. librosa mel filterbank difference verified from source code and PyTorch issue #1058. HiFi-GAN V2 architecture from original NeurIPS 2020 paper. All version compatibility confirmed. |
| Features | HIGH | Table stakes features are clearly scoped (vocoder swap + selection UI). Differentiators (per-model training) have clear architecture. Anti-features well-defined. Feature dependencies mapped with critical path identified. |
| Architecture | HIGH | Integration seam narrowly identified (5 callers). Component boundaries clean. Abstract vocoder interface preserves backward compatibility. Build order follows natural dependencies. BigVGAN vendoring strategy verified (MIT license, ~50KB code). |
| Pitfalls | HIGH | Critical pitfalls (mel mismatch, log compression, backward compatibility) verified against BigVGAN source code. GAN overfitting on small data supported by ICASSP 2024 research. MPS STFT limitation documented in PyTorch issue tracker. Recovery strategies identified for all critical pitfalls. |

**Overall confidence:** HIGH

This is a well-defined integration milestone with a narrow scope. The target model (BigVGAN-v2 44kHz/128band/512x) is a near-perfect parameter match for the project (n_fft, hop_size, n_mels all identical). The mel compatibility challenge is well-understood and has clear solutions. The primary uncertainty is in Phase 5 (per-model GAN training on small datasets), which is deferrable without losing the core value of the upgrade.

### Gaps to Address

1. **MPS compatibility for BigVGAN inference** -- BigVGAN uses Snake activations and anti-aliased upsampling that have not been tested on MPS. PyTorch 2.10 has expanded MPS operator coverage but edge cases may exist. **Handle during Phase 1:** test early on Apple Silicon, implement CPU fallback for mel computation (STFT), verify inference runs on MPS device.

2. **Quality of adapter path for v1.0 models** -- Existing models were trained on HTK mels. The MelAdapter converts log normalization but cannot fix the filterbank difference (Slaney vs HTK). BigVGAN's universality should handle this, but quality may be slightly lower than native BigVGAN mels. **Handle during Phase 1:** conduct A/B listening tests comparing adapter path vs Griffin-Lim for v1.0 models. If adapter path sounds worse than Griffin-Lim (should never happen), fall back to Griffin-Lim for legacy models.

3. **HiFi-GAN V2 training convergence on 5-50 file datasets** -- Standard HiFi-GAN training assumes thousands of files. Per-model fine-tuning on very small datasets is novel territory. Training time estimates are extrapolated. **Handle during Phase 5:** start with 50-file validation, test down to 20 files, document minimum dataset size requirement, implement early stopping on mel loss.

4. **BigVGAN memory footprint on 8 GB devices** -- 122M params in float32 = ~489 MB. Combined with VAE and intermediate tensors, may exceed 8 GB unified memory on base MacBooks. **Handle during Phase 1:** profile peak memory, implement float16 inference on CUDA, test CPU fallback path, consider chunked processing for long generations.

5. **HiFi-GAN V2 upsampling configuration for 512 hop** -- Standard V2 upsample_rates multiply to 256. Need `[8,8,2,2,2]` (5 layers) for 512x. This adds one upsampling layer vs standard V2, increasing model size from 0.92M to ~5-8M params. **Handle during Phase 5:** validate this configuration produces quality audio, update size estimates for `.distill` bundling.

## Sources

### Primary (HIGH confidence)
- [NVIDIA BigVGAN GitHub Repository](https://github.com/NVIDIA/BigVGAN) -- Architecture, source code, mel computation, weight norm handling
- [BigVGAN-v2 44kHz/128band/512x Model Card](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x) -- config.json with exact parameters verified
- [BigVGAN meldataset.py](https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py) -- Mel computation: librosa Slaney filterbank, center=False, log(clamp) compression
- [HiFi-GAN Reference Implementation](https://github.com/jik876/hifi-gan) -- V2 config (0.92M params, 4.23 MOS), architecture, training pipeline
- [librosa.filters.mel Documentation](https://librosa.org/doc/main/generated/librosa.filters.mel.html) -- Slaney norm default confirmed
- [torchaudio MelSpectrogram vs librosa Issue #1058](https://github.com/pytorch/audio/issues/1058) -- HTK vs Slaney incompatibility documented
- [HuggingFace Hub Download Docs](https://huggingface.co/docs/huggingface_hub/en/guides/download) -- Caching, offline support, progress callbacks

### Secondary (MEDIUM confidence)
- [BigVGAN Paper (ICLR 2023)](https://arxiv.org/abs/2206.04658) -- Anti-aliased multi-periodicity composition, Snake activation
- [HiFi-GAN Paper (NeurIPS 2020)](https://arxiv.org/abs/2010.05646) -- V2 architecture details, discriminator design, loss functions
- [Training GAN Vocoder with Limited Data (ICASSP 2024)](https://arxiv.org/abs/2403.16464) -- Augmentation-Conditional Discriminator for small-data training
- [Training GANs with Limited Data (NVIDIA, NeurIPS 2020)](https://arxiv.org/abs/2006.06676) -- Adaptive discriminator augmentation
- [NVIDIA BigVGAN-v2 Blog Post](https://developer.nvidia.com/blog/achieving-state-of-the-art-zero-shot-waveform-audio-generation-across-audio-types/) -- v2 capabilities
- [auraloss GitHub](https://github.com/csteinmetz1/auraloss) -- Multi-resolution STFT loss for GAN training
- [PyTorch MPS FFT Issue #78044](https://github.com/pytorch/pytorch/issues/78044) -- FFT operators on MPS backend

### Tertiary (LOW confidence, needs validation)
- MPS compatibility for BigVGAN inference with Snake activations -- inferred from standard PyTorch ops, not tested
- HiFi-GAN V2 convergence time on 5-500 file datasets -- extrapolated from standard training, not empirically validated
- BigVGAN peak memory on 8 GB Apple Silicon -- estimated from parameter count, needs profiling
- Quality of mel adapter path (HTK VAE mels -> BigVGAN via normalization conversion) -- theoretically sound, needs listening tests

---
*Research completed: 2026-02-21*
*Ready for roadmap: yes*
