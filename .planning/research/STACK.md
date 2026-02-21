# Technology Stack: v1.1 HiFi-GAN Vocoder Milestone

**Project:** Small Dataset Generative Audio -- Neural Vocoder Integration
**Researched:** 2026-02-21
**Confidence:** HIGH (verified against official repos, HuggingFace model cards, config files)

> This document covers ONLY the stack additions/changes needed for
> BigVGAN-v2 integration and optional per-model HiFi-GAN V2 training.
> The existing v1.0 stack (PyTorch 2.10.0, TorchAudio, Gradio, etc.)
> is validated and unchanged.

---

## Critical Compatibility Analysis: 48 kHz Pipeline vs BigVGAN Models

The project operates at 48 kHz internally. No BigVGAN model exists at 48 kHz.
This is the single most important architectural decision for this milestone.

### Available BigVGAN-v2 Models

| Model ID | Sample Rate | Mel Bands | n_fft | hop_size | win_size | fmin | fmax | Params | Size |
|----------|-------------|-----------|-------|----------|----------|------|------|--------|------|
| `bigvgan_v2_44khz_128band_512x` | 44100 | 128 | 2048 | 512 | 2048 | 0 | null (22050) | 122M | ~489MB |
| `bigvgan_v2_44khz_128band_256x` | 44100 | 128 | 1024 | 256 | 1024 | 0 | null (22050) | 112M | ~450MB |
| `bigvgan_v2_24khz_100band_256x` | 24000 | 100 | 1024 | 256 | 1024 | 0 | 12000 | 112M | ~450MB |
| `bigvgan_v2_22khz_80band_256x` | 22050 | 80 | 1024 | 256 | 1024 | 0 | 11025 | 112M | ~450MB |

### Recommended Model: `bigvgan_v2_44khz_128band_512x`

**Why this model:**

1. **Closest to 48 kHz** -- 44.1 kHz captures up to 22.05 kHz, well above human hearing (20 kHz). Upsampling 44.1 kHz output to 48 kHz with torchaudio.transforms.Resample is transparent and lossless for audible content.
2. **128 mel bands matches project default** -- The project already uses `n_mels=128`. This model uses 128 mel bands. No mel dimension change needed in the VAE architecture.
3. **Matching n_fft and hop_size** -- Project: `n_fft=2048, hop_length=512`. This model: `n_fft=2048, hop_size=512`. Near-identical spectral resolution.
4. **Highest quality** -- 512x upsampling with 122M parameters, trained on diverse audio (speech, environmental, instruments) for 5M steps.
5. **Universal model** -- Trained on diverse audio types, not just speech. Suitable for the project's genre-agnostic design.

### Sample Rate Strategy

```
VAE Training & Inference Pipeline:
  48 kHz audio --> resample to 44.1 kHz --> compute BigVGAN-compatible mel --> VAE encodes/decodes mel

Generation Pipeline (replacing Griffin-Lim):
  VAE decoded mel --> BigVGAN vocoder (44.1 kHz output) --> resample to 48 kHz --> spatial/export pipeline
```

The project already has a resampler cache (`_resampler_cache` in `generation.py`) and resamples at the end of the pipeline. The change is:
- **Before:** Internal processing at 48 kHz, Griffin-Lim at 48 kHz, resample at export if needed.
- **After:** Internal mel computation at 44.1 kHz parameters, BigVGAN synthesizes at 44.1 kHz, resample to 48 kHz before spatial/export.

This requires changing `SpectrogramConfig` defaults to 44.1 kHz to match BigVGAN, OR computing a separate BigVGAN-compatible mel at inference time. The latter is cleaner because it preserves backward compatibility with existing trained models.

---

## Recommended Stack Additions

### BigVGAN Integration (Inference Only)

| Technology | Version/Source | Purpose | Why |
|------------|---------------|---------|-----|
| BigVGAN (vendored) | From HuggingFace repo | Neural vocoder generator | Not available as a pip package. Must vendor the model class files (`bigvgan.py`, `activations.py`, `alias_free_activation/`, `env.py`) or use HuggingFace Hub download. |
| huggingface_hub | >=0.23.4 (1.4.1 installed) | Model download/caching | Already installed. Provides `from_pretrained()` with automatic download, caching at `~/.cache/huggingface/hub`, and version management. |
| librosa | >=0.10.0 | Mel filterbank computation | **Required** for BigVGAN mel compatibility. BigVGAN uses `librosa.filters.mel()` with Slaney normalization to build mel filterbanks. TorchAudio's `MelSpectrogram` uses HTK scale by default -- these produce DIFFERENT mel spectrograms. Using the wrong mel will produce garbage audio. |

**Confidence:** HIGH -- verified from BigVGAN source code (`meldataset.py`), model configs on HuggingFace, and NVIDIA GitHub repo.

### Mel Spectrogram Compatibility: Critical Details

The project's current mel computation (`AudioSpectrogram` class) is incompatible with BigVGAN:

| Parameter | Project (v1.0) | BigVGAN 44kHz/128band/512x | Compatible? |
|-----------|----------------|----------------------------|-------------|
| sample_rate | 48000 | 44100 | NO -- must resample |
| n_fft | 2048 | 2048 | YES |
| hop_length | 512 | 512 | YES |
| n_mels | 128 | 128 | YES |
| f_min | 0.0 | 0 | YES |
| f_max | None (24000) | None (22050) | NO -- different Nyquist |
| Mel scale | HTK (torchaudio default) | Slaney (librosa default) | **NO -- CRITICAL** |
| Normalization | None (torchaudio default) | Slaney norm (librosa default) | **NO -- CRITICAL** |
| Log compression | `torch.log1p(x)` | `torch.log(clamp(x, 1e-5))` | **NO -- CRITICAL** |
| center | True (torchaudio default) | False (BigVGAN explicit) | **NO** |
| Window | Hann (torchaudio) | Hann (torch.hann_window) | YES |

**Three incompatibilities require a dedicated BigVGAN mel computation function:**

1. **Mel scale and normalization:** librosa uses Slaney scale with area normalization by default. TorchAudio uses HTK with no normalization. These produce numerically different filterbanks.
2. **Log compression:** Project uses `log1p(x)` (log(1+x)), BigVGAN uses `log(clamp(x, min=1e-5))`. Different dynamic range mapping.
3. **Center padding:** TorchAudio defaults to `center=True` (pads signal), BigVGAN uses `center=False`.

**Solution:** Create a `BigVGANMelSpectrogram` class that uses `librosa.filters.mel()` for the filterbank and `torch.stft()` with BigVGAN-compatible parameters. This runs alongside the existing `AudioSpectrogram` -- the VAE still trains with the project's mel, but at inference the decoded mel gets converted to a BigVGAN-compatible mel for vocoding.

Actually, the cleaner approach: the VAE should train on BigVGAN-compatible mels from the start, so the decoded mel can be fed directly to BigVGAN without any intermediate conversion. This means the mel computation needs to change for new models (existing models keep their saved SpectrogramConfig for backward compatibility).

### HiFi-GAN V2 Training (Per-Model Fine-Tuning)

| Technology | Version/Source | Purpose | Why |
|------------|---------------|---------|-----|
| HiFi-GAN V2 (custom implementation) | Adapted from jik876/hifi-gan + BigVGAN discriminators | Per-model vocoder training | V2 has only 0.92M parameters (vs V1's 13.92M), making it practical to train per-model on a user's small dataset. Achieves 4.23 MOS quality. Train on the same mel spectrograms the VAE produces. |
| auraloss | >=0.4.0 | Multi-scale mel spectrogram loss | Provides `MultiResolutionSTFTLoss` with mel scaling. Used as auxiliary loss for GAN training alongside adversarial and feature matching losses. |
| nnAudio | >=0.3.3 | CQT discriminator (optional) | Provides differentiable Constant-Q Transform for the multi-scale sub-band CQT discriminator that BigVGAN-v2 uses. Only needed if implementing BigVGAN-style discriminator instead of classic HiFi-GAN MPD+MSD. |

**Confidence:** HIGH for HiFi-GAN V2 architecture (published paper, reference implementation). MEDIUM for training on small audio datasets (standard approach is large datasets; per-model fine-tuning on 5-500 files is novel territory).

### HiFi-GAN V2 Training Configuration (48 kHz Target)

For per-model HiFi-GAN V2 training, we train at 44.1 kHz to match BigVGAN mel parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sampling rate | 44100 | Match BigVGAN mel parameters |
| n_fft | 2048 | Match BigVGAN 44kHz/128band/512x |
| hop_size | 512 | Match BigVGAN 44kHz/128band/512x |
| win_size | 2048 | Match BigVGAN 44kHz/128band/512x |
| num_mels | 128 | Match BigVGAN and project default |
| fmin | 0 | Match BigVGAN |
| fmax | null (22050) | Match BigVGAN (Nyquist at 44.1kHz) |
| upsample_rates | [8, 8, 2, 2] | HiFi-GAN V2 default (8*8*2*2=256x; 512 hop * 1 would need 512x) |
| upsample_kernel_sizes | [16, 16, 4, 4] | HiFi-GAN V2 default |
| upsample_initial_channel | 128 | HiFi-GAN V2 default (compact) |
| resblock | "1" | HiFi-GAN V2 uses resblock type 1 |
| resblock_kernel_sizes | [3, 7, 11] | HiFi-GAN V2 default |
| resblock_dilation_sizes | [[1,3,5], [1,3,5], [1,3,5]] | HiFi-GAN V2 default |
| segment_size | 16384 | Adjusted for small datasets (original: 8192 at 22kHz -> ~16384 at 44kHz for same duration) |
| batch_size | 8-16 | Small datasets need smaller batches |
| learning_rate | 0.0002 | HiFi-GAN V2 default |

**Important:** HiFi-GAN V2 default upsampling gives 256x (hop_size 256 at 22kHz). For our 512 hop_size at 44.1kHz, we need upsample_rates that multiply to 512: `[8, 8, 2, 2, 2]` or `[8, 4, 4, 2, 2]`. This is a configuration adaptation, not an architecture change.

### GAN Training Loss Components

| Loss | Implementation | Weight | Purpose |
|------|---------------|--------|---------|
| Adversarial (generator) | LS-GAN loss | 1.0 | Train generator to fool discriminator |
| Adversarial (discriminator) | LS-GAN loss | 1.0 | Train discriminator to distinguish real/generated |
| Feature matching | L1 on discriminator intermediate features | 2.0 | Stabilize GAN training |
| Mel spectrogram | L1 on log-mel spectrograms | 45.0 | Perceptual reconstruction quality |
| Multi-scale mel (optional) | auraloss `MultiResolutionSTFTLoss` | 15.0 | Multi-resolution spectral fidelity |

### Discriminator Architecture

Use classic HiFi-GAN discriminators (not BigVGAN's CQT discriminator) for per-model training because:
1. **Simpler** -- MPD + MSD is well-understood and stable for small-scale training.
2. **No nnAudio dependency** -- Avoids CQT computation complexity.
3. **Proven** -- HiFi-GAN's discriminators work well even with limited data.

| Component | Description |
|-----------|-------------|
| Multi-Period Discriminator (MPD) | 5 sub-discriminators with periods [2, 3, 5, 7, 11]. Each reshapes 1D waveform into 2D by period, applies 2D convolutions. |
| Multi-Scale Discriminator (MSD) | 3 sub-discriminators at different scales (1x, 2x pool, 4x pool). Each applies 1D convolutions on progressively downsampled audio. |

---

## Model Download and Caching Strategy

### BigVGAN Model Management

| Concern | Solution |
|---------|----------|
| First download | `BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x')` downloads ~489MB to HuggingFace cache |
| Cache location | Default: `~/.cache/huggingface/hub`. Controllable via `HF_HUB_CACHE` env var or `cache_dir` parameter |
| Offline usage | After first download, works offline. `local_files_only=True` flag prevents network calls |
| Model loading | Call `model.remove_weight_norm()` after loading for inference (important for speed) |
| Device placement | Load to CPU first, then `.to(device)`. BigVGAN works on CPU, CUDA, and MPS |
| CUDA kernel | Optional `use_cuda_kernel=True` for 1.5-3x speedup on NVIDIA GPUs. Requires nvcc + ninja at first run. Skip for CPU/MPS |

### Vendoring Strategy for BigVGAN

BigVGAN is NOT a pip package. Options:

**Option A: Git submodule** (NOT recommended)
- Brings entire repo including training code, discriminators, datasets.
- 4GB+ with model weights.
- Fragile for downstream users.

**Option B: Vendor model files only** (RECOMMENDED)
- Copy only inference-needed files: `bigvgan.py`, `activations.py`, `alias_free_activation/`, `env.py`, `utils.py`
- Copy `meldataset.py` mel computation function (or reimplement with librosa).
- MIT licensed -- vendoring is explicitly allowed.
- Place in `src/distill/vocoder/bigvgan/` as a self-contained module.
- Pin to a specific commit hash for reproducibility.

**Option C: HuggingFace Hub download of model code** (VIABLE)
- `from_pretrained()` downloads model architecture files alongside weights.
- But requires adding the downloaded path to sys.path or importlib magic.
- Less explicit than vendoring, harder to modify/patch.

**Recommendation: Option B.** Vendor the 5-6 files (~50KB of code) into the project. The model weights are downloaded via HuggingFace Hub at runtime. This gives full control over the integration, easy patching for compatibility, and no runtime import hacks.

### Per-Model HiFi-GAN V2 Storage

| Artifact | Size | Storage Location |
|----------|------|-----------------|
| HiFi-GAN V2 generator | ~4MB (0.92M params) | Bundled in `.distill` model file |
| HiFi-GAN V2 discriminator+optimizer | ~50-100MB | Saved as separate checkpoint during training, discarded after training completes |
| Training checkpoint | ~150MB | `data/vocoder_training/` during training |

The `.distill` model format (v2) should add an optional `vocoder_state_dict` key alongside the existing `model_state_dict`. At ~4MB for the V2 generator, this is negligible.

---

## Dependencies: What to Add

### Required New Dependencies

```toml
# In pyproject.toml [project.dependencies]
"librosa>=0.10.0",          # BigVGAN mel filterbank (Slaney norm). ~30MB installed.
```

### Already Installed (No Changes)

| Package | Installed Version | Needed For |
|---------|------------------|------------|
| huggingface_hub | 1.4.1 | BigVGAN model download/caching via from_pretrained() |
| torch | 2.10.0+cu128 | BigVGAN inference, HiFi-GAN training |
| torchaudio | 2.10.0+cu128 | Resampling (44.1kHz <-> 48kHz) |
| numpy | 2.4.2 | Array operations |
| scipy | 1.17.0 | Signal processing (librosa dependency) |
| soundfile | 0.13.1 | Audio I/O |

### Optional Dependencies (HiFi-GAN Training Only)

```toml
# Not needed for BigVGAN inference-only usage.
# Only needed if user wants to train per-model HiFi-GAN.
# Consider making these optional via dependency group.

[dependency-groups]
vocoder-training = [
    "auraloss>=0.4.0",       # Multi-scale mel loss for GAN training
    "tensorboard>=2.0",      # Training monitoring (HiFi-GAN convention)
]
```

### What NOT to Add

| Package | Why NOT |
|---------|---------|
| nnAudio | Only needed for CQT discriminator. Use simpler MPD+MSD instead. |
| pesq | Speech-specific quality metric. Not relevant for general audio/music. |
| ninja | Only needed for BigVGAN CUDA kernel compilation. Optional optimization, not required. |
| tensorboard | Only if HiFi-GAN training monitoring is needed. The project uses Gradio for UI, not TensorBoard. Could use Rich logging instead. |
| einops | Not needed for vocoder integration. |
| pytorch-lightning | Not needed. HiFi-GAN training loop is simple enough for vanilla PyTorch. The project already has its own training loop. |

---

## Integration Points with Existing Codebase

### Files That Need Modification

| Existing File | Change | Reason |
|---------------|--------|--------|
| `audio/spectrogram.py` | Add `BigVGANMelSpectrogram` class or `bigvgan_mel()` function | Compute BigVGAN-compatible mels using librosa filterbank, torch.stft, and log compression |
| `inference/generation.py` | Replace `spectrogram.mel_to_waveform()` with vocoder call | Swap Griffin-Lim for BigVGAN neural vocoder |
| `inference/chunking.py` | Update `synthesize_continuous_mel` to skip Griffin-Lim | Continuous mel synthesis stays, only the final mel-to-audio step changes |
| `models/persistence.py` | Add vocoder_state_dict to save/load format | Store per-model HiFi-GAN V2 weights in .distill files |
| `config/defaults.py` | Add vocoder config defaults | Default vocoder type ("bigvgan"), model ID, cache settings |
| `training/runner.py` | Add vocoder training option | Orchestrate HiFi-GAN V2 training after VAE training |
| `pyproject.toml` | Add librosa dependency | Required for BigVGAN mel computation |

### New Files to Create

| New File | Purpose |
|----------|---------|
| `src/distill/vocoder/__init__.py` | Vocoder module entry point |
| `src/distill/vocoder/base.py` | Abstract vocoder interface (mel_to_waveform) |
| `src/distill/vocoder/bigvgan/` | Vendored BigVGAN model files |
| `src/distill/vocoder/bigvgan_vocoder.py` | BigVGAN wrapper (download, cache, inference) |
| `src/distill/vocoder/hifigan/` | HiFi-GAN V2 generator + discriminators |
| `src/distill/vocoder/hifigan_vocoder.py` | HiFi-GAN V2 wrapper (train, inference) |
| `src/distill/vocoder/mel_utils.py` | BigVGAN-compatible mel computation |
| `src/distill/vocoder/training.py` | HiFi-GAN V2 GAN training loop |

### Vocoder Interface Design

```python
class Vocoder(ABC):
    """Abstract vocoder interface replacing Griffin-Lim."""

    @abstractmethod
    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Parameters
        ----------
        mel : torch.Tensor
            Log-mel spectrogram [B, n_mels, T] or [B, 1, n_mels, T].

        Returns
        -------
        torch.Tensor
            Waveform [B, 1, samples] in [-1, 1] range.
        """
        ...

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Return the native sample rate of this vocoder's output."""
        ...
```

The existing `AudioSpectrogram.mel_to_waveform()` becomes a `GriffinLimVocoder` for backward compatibility, and the generation pipeline dispatches based on vocoder type.

---

## Installation

### Minimal (BigVGAN inference only)

```bash
# Add librosa to existing environment
uv add "librosa>=0.10.0"

# BigVGAN model downloads automatically on first use via huggingface_hub
# No manual download needed
```

### Full (including HiFi-GAN V2 training)

```bash
# Core dependency
uv add "librosa>=0.10.0"

# Optional: for per-model vocoder training
uv add --group vocoder-training "auraloss>=0.4.0"
```

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Universal vocoder | BigVGAN-v2 44kHz/128band/512x | BigVGAN-v2 24kHz/100band/256x | 24kHz cuts off at 12kHz -- unacceptable for professional audio (48kHz target). Would lose high frequency content. |
| Universal vocoder | BigVGAN-v2 | Vocos | Vocos is faster but BigVGAN-v2 has better universal quality across diverse audio types. Vocos optimized for speech. |
| Universal vocoder | BigVGAN-v2 | MusicHiFi | MusicHiFi is designed for music with stereo support, but not publicly available as pretrained weights. Would require training from scratch. |
| Per-model vocoder | HiFi-GAN V2 | HiFi-GAN V1 | V1 has 13.92M params vs V2's 0.92M. For per-model training on small datasets, V2's compact size prevents overfitting and fits in .distill files. |
| Per-model vocoder | HiFi-GAN V2 | BigVGAN fine-tuning | BigVGAN has 112-122M params. Fine-tuning this on 5-500 audio files would massively overfit. V2 is appropriately sized. |
| Mel computation | librosa filterbank + torch.stft | Pure torchaudio | BigVGAN was trained with librosa Slaney-norm filterbanks. Using torchaudio HTK filterbanks produces incompatible mels, even with matching parameters. Verified from BigVGAN source code. |
| Model download | HuggingFace Hub (huggingface_hub) | Manual download script | HF Hub provides versioned caching, offline support, resume on interrupted downloads. Already installed (1.4.1). |
| GAN discriminator | HiFi-GAN MPD+MSD | BigVGAN CQT discriminator | CQT discriminator requires nnAudio dependency and is designed for large-scale universal training. MPD+MSD is simpler, proven, and sufficient for per-model fine-tuning. |
| BigVGAN integration | Vendor model files | Git submodule | Submodule brings 4GB+ repo. Vendoring ~50KB of MIT-licensed code is cleaner and gives full control. |
| 44kHz model | 512x upsampling | 256x upsampling | 512x model uses same n_fft/hop as project (2048/512). 256x uses 1024/256 which would require changing VAE mel parameters. 512x is a drop-in match. |

---

## Version Compatibility Matrix

| Package | Version | Compatible With | Notes |
|---------|---------|-----------------|-------|
| librosa | >=0.10.0 | numpy 2.4.2, scipy 1.17.0 | librosa 0.10+ supports numpy 2.x. Install via uv. |
| huggingface_hub | 1.4.1 (installed) | BigVGAN from_pretrained | Exceeds BigVGAN's >=0.23.4 requirement |
| torch | 2.10.0+cu128 (installed) | BigVGAN inference, HiFi-GAN training | BigVGAN tested with PyTorch 2.3.1+ but works with 2.10 |
| torchaudio | 2.10.0+cu128 (installed) | Resampling only | Mel computation shifts to librosa-based for BigVGAN compat |
| auraloss | >=0.4.0 | torch 2.10.0, librosa | Optional, for HiFi-GAN training only |

---

## Device Compatibility Notes

### BigVGAN Inference

| Device | Status | Notes |
|--------|--------|-------|
| CUDA | Full support | Optional CUDA kernel for 1.5-3x speedup (requires nvcc + ninja) |
| MPS (Apple Silicon) | Works | Standard PyTorch ops, no CUDA kernel. May need float32 (no fp16 on MPS for some ops) |
| CPU | Works | Slower but functional. Adequate for non-real-time generation |

### HiFi-GAN V2 Training

| Device | Status | Notes |
|--------|--------|-------|
| CUDA | Full support | Recommended. GAN training benefits from GPU acceleration |
| MPS | Likely works | PyTorch 2.10 has mature MPS support. GAN training has no unusual ops |
| CPU | Works but slow | GAN training is compute-intensive. CPU training will be very slow but possible for small datasets |

### MPS Considerations

The existing project already handles MPS edge cases (e.g., `InverseMelScale` forced to CPU). BigVGAN inference uses standard PyTorch conv/activation ops that work on MPS. The custom CUDA kernel (`use_cuda_kernel=True`) should be disabled on MPS -- detect device and set accordingly.

---

## Sources

**HIGH Confidence (Official repos, model cards, source code):**
- [NVIDIA BigVGAN GitHub Repository](https://github.com/NVIDIA/BigVGAN) -- architecture, code, requirements
- [BigVGAN-v2 44kHz/128band/512x Model Card](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x) -- config.json verified
- [BigVGAN-v2 44kHz/128band/256x Model Card](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_256x) -- config.json verified
- [BigVGAN meldataset.py source](https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py) -- mel computation verified
- [BigVGAN bigvgan.py source](https://github.com/NVIDIA/BigVGAN/blob/main/bigvgan.py) -- model class verified
- [HiFi-GAN Reference Implementation](https://github.com/jik876/hifi-gan) -- V2 config, architecture, training
- [HuggingFace Hub Download Docs](https://huggingface.co/docs/huggingface_hub/en/guides/download) -- caching, offline
- [librosa.filters.mel Documentation](https://librosa.org/doc/main/generated/librosa.filters.mel.html) -- Slaney norm default
- [TorchAudio MelSpectrogram vs librosa](https://github.com/pytorch/audio/issues/1058) -- compatibility analysis

**MEDIUM Confidence (Paper, multiple sources agree):**
- [BigVGAN Paper (ICLR 2023)](https://arxiv.org/abs/2206.04658)
- [HiFi-GAN Paper (NeurIPS 2020)](https://arxiv.org/abs/2010.05646) -- V2 architecture, 0.92M params, 4.23 MOS
- [auraloss GitHub](https://github.com/csteinmetz1/auraloss) -- audio loss functions
- [nnAudio GitHub](https://github.com/KinWaiCheuk/nnAudio) -- CQT implementation
- [MusicHiFi (IEEE 2024)](https://arxiv.org/abs/2403.10493) -- stereo vocoder reference
- [PyTorchModelHubMixin docs](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins) -- from_pretrained mechanics

---
*Stack research for: v1.1 HiFi-GAN Vocoder Milestone*
*Researched: 2026-02-21*
