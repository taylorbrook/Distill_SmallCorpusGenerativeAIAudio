# Architecture: Neural Vocoder Integration

**Domain:** HiFi-GAN / BigVGAN-v2 neural vocoder integration into existing generative audio pipeline
**Researched:** 2026-02-21
**Confidence:** HIGH

## Executive Summary

The neural vocoder replaces Griffin-Lim (the weakest link in the current pipeline) with a learned mel-to-waveform conversion. The integration point is surgically narrow: only `AudioSpectrogram.mel_to_waveform()` and a small set of callers need to change. However, the mel spectrogram normalization mismatch between the existing project (`log1p`) and BigVGAN (`log(clamp(x))`) is a critical compatibility issue that drives several architectural decisions.

The recommended architecture introduces a `vocoder/` module alongside the existing `audio/` module, a mel normalization adapter layer, a shared vocoder cache for BigVGAN weights, optional per-model HiFi-GAN state bundled into `.distill` files, and a new HiFi-GAN training pipeline that runs independently from the VAE training loop.

## Current Generation Pipeline

```
VAE Decoder                        AudioSpectrogram               Output
   |                                    |                           |
   z [B, latent_dim]                    |                           |
   |                                    |                           |
   model.decode(z)                      |                           |
   |                                    |                           |
   mel_log [B, 1, n_mels, time]        |                           |
   |                                    |                           |
   +--- spectrogram.mel_to_waveform(mel_log) ----+                  |
        |                                         |                 |
        torch.expm1(mel_log)  [undo log1p]       |                 |
        |                                         |                 |
        InverseMelScale(mel) -> linear_spec      |                 |
        |                                         |                 |
        GriffinLim(linear_spec) -> waveform  ----+----> waveform   |
```

### Current Callers of mel_to_waveform

| Caller | File | Context |
|--------|------|---------|
| `generate_chunks_crossfade` | `inference/chunking.py` | Main generation path |
| `generate_chunks_latent_interp` | `inference/chunking.py` | Alternative generation path |
| `_generate_chunks_from_vector` | `inference/generation.py` | Slider-controlled generation |
| `generate_preview` | `training/preview.py` | Training preview audio |
| `generate_reconstruction_preview` | `training/preview.py` | Original vs. reconstruction |

All five callers use the same pattern: `spectrogram.mel_to_waveform(combined_mel)`. This is the single integration seam.

## Proposed Architecture

### System Overview

```
VAE Decoder                   Vocoder Router                    Output
   |                               |                              |
   mel_log [B, 1, n_mels, T]      |                              |
   |                               |                              |
   +--- vocoder_router(mel_log) ---+                              |
        |                          |                              |
        +--- BigVGAN path ----+    +--- HiFi-GAN path ----+     |
        |                     |    |                       |     |
        | mel_adapter(mel_log)|    | mel_adapter(mel_log)  |     |
        | [undo log1p, apply  |    | [undo log1p, apply   |     |
        |  log(clamp)]        |    |  log(clamp)]         |     |
        |                     |    |                       |     |
        | bigvgan(mel) -------+    | hifigan(mel) --------+     |
        |                          |                              |
        +--- waveform [B, 1, T] --+--- waveform [B, 1, T] ------+
```

### New Module: `src/distill/vocoder/`

```
src/distill/vocoder/
    __init__.py              # Public API: VocoderType, get_vocoder, VocoderConfig
    adapter.py               # Mel normalization adapter (log1p <-> log(clamp))
    base.py                  # Abstract Vocoder interface
    bigvgan.py               # BigVGAN-v2 wrapper (loads from HF, caches)
    hifigan.py               # HiFi-GAN V2 model definition + wrapper
    training.py              # HiFi-GAN adversarial training loop
    discriminator.py         # MPD + MSD/CQTD discriminators for HiFi-GAN training
    config.py                # VocoderConfig, HiFiGANTrainingConfig
    cache.py                 # Shared model cache management
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| `vocoder.base.Vocoder` | Abstract interface: `mel_to_waveform(mel_log) -> waveform` | Called by `AudioSpectrogram`, `chunking.py`, `generation.py` |
| `vocoder.adapter.MelAdapter` | Converts between VAE's log1p normalization and vocoder's log(clamp) normalization | Used by BigVGAN and HiFi-GAN wrappers |
| `vocoder.bigvgan.BigVGANVocoder` | Loads BigVGAN-v2 from HuggingFace, runs inference | Uses `adapter`, `cache` |
| `vocoder.hifigan.HiFiGANVocoder` | HiFi-GAN V2 generator model definition and inference | Uses `adapter` |
| `vocoder.training.HiFiGANTrainer` | Adversarial training loop for per-model HiFi-GAN | Uses `hifigan`, `discriminator`, existing training data |
| `vocoder.discriminator` | Multi-Period + Multi-Scale discriminators | Used only by `training` |
| `vocoder.cache.VocoderCache` | Downloads/caches BigVGAN weights, tracks HiFi-GAN checkpoints | Used by `bigvgan`, persistence |
| `vocoder.config.VocoderConfig` | Vocoder selection + parameters | Stored in `.distill` files, used by UI/CLI |

### Modified Existing Components

| Component | Change | Reason |
|-----------|--------|--------|
| `audio/spectrogram.py` | Add optional `vocoder` parameter to constructor; delegate `mel_to_waveform` to vocoder when present | Backward-compatible: default behavior unchanged |
| `models/persistence.py` | Add `vocoder_state_dict` and `vocoder_config` to saved model dict; bump `SAVED_MODEL_VERSION` to 2 | Store per-model HiFi-GAN weights alongside VAE |
| `inference/generation.py` | Accept vocoder in `GenerationPipeline.__init__`; pass to spectrogram | Pipeline carries vocoder through |
| `training/preview.py` | Use vocoder (if available) for preview generation | Better preview quality during training |
| `ui/tabs/generate_tab.py` | Add vocoder selection dropdown | User selects BigVGAN vs HiFi-GAN vs Griffin-Lim |
| `ui/tabs/train_tab.py` | Add "Train Vocoder" section | HiFi-GAN training UI |
| `ui/state.py` | Add `vocoder` field to `AppState` | Track active vocoder |
| `cli/generate.py` | Add `--vocoder` flag | CLI vocoder selection |
| `cli/train.py` | Add `train-vocoder` subcommand | CLI HiFi-GAN training |
| `config/defaults.py` | Add `vocoder` section to `DEFAULT_CONFIG` | Default vocoder settings |

## Critical Design Decisions

### Decision 1: Mel Normalization Adapter

**Problem:** The VAE produces mel spectrograms normalized with `log1p(mel)` (range: 0 to ~11 for typical audio). BigVGAN and HiFi-GAN expect `log(clamp(mel, min=1e-5))` (range: ~-11.5 to ~11). These are NOT interchangeable -- feeding one format to a model trained on the other produces garbage audio.

**Solution:** A stateless adapter that converts between normalizations at the vocoder boundary.

```python
class MelAdapter:
    """Convert between VAE mel normalization and vocoder mel normalization."""

    @staticmethod
    def vae_to_vocoder(mel_log1p: torch.Tensor) -> torch.Tensor:
        """Convert log1p-normalized mel to log-normalized mel.

        VAE output:    log1p(mel) = log(1 + mel)
        Vocoder input: log(clamp(mel, min=1e-5))

        Conversion:    mel = expm1(mel_log1p)
                       mel_vocoder = log(clamp(mel, min=1e-5))
        """
        mel_linear = torch.expm1(mel_log1p.clamp(min=0))
        return torch.log(torch.clamp(mel_linear, min=1e-5))

    @staticmethod
    def vocoder_to_vae(mel_log: torch.Tensor) -> torch.Tensor:
        """Convert log-normalized mel back to log1p-normalized mel.

        Used during HiFi-GAN training when computing mel loss
        on VAE-reconstructed spectrograms.
        """
        mel_linear = torch.exp(mel_log)
        return torch.log1p(mel_linear)
```

**Confidence:** HIGH -- both normalizations are well-defined, and the conversion is mathematically exact.

**Why not retrain the VAE with different normalization?** This would invalidate all existing trained models and require retraining. The adapter is zero-cost at inference time and preserves backward compatibility.

### Decision 2: BigVGAN-v2 Model Variant Selection

**Problem:** BigVGAN-v2 has multiple pretrained checkpoints. The project uses 48kHz/128 mels/512 hop_length/2048 n_fft. No BigVGAN variant matches 48kHz exactly.

**Available BigVGAN-v2 variants:**

| Variant | Sample Rate | n_mels | hop_size | n_fft | Params | Match Quality |
|---------|-------------|--------|----------|-------|--------|---------------|
| `44khz_128band_512x` | 44100 | 128 | 512 | 2048 | 122M | Closest: n_mels, hop, n_fft match |
| `44khz_128band_256x` | 44100 | 128 | 256 | 1024 | 112M | n_mels match, but hop/n_fft differ |
| `24khz_100band_256x` | 24000 | 100 | 256 | 1024 | 112M | Nothing matches |
| `22khz_80band_256x` | 22050 | 80 | 256 | 1024 | -- | Nothing matches |

**Recommendation:** Use `bigvgan_v2_44khz_128band_512x` (122M params).

Rationale:
- **n_mels=128** matches exactly
- **hop_size=512** matches exactly
- **n_fft=2048** matches exactly
- **fmin=0, fmax=null** matches exactly
- Only **sample_rate differs**: 44100 vs 48000

The sample rate mismatch (44.1kHz vs 48kHz) is handled by:
1. Resample the VAE's mel spectrogram time axis from 48kHz-equivalent to 44.1kHz-equivalent frame rate (ratio: 44100/48000 = 0.91875), or
2. Generate BigVGAN output at 44.1kHz and resample the waveform to 48kHz (the project already has resampling infrastructure in `_get_resampler`)

Option 2 is simpler and recommended: let BigVGAN produce 44.1kHz audio, then resample to 48kHz. The resampler already exists in the generation pipeline.

**Confidence:** HIGH for the model selection. MEDIUM for the sample rate handling -- perceptual testing needed to confirm quality.

### Decision 3: Vocoder Storage Strategy

**Problem:** Where do vocoder model weights live? Options:
1. Bundle vocoder state into each `.distill` file
2. Separate vocoder files alongside `.distill` files
3. Shared cache directory

**Recommendation:** Hybrid approach with three tiers.

```
Tier 1: BigVGAN (shared cache, ~500MB one-time download)
    data/vocoders/bigvgan_v2_44khz_128band_512x/
        bigvgan_generator.pt
        config.json

Tier 2: Per-model HiFi-GAN (bundled in .distill file)
    data/models/my_model.distill
        format: "distill_model"
        version: 2
        model_state_dict: {...}          # VAE weights
        vocoder_state_dict: {...}        # HiFi-GAN weights (optional)
        vocoder_config: {...}            # HiFi-GAN config (optional)
        spectrogram_config: {...}
        latent_analysis: {...}
        training_config: {...}
        metadata: {...}

Tier 3: No vocoder (Griffin-Lim fallback)
    Older .distill files without vocoder_state_dict
    Still work with BigVGAN universal or Griffin-Lim
```

**Rationale:**
- BigVGAN is 122M params (~500MB on disk). Bundling it in every `.distill` file would waste disk space and slow save/load. A shared cache downloaded once is the right approach.
- Per-model HiFi-GAN is small enough to bundle (~5-15MB) and is model-specific, so it belongs in the `.distill` file.
- Griffin-Lim requires zero storage. Keeping it as fallback ensures models work even without vocoder downloads.

**Model version bump:** `SAVED_MODEL_VERSION = 2`. Version 1 files load fine (no vocoder fields). Version 2 files include optional vocoder fields.

**Confidence:** HIGH

### Decision 4: HiFi-GAN Training Integration

**Problem:** HiFi-GAN training is adversarial (generator + discriminator), which is fundamentally different from the existing VAE training loop. How to integrate?

**Recommendation:** Separate training pipeline, not integrated into the VAE training loop.

```
Existing VAE Training Loop (unchanged)
    train() -> {model, metrics, checkpoint}
                |
                v (after VAE training completes)
HiFi-GAN Training Pipeline (new, optional)
    train_vocoder(
        vae_model,        # Frozen, used to generate mel specs
        audio_files,      # Same training data
        vocoder_config,   # HiFi-GAN architecture config
    ) -> {vocoder, metrics}
```

**Why separate?**
1. VAE training produces mel spectrograms; HiFi-GAN training consumes them. They have a producer-consumer relationship, not a joint optimization.
2. HiFi-GAN has its own discriminator, loss functions (adversarial + mel-spec L1), and optimizer -- mixing these with VAE training would be fragile.
3. HiFi-GAN training is optional (BigVGAN universal is the default).
4. The user should be able to train a HiFi-GAN vocoder for an already-saved model without retraining the VAE.

**HiFi-GAN training data flow:**

```
For each batch:
    1. Load audio waveforms from dataset
    2. Compute mel spectrogram (using VAE's spectrogram config)
    3. Convert mel to vocoder normalization (MelAdapter)
    4. Generator: mel -> waveform_gen
    5. Discriminator: evaluate waveform_gen vs waveform_real
    6. Loss: adversarial + mel-spectrogram L1 (on generated vs real)
    7. Update generator and discriminator
```

**Training time estimate:** HiFi-GAN fine-tuning from scratch takes ~200K-500K steps. At ~3 steps/sec on a consumer GPU, that is roughly 18-46 hours. However, since this is on a small dataset (5-500 files), convergence should be faster -- likely 50K-100K steps (4-9 hours).

**Confidence:** HIGH for architecture, MEDIUM for training time estimates.

### Decision 5: Vocoder Selection and Routing

**Problem:** Three vocoder backends with different characteristics. How does the user choose?

| Backend | Quality | Speed | Requirements | When to Use |
|---------|---------|-------|-------------|-------------|
| BigVGAN-v2 | Very High | ~50x realtime on GPU | ~500MB download, GPU recommended | Default for all models |
| Per-model HiFi-GAN | Highest (tuned) | ~150x realtime on GPU | Training required | Maximum fidelity after vocoder training |
| Griffin-Lim | Low | ~5x realtime on CPU | None | Fallback, quick previews |

**Recommendation:** Enum-based selection with automatic fallback.

```python
class VocoderType(Enum):
    AUTO = "auto"           # HiFi-GAN if available, else BigVGAN, else Griffin-Lim
    BIGVGAN = "bigvgan"     # BigVGAN-v2 universal
    HIFIGAN = "hifigan"     # Per-model HiFi-GAN (error if not trained)
    GRIFFIN_LIM = "griffin_lim"  # Legacy Griffin-Lim

def get_vocoder(
    vocoder_type: VocoderType,
    model_path: Path | None = None,    # For loading HiFi-GAN from .distill
    device: torch.device = "cpu",
    cache_dir: Path | None = None,     # For BigVGAN cache
) -> Vocoder:
    ...
```

`AUTO` checks: (1) does this `.distill` file contain HiFi-GAN weights? If yes, use them. (2) Is BigVGAN cached or downloadable? If yes, use it. (3) Fall back to Griffin-Lim.

**Confidence:** HIGH

## Data Flow Changes

### Generation Pipeline (Modified)

```
Before (v1.0):
    chunking.py: synthesize_continuous_mel() -> mel [B, 1, n_mels, T]
    chunking.py: spectrogram.mel_to_waveform(mel) -> waveform  (Griffin-Lim)

After (v1.1):
    chunking.py: synthesize_continuous_mel() -> mel [B, 1, n_mels, T]
    chunking.py: vocoder.mel_to_waveform(mel) -> waveform  (BigVGAN/HiFi-GAN/Griffin-Lim)
```

The vocoder is injected through `AudioSpectrogram` or passed alongside it. The change is localized to the five callers listed above.

### Model Save/Load (Modified)

```
Save (v1.1):
    save_model() now accepts optional vocoder_state_dict + vocoder_config
    Writes version: 2 in saved dict
    Vocoder state is None when no per-model HiFi-GAN trained

Load (v1.1):
    load_model() reads vocoder_state_dict if present
    Reconstructs HiFi-GAN generator if vocoder state exists
    Returns LoadedModel with new vocoder field
    Backward compatible: version 1 files load with vocoder=None
```

### Training Pipeline (Extended)

```
Existing VAE flow (unchanged):
    audio files -> DataLoader -> waveform -> mel -> VAE -> loss -> optimize

New HiFi-GAN flow (separate, optional):
    audio files -> DataLoader -> waveform --------+
                                                   |
    audio files -> DataLoader -> waveform -> mel --+
                                                   |
    HiFi-GAN Generator: mel -> waveform_gen ------+
                                                   |
    Discriminator: (waveform_real, waveform_gen) --+
                                                   |
    Loss: adversarial + mel_L1 -> optimize generator + discriminator
```

## Vocoder Abstract Interface

```python
from abc import ABC, abstractmethod
import torch

class Vocoder(ABC):
    """Abstract base for all vocoder backends."""

    @abstractmethod
    def mel_to_waveform(self, mel_log: torch.Tensor) -> torch.Tensor:
        """Convert log1p-normalized mel spectrogram to waveform.

        Parameters
        ----------
        mel_log : torch.Tensor
            Shape [B, 1, n_mels, time] -- VAE's log1p normalization.

        Returns
        -------
        torch.Tensor
            Shape [B, 1, samples] -- audio waveform.
        """
        ...

    @abstractmethod
    def to(self, device: torch.device) -> "Vocoder":
        """Move vocoder to device."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native output sample rate of this vocoder."""
        ...
```

The key design choice: the vocoder interface accepts **VAE-normalized mel** (log1p), not vocoder-normalized mel. Each implementation handles its own normalization internally via `MelAdapter`. This keeps callers simple -- they pass the same mel tensor they always did.

## HiFi-GAN V2 Architecture (for per-model training)

Use HiFi-GAN V2 (not V1 or V3) because it balances quality and model size. Configuration adapted for this project:

```python
@dataclass
class HiFiGANConfig:
    """HiFi-GAN V2 generator configuration for 48kHz audio."""

    # Generator
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 2, 2, 2])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4, 4])
    upsample_initial_channel: int = 128
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )

    # Matches existing SpectrogramConfig
    sample_rate: int = 48_000
    n_fft: int = 2048
    hop_size: int = 512
    n_mels: int = 128

    # Training
    learning_rate: float = 2e-4
    adam_betas: tuple[float, float] = (0.8, 0.99)
    lr_decay: float = 0.999
    segment_size: int = 24_576  # ~0.5 sec at 48kHz (must be multiple of product(upsample_rates))
```

Note: `product(upsample_rates) = 8 * 8 * 2 * 2 * 2 = 512` which matches the project's `hop_length=512`. This means each mel frame produces 512 audio samples, which is exactly right.

**Model size estimate:** ~5-8M parameters (~20-30MB on disk). Small enough to bundle in `.distill` files.

## Integration with AudioSpectrogram

The cleanest integration preserves backward compatibility while enabling vocoder injection:

```python
class AudioSpectrogram:
    def __init__(
        self,
        config: SpectrogramConfig | None = None,
        vocoder: Vocoder | None = None,  # NEW: optional vocoder
    ) -> None:
        ...
        self._vocoder = vocoder

    def mel_to_waveform(self, mel_log: torch.Tensor) -> torch.Tensor:
        """Convert mel to waveform using vocoder or Griffin-Lim fallback."""
        if self._vocoder is not None:
            return self._vocoder.mel_to_waveform(mel_log)

        # Existing Griffin-Lim path (unchanged)
        mel = torch.expm1(mel_log.squeeze(1).clamp(min=0))
        linear_spec = self.inverse_mel(mel.cpu())
        waveform = self.griffin_lim(linear_spec)
        return waveform.unsqueeze(1)

    def set_vocoder(self, vocoder: Vocoder | None) -> None:
        """Set or clear the vocoder backend."""
        self._vocoder = vocoder
```

This approach:
- Zero breaking changes for existing code
- All callers of `mel_to_waveform` automatically get vocoder output
- Vocoder can be swapped at runtime (useful for A/B comparison)
- Griffin-Lim remains as fallback when `vocoder=None`

## BigVGAN Integration Details

### Loading and Caching

```python
class BigVGANVocoder(Vocoder):
    """BigVGAN-v2 universal vocoder wrapper."""

    MODEL_ID = "nvidia/bigvgan_v2_44khz_128band_512x"
    NATIVE_SAMPLE_RATE = 44100

    def __init__(self, device: torch.device, cache_dir: Path | None = None):
        # Load from HuggingFace (cached after first download)
        import bigvgan as bigvgan_lib
        self._model = bigvgan_lib.BigVGAN.from_pretrained(
            self.MODEL_ID,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self._model.remove_weight_norm()
        self._model.eval().to(device)
        self._adapter = MelAdapter()
        self._device = device

    def mel_to_waveform(self, mel_log: torch.Tensor) -> torch.Tensor:
        # 1. Squeeze channel dim: [B, 1, n_mels, T] -> [B, n_mels, T]
        mel_squeezed = mel_log.squeeze(1)

        # 2. Convert normalization: log1p -> log(clamp)
        mel_vocoder = self._adapter.vae_to_vocoder(mel_squeezed)

        # 3. Run BigVGAN inference
        with torch.inference_mode():
            wav = self._model(mel_vocoder.to(self._device))

        # 4. Return [B, 1, samples]
        return wav.unsqueeze(1) if wav.ndim == 2 else wav

    @property
    def sample_rate(self) -> int:
        return self.NATIVE_SAMPLE_RATE  # 44100 -- caller resamples to 48kHz
```

### Dependency: `bigvgan` package

BigVGAN is NOT on PyPI. It is cloned from GitHub and installed locally, or loaded via `huggingface_hub`. The recommended approach is to vendor-install the BigVGAN package:

```bash
pip install git+https://github.com/NVIDIA/BigVGAN.git
```

Or add to `pyproject.toml`:
```toml
dependencies = [
    ...
    "bigvgan @ git+https://github.com/NVIDIA/BigVGAN.git",
]
```

Alternatively, the BigVGAN model can be loaded directly via `torch.hub` or by cloning the repo. The HuggingFace Hub integration is the cleanest path.

**MPS Compatibility Note:** BigVGAN uses custom CUDA kernels for anti-aliased activations (`use_cuda_kernel=True`). On MPS/CPU, set `use_cuda_kernel=False` (the default). All standard PyTorch ops in BigVGAN should work on MPS. However, this needs verification -- the Snake activation and filtered nonlinearities may have MPS edge cases.

**Confidence:** MEDIUM for MPS compatibility -- needs testing.

## Suggested Build Order

### Phase 1: Vocoder Interface and BigVGAN Integration (Core Path)

**Goal:** Replace Griffin-Lim with BigVGAN-v2 for generation.

**New files:**
- `vocoder/__init__.py`
- `vocoder/base.py` (Vocoder ABC)
- `vocoder/adapter.py` (MelAdapter)
- `vocoder/bigvgan.py` (BigVGANVocoder)
- `vocoder/config.py` (VocoderType, VocoderConfig)
- `vocoder/cache.py` (download/cache management)

**Modified files:**
- `audio/spectrogram.py` (add vocoder parameter)
- `config/defaults.py` (add vocoder section)

**Why first:** This is the highest-value change. BigVGAN replaces Griffin-Lim with zero per-model training required. Every existing model immediately sounds better.

**Dependencies:** None on existing training pipeline. Only touches generation path.

**Risk:** LOW-MEDIUM. Mel normalization adapter is the main risk. Testing with existing trained models validates it.

### Phase 2: Model Persistence Update

**Goal:** Version 2 `.distill` format that supports vocoder state.

**Modified files:**
- `models/persistence.py` (version bump, optional vocoder fields)
- `library/catalog.py` (add vocoder metadata to entries)

**Why second:** Required before HiFi-GAN training (need to store results). But BigVGAN-only path (Phase 1) works without this -- BigVGAN weights are shared, not per-model.

**Dependencies:** Phase 1 (vocoder config types).

**Risk:** LOW. Backward-compatible version bump. Well-understood pattern from existing code.

### Phase 3: Generation Pipeline Integration

**Goal:** Wire vocoder through GenerationPipeline, chunking, and preview.

**Modified files:**
- `inference/generation.py` (GenerationPipeline accepts vocoder)
- `inference/chunking.py` (pass vocoder through)
- `training/preview.py` (use vocoder for previews)

**Why third:** Makes the vocoder actually reachable from the UI/CLI. Phase 1 created the vocoder; Phase 3 connects it.

**Dependencies:** Phase 1 (vocoder exists).

**Risk:** LOW. Small changes to well-understood code paths. The five callers are clearly identified.

### Phase 4: UI and CLI Integration

**Goal:** User can select vocoder type, see download progress.

**Modified files:**
- `ui/tabs/generate_tab.py` (vocoder dropdown)
- `ui/state.py` (vocoder field)
- `cli/generate.py` (--vocoder flag)

**Why fourth:** User-facing changes come after the backend works.

**Dependencies:** Phases 1-3 (working vocoder pipeline).

**Risk:** LOW. Standard Gradio/Typer UI patterns already established.

### Phase 5: HiFi-GAN V2 Model + Training

**Goal:** Per-model HiFi-GAN training for maximum fidelity.

**New files:**
- `vocoder/hifigan.py` (HiFi-GAN V2 generator)
- `vocoder/discriminator.py` (MPD + MSD discriminators)
- `vocoder/training.py` (adversarial training loop)

**Modified files:**
- `models/persistence.py` (save/load HiFi-GAN state in .distill)
- `training/runner.py` (add VocoderTrainingRunner or extend)
- `ui/tabs/train_tab.py` ("Train Vocoder" section)
- `cli/train.py` (train-vocoder subcommand)

**Why fifth:** This is the most complex phase. It introduces GAN training (adversarial loss, discriminator, dual optimizer), which is fundamentally different from the existing VAE training. The BigVGAN universal vocoder (Phase 1) provides good quality without this. HiFi-GAN training is an optimization for users who want the best possible quality.

**Dependencies:** Phases 1-4 (vocoder infrastructure, persistence, UI).

**Risk:** MEDIUM-HIGH. GAN training is finicky (mode collapse, instability). Small datasets may not train well.

### Phase 6: Griffin-Lim Removal

**Goal:** Remove Griffin-Lim as a generation path, keep only neural vocoders.

**Modified files:**
- `audio/spectrogram.py` (remove InverseMelScale, GriffinLim imports and usage)
- Tests and references

**Why last:** Only remove Griffin-Lim after neural vocoders are proven stable on all hardware (CUDA, MPS, CPU).

**Dependencies:** All previous phases validated.

**Risk:** LOW if previous phases work. But removing fallback means vocoder failures are fatal.

**Recommendation:** Consider keeping Griffin-Lim as an emergency fallback even after Phase 6, gated behind a hidden debug flag. If BigVGAN download fails or MPS has issues, Griffin-Lim still works.

## Patterns to Follow

### Pattern 1: Lazy Vocoder Loading

```python
# Match existing project pattern: lazy imports, initialize on first use
class BigVGANVocoder(Vocoder):
    def __init__(self, device, cache_dir=None):
        self._model = None  # Lazy
        self._device = device
        self._cache_dir = cache_dir

    def _ensure_loaded(self):
        if self._model is None:
            import bigvgan  # Lazy import
            self._model = bigvgan.BigVGAN.from_pretrained(...)
            self._model.remove_weight_norm()
            self._model.eval().to(self._device)

    def mel_to_waveform(self, mel_log):
        self._ensure_loaded()
        ...
```

### Pattern 2: Vocoder-Aware Chunk Processing

The existing pipeline builds a continuous mel via overlap-add, then converts the entire mel to audio in one pass. This is ideal for neural vocoders -- they handle long sequences better than short ones (more context for phase estimation).

```python
# GOOD: Single vocoder call on full mel (existing pattern works perfectly)
combined_mel = synthesize_continuous_mel(model, spectrogram, trajectory)
wav = vocoder.mel_to_waveform(combined_mel)  # One call on full mel

# BAD: Per-chunk vocoder calls with waveform crossfade
for chunk in chunks:
    mel = model.decode(z)
    wav = vocoder.mel_to_waveform(mel)  # Separate call per chunk
    # Crossfade waveform chunks -- introduces artifacts at boundaries
```

### Pattern 3: HiFi-GAN Training Checkpoint Independence

```python
# HiFi-GAN checkpoints are separate from VAE checkpoints
# Saved in training output dir, then bundled into .distill on completion
data/
    training_output/
        checkpoints/           # VAE checkpoints (existing)
        vocoder_checkpoints/   # HiFi-GAN checkpoints (new)
            hifigan_step_050000.pt
            hifigan_step_100000.pt
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Joint VAE + HiFi-GAN Training

**What:** Training the VAE and HiFi-GAN simultaneously in the same loop.

**Why bad:** The VAE mel output changes as the VAE trains, making the HiFi-GAN chase a moving target. This leads to instability and poor convergence.

**Instead:** Train VAE first, freeze it, then train HiFi-GAN on the frozen VAE's mel outputs.

### Anti-Pattern 2: Bundling BigVGAN Weights in Every .distill File

**What:** Saving 500MB of BigVGAN weights inside each model file.

**Why bad:** A user with 10 models would use 5GB just for identical copies of BigVGAN. Save/load becomes slow. Download bandwidth wasted.

**Instead:** Shared cache directory, downloaded once. Only per-model HiFi-GAN weights go in .distill files.

### Anti-Pattern 3: Computing Vocoder-Format Mel in the VAE Pipeline

**What:** Changing `waveform_to_mel()` to produce BigVGAN-format mel throughout the pipeline.

**Why bad:** Breaks all existing trained models. Changes the VAE's training target. Invalidates the latent space analysis.

**Instead:** Adapter converts at the vocoder boundary only. VAE pipeline remains unchanged.

### Anti-Pattern 4: Ignoring Sample Rate Mismatch

**What:** Feeding 48kHz-parameterized mel to a 44.1kHz BigVGAN model without adjustment.

**Why bad:** The mel-to-waveform upsampling ratio is baked into the model. BigVGAN's 512x upsampling at 44.1kHz means each mel frame = 512/44100 seconds. At 48kHz, each mel frame = 512/48000 seconds. The waveform will be slightly time-stretched/compressed.

**Instead:** Accept BigVGAN output at 44.1kHz and resample to 48kHz. The existing `_get_resampler` handles this cleanly.

## Hardware Considerations

### MPS (Apple Silicon)

BigVGAN inference should work on MPS with `use_cuda_kernel=False`. The Snake activation function and anti-aliased upsampling use standard PyTorch ops. However, BigVGAN's 122M parameters may stress MPS memory on smaller MacBooks (8GB unified memory). Need to test:
- Does BigVGAN inference run on MPS without errors?
- Memory usage during inference
- Fallback to CPU if MPS fails

### CUDA

Full support expected. BigVGAN's CUDA kernels provide 1.5-3x speedup. HiFi-GAN training requires CUDA for practical training times.

### CPU Fallback

Both BigVGAN and HiFi-GAN inference work on CPU. Slower (~5-10x realtime instead of 50-150x) but functional. HiFi-GAN training on CPU is impractical (days instead of hours).

## Sources

### High Confidence (Official Documentation)

- [BigVGAN GitHub Repository](https://github.com/NVIDIA/BigVGAN) -- Official NVIDIA implementation, model architecture, mel processing code
- [BigVGAN-v2 44kHz 128band 512x HuggingFace](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x) -- Pretrained weights, config.json, 122M params
- [BigVGAN-v2 44kHz 128band 256x HuggingFace](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_256x) -- Alternative variant, 112M params
- [HiFi-GAN GitHub Repository](https://github.com/jik876/hifi-gan) -- Original HiFi-GAN implementation, config files, training pipeline
- [NVIDIA BigVGAN Technical Blog](https://developer.nvidia.com/blog/achieving-state-of-the-art-zero-shot-waveform-audio-generation-across-audio-types/) -- BigVGAN-v2 capabilities and training details
- [BigVGAN meldataset.py](https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py) -- Mel spectrogram computation with `dynamic_range_compression_torch`

### Medium Confidence (Verified Cross-References)

- [torchaudio HiFiGAN Vocoder](https://docs.pytorch.org/audio/stable/generated/torchaudio.prototype.pipelines.HiFiGANVocoderBundle.html) -- torchaudio's HiFi-GAN (deprecated in 2.8, removed in 2.9 -- do NOT use)
- [HiFi-GAN Paper](https://arxiv.org/abs/2010.05646) -- Architecture details, discriminator design, loss functions
- [BigVGAN Paper](https://arxiv.org/abs/2206.04658) -- Anti-aliased multi-periodicity composition, Snake activation

### Low Confidence (Needs Validation)

- MPS compatibility for BigVGAN inference -- inferred from PyTorch MPS support, not tested with BigVGAN specifically
- HiFi-GAN training time on small datasets -- extrapolated from general HiFi-GAN training, not validated for 5-500 file datasets
- BigVGAN model file size estimate (~500MB) -- based on parameter count, not verified download size

---

*Architecture research for: Neural Vocoder Integration (v1.1 Milestone)*
*Researched: 2026-02-21*
