# Phase 12: Vocoder Interface & BigVGAN Integration - Research

**Researched:** 2026-02-21
**Domain:** Neural vocoder integration (mel-to-waveform), BigVGAN-v2 universal vocoder
**Confidence:** HIGH

## Summary

Phase 12 replaces Griffin-Lim reconstruction with BigVGAN-v2, a 122M-parameter universal neural vocoder that produces dramatically better waveforms from mel spectrograms. The core technical challenge is not BigVGAN itself (it is well-documented, MIT-licensed, and provides a clean `from_pretrained` API), but the **mel adapter** that bridges two fundamentally different mel spectrogram representations: the project's current torchaudio-based HTK mels with log1p normalization versus BigVGAN's librosa-based Slaney mels with log-clamp normalization.

The vendoring strategy is straightforward (full repository copy into `vendor/bigvgan/`), and BigVGAN's HuggingFace Hub integration provides automatic weight downloading with caching and progress indication. The vocoder interface should be a simple abstract base class supporting BigVGAN now and per-model HiFi-GAN V2 in Phase 16.

**Primary recommendation:** Build the mel adapter as a dedicated, well-tested module that converts the VAE's `log1p(mel_power_htk)` output into BigVGAN's `log(clamp(mel_magnitude_slaney, 1e-5))` input space. This is the highest-risk component and requires the most careful implementation and testing.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Full BigVGAN package copy into `vendor/bigvgan/` (top-level vendor directory, separated from project source)
- All model variants, training code, and configs included -- nothing stripped
- BigVGAN's LICENSE file preserved as-is in the vendored copy
- Fully offline after initial download -- no network connectivity required once weights are cached, app never phones home or checks for updates
- Model: `bigvgan_v2_44khz_128band_512x` (the one best model, per REQUIREMENTS.md)
- No Griffin-Lim fallback needed -- if BigVGAN isn't available, it's an error not a graceful degradation
- No audio quality comparison mechanism -- trust BigVGAN's established quality

### Claude's Discretion
- Version pinning mechanism (commit hash file vs git submodule)
- Weight cache location (user-global vs HuggingFace cache)
- Download UX pattern (blocking with progress vs background)
- Cache management CLI commands (if any)
- Default vocoder activation timing (Phase 12 immediate vs Phase 14 wiring)
- Resampling layer ownership (vocoder returns 48kHz vs pipeline handles it)
- Vocoder interface abstraction depth (enough for BigVGAN + HiFi-GAN)
- Device selection logic (auto-detect best available: CUDA > MPS > CPU)
- Mel adapter implementation details (log1p -> log-clamp conversion)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VOC-01 | BigVGAN-v2 universal vocoder converts mel spectrograms to waveforms as the default reconstruction method | BigVGAN `from_pretrained` API, vocoder interface pattern, mel adapter conversion chain |
| VOC-02 | Mel adapter converts VAE's log1p-normalized mels to BigVGAN's log-clamp format | Critical filterbank + normalization difference documented below; full conversion formula derived |
| VOC-03 | BigVGAN model downloads automatically on first use with progress indication | HuggingFace Hub `hf_hub_download` with built-in progress bars; `local_files_only` for offline mode |
| VOC-04 | Vocoder inference runs on CUDA, MPS (Apple Silicon), and CPU | Snake activations use standard PyTorch ops (sin, pow); MPS supported without CUDA kernel; tested with `use_cuda_kernel=False` |
| VOC-06 | BigVGAN source code vendored with version pinning (not pip-installed) | Full repo copy to `vendor/bigvgan/`; commit hash file for pinning; sys.path manipulation for imports |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| BigVGAN | v2 (commit `68eac6e`) | Neural vocoder (mel -> waveform) | NVIDIA's official universal vocoder; 122M params; MIT license; trained on diverse audio; HuggingFace integration |
| huggingface_hub | >=0.20 | Model weight download + caching | BigVGAN's `from_pretrained` uses `PyTorchModelHubMixin` which depends on this; handles progress bars, caching, offline mode |
| librosa | >=0.10 | Slaney-normalized mel filterbank computation | BigVGAN was trained with `librosa.filters.mel(norm='slaney')` filterbanks; must match exactly for correct inference |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchaudio | >=2.10 (existing) | Resampling (44.1kHz -> 48kHz) | Already in project; `torchaudio.transforms.Resample` for vocoder output resampling |
| rich | >=14.0 (existing) | Progress display during download | Already in project; can wrap HF Hub progress or add custom messaging |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| HuggingFace Hub for downloads | Manual urllib/requests download | HF Hub handles caching, resumable downloads, progress, offline mode -- no reason to hand-roll |
| librosa for mel filterbank | Recompute Slaney filterbank manually | librosa is the authoritative implementation BigVGAN was trained against; manual reimplementation risks subtle bugs |
| Vendored BigVGAN source | pip install from git | User decision: vendoring chosen for reproducibility, offline capability, and version control |

### Installation

```bash
uv add huggingface_hub librosa
```

Note: `librosa` has a dependency on `numba` and `llvmlite`. Check if this causes issues on some platforms. An alternative is to compute the Slaney mel filterbank from the librosa source formula directly (it is ~30 lines of numpy), avoiding the heavy `librosa` dependency. This is a discretion item to evaluate during implementation.

## Architecture Patterns

### Recommended Project Structure

```
src/distill/
  vocoder/
    __init__.py          # Public API: get_vocoder(), VocoderBase
    base.py              # Abstract VocoderBase class
    bigvgan_vocoder.py   # BigVGAN implementation wrapping vendored code
    mel_adapter.py       # log1p(HTK) -> log(clamp(Slaney)) conversion
    weight_manager.py    # Download, cache, and load BigVGAN weights
vendor/
  bigvgan/               # Full vendored copy of NVIDIA/BigVGAN
    bigvgan.py
    meldataset.py
    activations.py
    alias_free_activation/
    env.py
    utils.py
    configs/
    LICENSE
    VENDOR_PIN.txt        # Commit hash for version pinning
```

### Pattern 1: Abstract Vocoder Interface

**What:** A base class defining the vocoder contract, with BigVGAN as the default implementation.
**When to use:** Always -- enables HiFi-GAN V2 in Phase 16 without refactoring.
**Example:**

```python
# src/distill/vocoder/base.py
from abc import ABC, abstractmethod
import torch

class VocoderBase(ABC):
    """Abstract vocoder interface for mel-to-waveform conversion."""

    @abstractmethod
    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Parameters
        ----------
        mel : torch.Tensor
            VAE output mel spectrogram in log1p format.
            Shape: [B, 1, n_mels, time]

        Returns
        -------
        torch.Tensor
            Waveform at vocoder's native sample rate.
            Shape: [B, 1, samples]
        """
        ...

    @abstractmethod
    def sample_rate(self) -> int:
        """Native output sample rate of this vocoder."""
        ...

    @abstractmethod
    def to(self, device: torch.device) -> "VocoderBase":
        """Move vocoder to device."""
        ...
```

### Pattern 2: Mel Adapter as Explicit Conversion Layer

**What:** A dedicated module that converts between mel representations, NOT hidden inside the vocoder.
**When to use:** At the boundary between VAE output and vocoder input.
**Why explicit:** The mel conversion is the highest-risk component. Making it a first-class module with its own tests, rather than burying it in vocoder code, ensures correctness is verifiable.

```python
# src/distill/vocoder/mel_adapter.py
import torch

class MelAdapter:
    """Convert VAE log1p-HTK mels to BigVGAN log-clamp-Slaney mels.

    The VAE produces: log1p(power_mel_htk)
    BigVGAN expects: log(clamp(magnitude_mel_slaney, min=1e-5))

    Conversion chain:
    1. Undo log1p: mel_power_htk = expm1(mel_log1p)
    2. Power to magnitude: mel_mag_htk = sqrt(mel_power_htk)
    3. Re-apply filterbank: mel_mag_slaney = slaney_fb @ stft_mag
       (requires regenerating from STFT, or using adapter matrix)
    4. Apply BigVGAN normalization: log(clamp(mel_mag_slaney, 1e-5))
    """
```

### Pattern 3: Weight Manager with HuggingFace Hub

**What:** A module that handles first-use download, caching, and offline loading.
**When to use:** Before vocoder instantiation.

```python
# src/distill/vocoder/weight_manager.py
from huggingface_hub import hf_hub_download
from pathlib import Path

BIGVGAN_REPO_ID = "nvidia/bigvgan_v2_44khz_128band_512x"
BIGVGAN_GENERATOR_FILE = "bigvgan_generator.pt"
BIGVGAN_CONFIG_FILE = "config.json"

def ensure_bigvgan_weights(cache_dir: Path | None = None) -> Path:
    """Download BigVGAN weights if not cached, return local path.

    Uses HuggingFace Hub's built-in caching and progress bars.
    After first download, works fully offline via local_files_only.
    """
    ...
```

### Anti-Patterns to Avoid

- **Modifying vendored BigVGAN code:** Never edit files in `vendor/bigvgan/`. Any adaptations go in `src/distill/vocoder/`. This keeps the vendor clean for updates.
- **Importing BigVGAN directly:** Always go through the `distill.vocoder` wrapper. The vendored code uses relative imports (`from env import AttrDict`) that need sys.path setup.
- **Skipping mel adapter and feeding VAE output directly to BigVGAN:** The filterbank mismatch (HTK vs Slaney) and normalization mismatch (log1p vs log-clamp) will produce muffled, distorted, or silent output. This is the #1 integration risk.
- **Using `use_cuda_kernel=True`:** The CUDA kernel requires `nvcc` + `ninja` at runtime and breaks MPS/CPU. Always use `use_cuda_kernel=False` for cross-platform compatibility.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model weight download + caching | Custom HTTP download with progress | `huggingface_hub.hf_hub_download` | Handles resumable downloads, version-aware caching, offline mode, progress bars, auth tokens |
| Mel filterbank (Slaney) | Manual triangular filter computation | `librosa.filters.mel(norm='slaney')` (or extract its formula) | BigVGAN was trained against librosa's exact output; subtle differences produce bad audio |
| Neural vocoder architecture | Custom GAN vocoder | Vendored BigVGAN | 122M params trained on massive diverse dataset; impossible to replicate |
| STFT computation for mel | Custom FFT + windowing | `torch.stft` (matching BigVGAN's meldataset.py) | Must match BigVGAN's exact STFT parameters for correct inference |

**Key insight:** Every component in the mel-to-waveform chain must exactly match BigVGAN's training configuration. Any mismatch in filterbank, normalization, STFT windowing, or frequency range will degrade output quality. Use BigVGAN's own code (vendored) for mel computation where possible.

## Common Pitfalls

### Pitfall 1: Mel Filterbank Mismatch (CRITICAL)

**What goes wrong:** Audio output is muffled, tinny, or has frequency response artifacts.
**Why it happens:** The project's current mel spectrograms use `torchaudio.transforms.MelSpectrogram` which defaults to `mel_scale='htk'` and `norm=None`. BigVGAN was trained with `librosa.filters.mel` which defaults to `htk=False` (Slaney scale) and `norm='slaney'` (area normalization). These produce **different filterbank matrices** -- different center frequencies, different bandwidths, and different normalization.
**How to avoid:** The mel adapter must recompute the mel spectrogram using BigVGAN's exact parameters, NOT simply transform the values. The safest approach is to run BigVGAN's own `mel_spectrogram()` function from the vendored `meldataset.py` on reconstructed audio.
**Warning signs:** Output sounds like it has a frequency-dependent volume envelope, or is missing high/low frequencies, or has "underwater" quality.

### Pitfall 2: Power vs Magnitude Spectrogram Mismatch

**What goes wrong:** Audio is too quiet, too loud, or has wrong dynamic range.
**Why it happens:** The project's `AudioSpectrogram` uses `power=2.0` (power spectrogram) and applies `log1p`. BigVGAN's `mel_spectrogram()` computes magnitude (sqrt of sum of squares, equivalent to power=1.0) and applies `log(clamp(x, min=1e-5))`.
**How to avoid:** The mel adapter conversion must account for: (a) power -> magnitude (square root), and (b) different log normalizations (log1p vs log-clamp).
**Warning signs:** Audio dynamic range is wrong -- either everything is nearly silent or clipping.

### Pitfall 3: STFT Parameter Mismatch

**What goes wrong:** Temporal artifacts, wrong frame alignment, clicks.
**Why it happens:** The project uses `n_fft=2048, hop_length=512, sample_rate=48000`. BigVGAN 44kHz model uses `n_fft=2048, hop_size=512, win_size=2048, sampling_rate=44100`. The sample rates differ (48kHz vs 44.1kHz), meaning the same STFT params cover different frequency ranges and time spans.
**How to avoid:** Do NOT try to adapt existing 48kHz mel spectrograms frame-by-frame. Instead, reconstruct a waveform approximation from the VAE's mel, resample to 44.1kHz, then compute BigVGAN-compatible mels from scratch using BigVGAN's own `mel_spectrogram()`.
**Warning signs:** Clicks between frames, pitch shift, temporal smearing.

### Pitfall 4: MPS Compatibility for Snake Activations

**What goes wrong:** Runtime error or `NotImplementedError` on Apple Silicon.
**Why it happens:** BigVGAN uses `SnakeBeta` activations (`x + 1/beta * sin^2(x * alpha)`) with `alpha_logscale=True` (requiring `torch.exp`). These use standard PyTorch ops (sin, pow, exp) that are generally MPS-compatible as of PyTorch 2.x, but edge cases may exist.
**How to avoid:** Test on MPS before declaring support. Keep `PYTORCH_ENABLE_MPS_FALLBACK=1` as a known workaround. The Snake activation implementation in `activations.py` uses only standard ops (no custom CUDA kernels when `use_cuda_kernel=False`).
**Warning signs:** `NotImplementedError` mentioning a specific operation, NaN outputs on MPS.

### Pitfall 5: Vendored Import Path Conflicts

**What goes wrong:** `ImportError` or wrong module loaded.
**Why it happens:** BigVGAN's code uses relative imports like `from env import AttrDict`, `from activations import SnakeBeta`. These won't work from `vendor/bigvgan/` without `sys.path` manipulation, and may conflict with project modules.
**How to avoid:** Add `vendor/bigvgan/` to `sys.path` temporarily during BigVGAN module loading, or use `importlib` to load the vendored modules explicitly. Wrap in a clean import utility function.
**Warning signs:** `ModuleNotFoundError: No module named 'env'`, or loading the wrong `utils` module.

### Pitfall 6: Generator Weight File Size (489 MB)

**What goes wrong:** Long first-launch experience, timeout, or disk space issues.
**Why it happens:** `bigvgan_generator.pt` is 489 MB. On slow connections, this takes minutes.
**How to avoid:** Use HuggingFace Hub's resumable download support. Show clear progress indication. Cache in a stable location. Document the download requirement in user-facing text.
**Warning signs:** User thinks app is frozen on first launch.

## Code Examples

### BigVGAN Mel Spectrogram Computation (from vendored meldataset.py)

```python
# Source: NVIDIA/BigVGAN meldataset.py (vendored)
# This is how BigVGAN computes its input mels -- the mel adapter must produce IDENTICAL output

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # Mel filterbank: librosa with Slaney normalization (default norm='slaney')
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(device)

    # Reflective padding
    padding = (n_fft - hop_size) // 2
    y = F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    # STFT -> magnitude spectrogram
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                       window=hann_window, center=False, pad_mode="reflect",
                       normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    # Apply mel filterbank + log compression
    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))  # spectral_normalize_torch

    return mel_spec
```

### BigVGAN Config for 44kHz 128-band Model

```json
// Source: https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x/config.json
{
    "sampling_rate": 44100,
    "num_mels": 128,
    "n_fft": 2048,
    "hop_size": 512,
    "win_size": 2048,
    "fmin": 0,
    "fmax": null,       // null = Nyquist (22050 Hz)
    "segment_size": 65536,
    "upsample_rates": [8, 4, 2, 2, 2, 2],
    "upsample_initial_channel": 1536,
    "activation": "snakebeta",
    "snake_logscale": true,
    "normalize_volume": true
}
```

### BigVGAN Loading and Inference

```python
# Source: NVIDIA/BigVGAN README / HuggingFace model card
import sys
sys.path.insert(0, "vendor/bigvgan")

import bigvgan
from meldataset import get_mel_spectrogram

model = bigvgan.BigVGAN.from_pretrained(
    'nvidia/bigvgan_v2_44khz_128band_512x',
    use_cuda_kernel=False,  # MUST be False for MPS/CPU compatibility
)
model.remove_weight_norm()
model = model.eval().to(device)

# Input: mel spectrogram [B, 128, T] in log-clamp format
# Output: waveform [B, 1, T*512] in [-1, 1] range
with torch.inference_mode():
    wav_gen = model(mel)  # mel shape: [B, C_mel=128, T_frame]
```

### Mel Adapter Conversion Strategy

```python
# The safest mel adapter approach: work through waveform reconstruction

def adapt_vae_mel_to_bigvgan(vae_mel_log1p: torch.Tensor,
                              spectrogram: AudioSpectrogram,
                              bigvgan_config) -> torch.Tensor:
    """Convert VAE mel output to BigVGAN-compatible mel input.

    Strategy: Go through waveform domain to avoid filterbank mismatch.
    1. VAE mel (log1p, HTK, 48kHz) -> approximate waveform via Griffin-Lim
    2. Resample waveform 48kHz -> 44.1kHz
    3. Compute BigVGAN-format mel (log-clamp, Slaney, 44.1kHz)

    This is the safest path but has quality loss from Griffin-Lim.
    """
    # Step 1: Reconstruct approximate waveform from VAE mel
    waveform = spectrogram.mel_to_waveform(vae_mel_log1p)  # Griffin-Lim

    # Step 2: Resample 48kHz -> 44.1kHz
    resampler = torchaudio.transforms.Resample(48000, 44100)
    waveform_44k = resampler(waveform.squeeze(1))

    # Step 3: Compute BigVGAN mel using vendored meldataset
    from meldataset import get_mel_spectrogram
    mel_bigvgan = get_mel_spectrogram(waveform_44k, bigvgan_config)

    return mel_bigvgan

# ALTERNATIVE (better quality): Direct mel-domain conversion
# If filterbank matrices are known, convert via:
#   mel_slaney = slaney_fb @ inv(htk_fb) @ htk_mel
# This avoids waveform-domain round-trip but requires careful linear algebra.
```

### HuggingFace Hub Weight Download

```python
# Source: https://huggingface.co/docs/huggingface_hub/en/guides/download
from huggingface_hub import hf_hub_download
from pathlib import Path

def download_bigvgan_weights() -> tuple[Path, Path]:
    """Download generator weights and config, return local paths."""
    config_path = hf_hub_download(
        repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
        filename="config.json",
    )
    weights_path = hf_hub_download(
        repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
        filename="bigvgan_generator.pt",
    )
    return Path(config_path), Path(weights_path)

# For offline mode (after initial download):
def load_bigvgan_weights_offline() -> tuple[Path, Path]:
    config_path = hf_hub_download(
        repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
        filename="config.json",
        local_files_only=True,  # Never contacts network
    )
    weights_path = hf_hub_download(
        repo_id="nvidia/bigvgan_v2_44khz_128band_512x",
        filename="bigvgan_generator.pt",
        local_files_only=True,
    )
    return Path(config_path), Path(weights_path)
```

## Mel Adapter Deep Dive: The Critical Conversion

This section documents the exact differences between the project's mel representation and BigVGAN's expected input, as this is the highest-risk component.

### Current Project Mel (VAE output)

| Parameter | Value | Source |
|-----------|-------|--------|
| Sample rate | 48,000 Hz | `SpectrogramConfig.sample_rate` |
| n_fft | 2048 | `SpectrogramConfig.n_fft` |
| hop_length | 512 | `SpectrogramConfig.hop_length` |
| n_mels | 128 | `SpectrogramConfig.n_mels` |
| f_min | 0.0 | `SpectrogramConfig.f_min` |
| f_max | None (24,000 Hz Nyquist) | `SpectrogramConfig.f_max` |
| power | 2.0 (power spectrogram) | `SpectrogramConfig.power` |
| Mel scale | HTK | torchaudio default `mel_scale='htk'` |
| Mel norm | None | torchaudio default `norm=None` |
| Log normalization | `log1p(mel)` | `AudioSpectrogram.waveform_to_mel` |
| Output shape | `[B, 1, 128, T]` | With channel dim |

### BigVGAN Expected Mel Input

| Parameter | Value | Source |
|-----------|-------|--------|
| Sample rate | 44,100 Hz | `config.json: sampling_rate` |
| n_fft | 2048 | `config.json: n_fft` |
| hop_size | 512 | `config.json: hop_size` |
| win_size | 2048 | `config.json: win_size` |
| num_mels | 128 | `config.json: num_mels` |
| fmin | 0 | `config.json: fmin` |
| fmax | null (22,050 Hz Nyquist) | `config.json: fmax` |
| Power | 1.0 (magnitude spectrogram) | `torch.sqrt(spec.pow(2).sum(-1) + 1e-9)` |
| Mel scale | Slaney | librosa default `htk=False` |
| Mel norm | Slaney (area normalized) | librosa default `norm='slaney'` |
| Log normalization | `log(clamp(mel, min=1e-5))` | `dynamic_range_compression_torch` |
| Input shape | `[B, 128, T]` | No channel dim |

### Differences Summary

| Aspect | Project (VAE) | BigVGAN | Impact |
|--------|--------------|---------|--------|
| Sample rate | 48,000 Hz | 44,100 Hz | Different Nyquist, different time resolution |
| Power vs magnitude | power=2.0 | sqrt (power=1.0) | Values differ by square root |
| Mel scale | HTK | Slaney | Different center frequencies for mel bands |
| Mel normalization | None | Slaney (area) | Different relative band amplitudes |
| Log normalization | log1p(x) | log(clamp(x, 1e-5)) | Different dynamic range mapping |
| f_max | 24,000 Hz | 22,050 Hz | Different top frequency coverage |
| STFT center | True (torchaudio default) | False | Different frame alignment |

### Recommended Mel Adapter Strategy

**Approach A: Waveform round-trip (simpler, lower quality):**
1. VAE mel -> Griffin-Lim waveform (48kHz) -- uses existing `mel_to_waveform()`
2. Resample 48kHz -> 44.1kHz
3. BigVGAN `mel_spectrogram()` on the 44.1kHz waveform
4. Feed to BigVGAN

Pros: Guaranteed correct mel format. Simple implementation.
Cons: Griffin-Lim introduces artifacts that BigVGAN then reconstructs. Quality ceiling limited by intermediate Griffin-Lim step.

**Approach B: Direct mel-domain conversion (complex, higher quality):**
1. Undo VAE normalization: `expm1(clamp(mel_log1p, min=0))` -> power mel
2. Convert power to magnitude: `sqrt(mel_power)` -> magnitude mel
3. Convert HTK mel to linear spectrum (pseudo-inverse of HTK filterbank)
4. Convert linear spectrum to Slaney mel (multiply by Slaney filterbank)
5. Apply BigVGAN normalization: `log(clamp(mel_slaney, min=1e-5))`

Pros: No waveform round-trip, preserves VAE's learned detail.
Cons: Pseudo-inverse of mel filterbank is lossy (underdetermined). Requires computing both filterbanks. Risk of accumulating numerical errors.

**Approach C: Hybrid (recommended):**
1. For the mel filterbank difference: compute a transfer matrix `T = slaney_fb @ pinv(htk_fb)` offline
2. Apply: `mel_slaney_mag = T @ mel_htk_mag` (fast matrix multiply at inference time)
3. Handle normalization differences with the explicit math:
   - Input: `log1p(mel_power_htk)`
   - Step 1: `mel_power_htk = expm1(input).clamp(min=0)`
   - Step 2: `mel_mag_htk = sqrt(mel_power_htk)`
   - Step 3: `mel_mag_slaney = T @ mel_mag_htk` (precomputed transfer matrix)
   - Step 4: `output = log(clamp(mel_mag_slaney, min=1e-5))`

Pros: Best quality (no waveform round-trip), reasonably simple at inference time. Transfer matrix computed once and cached.
Cons: Transfer matrix is approximate (pseudo-inverse loses information). Requires validation with listening tests.

**Recommendation:** Implement Approach C as the primary path, with Approach A as a validation tool (to compare against). The transfer matrix approach is the standard technique used in vocoder adaptation research.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Griffin-Lim (iterative STFT) | Neural vocoders (HiFi-GAN family) | 2020 (HiFi-GAN paper) | Dramatic quality improvement; near-transparent reconstruction |
| HiFi-GAN V1 (per-dataset) | BigVGAN-v2 (universal) | 2024 (BigVGAN-v2 release) | Single model works on any audio; no per-dataset training needed |
| pip install for vocoders | Vendored source + HuggingFace weights | Current best practice | Reproducibility, version control, offline capability |
| Custom CUDA kernels | Standard PyTorch ops | Ongoing | Cross-platform compatibility (MPS, CPU) at cost of ~2x speed |

**Deprecated/outdated:**
- **Griffin-Lim reconstruction:** Being replaced by this phase. Phase 16 removes it entirely.
- **WaveGlow, WaveRNN:** Superseded by GAN-based vocoders (HiFi-GAN, BigVGAN) which are faster and higher quality.
- **BigVGAN v1:** Superseded by v2 with better architecture (SnakeBeta, anti-aliased activations, CQT discriminator).

## Discretion Recommendations

Based on research, here are recommendations for the Claude's Discretion items:

### Version Pinning: Commit Hash File (not submodule)

**Recommendation:** Use a `VENDOR_PIN.txt` file in `vendor/bigvgan/` containing the commit hash.
**Rationale:** Git submodules add complexity (recursive clone, detached HEAD state, CI complications). A simple text file with the commit hash (`68eac6eeda4c4fde3e666a4f61387a54c79a742e`) is sufficient for a vendored copy that changes rarely. It documents provenance without adding tooling burden.

### Weight Cache: HuggingFace Hub Default Cache

**Recommendation:** Use HuggingFace Hub's default cache directory (`~/.cache/huggingface/hub/`).
**Rationale:** BigVGAN's `from_pretrained` already uses HF Hub internally. Fighting its caching would mean reimplementing download logic. The HF cache is well-tested, supports resumable downloads, handles concurrent access, and users of other HF models already have it. The `local_files_only=True` flag provides the offline guarantee.

### Download UX: Blocking with HF Hub Progress Bar

**Recommendation:** Blocking download with HuggingFace Hub's built-in `tqdm` progress bar.
**Rationale:** The download is a one-time event. Background download adds complexity (async state, "model not ready yet" errors, partial state). Blocking with a clear progress bar is the simplest correct approach. The 489 MB download takes ~1 minute on a typical connection.

### Default Vocoder Activation: Phase 14

**Recommendation:** Phase 12 builds and tests the vocoder infrastructure; Phase 14 wires it into generation paths.
**Rationale:** Phase 12's scope is already substantial (vendoring, interface, mel adapter, weight management, cross-platform testing). Wiring through all generation paths (crossfade, latent interp, preview, reconstruction) is Phase 14's explicit responsibility. Phase 12 should expose the vocoder as a callable module and verify it works in isolation.

### Resampling Ownership: Vocoder Layer Returns 44.1kHz, Pipeline Resamples

**Recommendation:** The vocoder returns audio at its native sample rate (44,100 Hz). The generation pipeline (existing `_get_resampler`) handles 44.1kHz -> 48kHz conversion.
**Rationale:** The generation pipeline already has resampling logic. Making the vocoder responsible for output sample rate would couple it to the pipeline's target rate. Keeping it at native rate is cleaner and matches the VocoderBase interface contract. The pipeline already resamples to `config.sample_rate` at the end.

### Vocoder Interface Depth: Minimal (BigVGAN + HiFi-GAN V2)

**Recommendation:** Abstract base class with `mel_to_waveform()`, `sample_rate`, `to(device)`. No complex registry or plugin system.
**Rationale:** Only two implementations are planned (BigVGAN universal, HiFi-GAN V2 per-model). A factory function `get_vocoder(type="bigvgan")` is sufficient. Over-abstracting for a two-implementation interface adds complexity without value.

### Device Selection: Reuse Existing `select_device()`

**Recommendation:** Use the project's existing `distill.hardware.device.select_device()` for device selection. BigVGAN model is moved to the same device as the VAE.
**Rationale:** The project already has a robust device selection system with auto-detection (CUDA > MPS > CPU), smoke testing, and fallback. No need to create a separate device selection for the vocoder.

## Open Questions

1. **Mel adapter quality via transfer matrix approach**
   - What we know: The transfer matrix `T = slaney_fb @ pinv(htk_fb)` is mathematically sound but the pseudo-inverse is lossy because mel filterbanks are not square.
   - What's unclear: How much quality is lost in practice. Does the conversion produce "good enough" mels for BigVGAN, or do artifacts accumulate?
   - Recommendation: Implement both approaches (transfer matrix and waveform round-trip), generate audio from the same VAE output, and compare by ear. If transfer matrix sounds good, use it. If not, fall back to waveform round-trip.

2. **MPS compatibility with BigVGAN 122M model**
   - What we know: Snake/SnakeBeta activations use standard PyTorch ops (sin, pow, exp). PyTorch 2.10 MPS support is mature. BigVGAN uses Conv1d and ConvTranspose1d which are MPS-supported.
   - What's unclear: Whether the 122M parameter model fits in Apple Silicon unified memory during inference, and whether any edge-case ops fail on MPS.
   - Recommendation: Test early in implementation. Have `PYTORCH_ENABLE_MPS_FALLBACK=1` as a documented workaround.

3. **librosa dependency weight**
   - What we know: `librosa` pulls in `numba`, `llvmlite`, `scipy`, `scikit-learn` (some already in project). The mel filterbank computation itself is ~30 lines of numpy.
   - What's unclear: Whether adding librosa significantly increases install size/time, or conflicts with existing dependencies.
   - Recommendation: First try adding `librosa` as a dependency. If it causes problems, extract the Slaney filterbank computation (it is pure numpy) into a utility function. The STATE.md already notes "librosa (new dep) for Slaney-normalized mel filterbanks" as a project decision.

4. **BigVGAN vendored import mechanism**
   - What we know: BigVGAN uses relative imports (`from env import AttrDict`, `from activations import ...`). The vendored copy will be at `vendor/bigvgan/`.
   - What's unclear: Best practice for importing vendored Python packages with relative imports -- sys.path manipulation, importlib, or package-ifying with `__init__.py`.
   - Recommendation: Add `vendor/bigvgan/` to `sys.path` temporarily during import in a context manager. This is the simplest approach that preserves the vendored code unmodified.

## Sources

### Primary (HIGH confidence)
- [NVIDIA/BigVGAN GitHub Repository](https://github.com/NVIDIA/BigVGAN) - Complete source code, README, license, architecture details
- [BigVGAN v2 44kHz 128-band HuggingFace](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x) - Model card, config.json (extracted verbatim), file sizes, usage examples
- [BigVGAN meldataset.py source](https://github.com/NVIDIA/BigVGAN/blob/main/meldataset.py) - Complete mel spectrogram computation code with librosa filterbank
- [HuggingFace Hub Download Guide](https://huggingface.co/docs/huggingface_hub/en/guides/download) - `hf_hub_download` API, caching, offline mode, progress bars
- [librosa.filters.mel docs](https://librosa.org/doc/main/generated/librosa.filters.mel.html) - Default parameters: `norm='slaney'`, `htk=False`
- [torchaudio MelSpectrogram docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html) - Default parameters: `mel_scale='htk'`, `norm=None`
- Project source code: `src/distill/audio/spectrogram.py`, `src/distill/inference/generation.py`, `src/distill/inference/chunking.py`, `src/distill/models/vae.py`, `src/distill/hardware/device.py`, `src/distill/models/persistence.py`

### Secondary (MEDIUM confidence)
- [MLX BigVGAN port](https://github.com/yrom/mlx-bigvgan) - Confirmed `mel_norm="slaney"` usage in alternative implementation
- [BigVGAN ICLR 2023 paper](https://openreview.net/pdf?id=iTtGCMDEzS_) - Architecture details, training methodology
- [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html) - MPS backend capabilities and limitations

### Tertiary (LOW confidence)
- [PyTorch MPS fallback issue #134416](https://github.com/pytorch/pytorch/issues/134416) - MPS Conv1d issues in some configurations; needs validation for BigVGAN's specific usage pattern

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - BigVGAN is well-documented with clear APIs, HuggingFace integration proven
- Architecture: HIGH - Vocoder interface is a straightforward abstract pattern; vendoring is a file copy
- Mel adapter: MEDIUM - The filterbank conversion is mathematically sound but unvalidated in this specific project context; listening tests needed
- MPS compatibility: MEDIUM - Standard ops should work but 122M model on MPS not specifically tested
- Pitfalls: HIGH - All pitfalls are derived from concrete code analysis of both codebases

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (BigVGAN v2 is stable/final release; no further changes expected)
