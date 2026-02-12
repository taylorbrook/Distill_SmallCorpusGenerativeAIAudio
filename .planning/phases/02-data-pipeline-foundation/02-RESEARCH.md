# Phase 2: Data Pipeline Foundation - Research

**Researched:** 2026-02-12
**Domain:** Audio I/O, validation, preprocessing, augmentation (Python/PyTorch)
**Confidence:** HIGH

## Summary

Phase 2 builds the data pipeline: importing audio files (WAV, AIFF, MP3, FLAC), validating dataset integrity, computing summaries with waveform thumbnails, and applying data augmentation (pitch shift, time stretch, noise injection, loudness variation). The project already has torchaudio 2.10.0 installed, which provides all needed transforms for augmentation, but its I/O layer now delegates to TorchCodec (a separate package requiring FFmpeg). This creates a dependency chain that needs careful management, especially on macOS where TorchCodec has known FFmpeg discoverability issues with Homebrew.

The recommended approach is to use **soundfile** (backed by libsndfile, bundled in the pip package) as the primary audio I/O library, converting loaded numpy arrays to torch tensors for the augmentation pipeline. This sidesteps the TorchCodec/FFmpeg dependency entirely for I/O while keeping all torchaudio.transforms for augmentation (PitchShift, Vol, SpeedPerturbation, AddNoise, Resample -- all verified working in the current environment without TorchCodec). For waveform thumbnails, use matplotlib with the Agg backend (no GUI needed) for simplicity and quality, adding it as a dependency.

**Primary recommendation:** Use soundfile for audio I/O (avoids FFmpeg dependency), torchaudio.transforms for augmentation (already installed, verified working), and matplotlib/Agg for waveform thumbnails. Build the data pipeline as a pure backend module with no UI -- Gradio integration comes in Phase 8.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| soundfile | >=0.13.0 | Audio file I/O (read, write, metadata) | libsndfile bundled in pip, no FFmpeg needed, supports WAV/AIFF/FLAC/MP3, returns numpy arrays easily converted to tensors |
| torchaudio | 2.10.0 (already installed) | Audio transforms (augmentation, resampling) | PitchShift, Vol, SpeedPerturbation, AddNoise, Resample all verified working; no TorchCodec needed for transforms |
| torch | 2.10.0 (already installed) | Tensor operations, GPU acceleration | Core framework, already a dependency |
| numpy | >=1.26 | Array operations for I/O bridge | soundfile returns numpy arrays; needed to bridge to torch tensors |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | >=3.9 | Waveform thumbnail generation (Agg backend) | Dataset summary view: generating small PNG waveform images for each audio file |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| soundfile | torchcodec (via torchaudio.load) | TorchCodec requires FFmpeg system dependency; Homebrew FFmpeg has known path issues on macOS (github.com/meta-pytorch/torchcodec/issues/570); adds complexity to setup |
| soundfile | librosa | librosa is heavy (pulls scipy, numba, etc.); soundfile is its backend anyway |
| matplotlib | pillow + numpy (manual drawing) | More code, less polished waveforms; matplotlib gives axes/labels/anti-aliasing for free |

### Why NOT TorchCodec for I/O

As of torchaudio 2.10.0, `torchaudio.load()` and `torchaudio.save()` are aliases to `load_with_torchcodec()` / `save_with_torchcodec()`. They **require** the `torchcodec` package (verified: `ImportError: TorchCodec is required for load_with_torchcodec` in current env). TorchCodec in turn requires FFmpeg (versions 4-8) as a system dependency. On macOS with Homebrew-installed FFmpeg, there are known `@rpath` resolution failures. While workarounds exist (`DYLD_FALLBACK_LIBRARY_PATH`), this adds user-facing complexity. Using soundfile avoids this entirely since libsndfile is bundled in the pip wheel.

The old `torchaudio.info()` function has been **removed** in 2.10 (verified: `hasattr(torchaudio, 'info')` returns `False`). Metadata must come from either TorchCodec's `AudioDecoder.metadata` or soundfile's `sf.info()`.

**Installation:**
```bash
uv add "soundfile>=0.13.0" "numpy>=1.26" "matplotlib>=3.9"
```

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── audio/
│   ├── __init__.py          # Public API re-exports
│   ├── io.py                # Audio file loading, saving, metadata
│   ├── validation.py        # Dataset integrity checks (corrupt files, sample rates, min count)
│   ├── augmentation.py      # Data augmentation pipeline (pitch, speed, noise, volume)
│   ├── preprocessing.py     # Resampling, normalization, chunking for training
│   └── thumbnails.py        # Waveform thumbnail generation
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Dataset class wrapping a collection of audio files
│   └── summary.py           # Dataset summary computation (stats, thumbnails)
```

### Pattern 1: Audio I/O Abstraction Layer
**What:** All file I/O goes through `audio.io` module, which wraps soundfile and returns a consistent AudioFile dataclass.
**When to use:** Every time audio is loaded or metadata is read.
**Why:** Isolates the I/O library choice. If we later switch to TorchCodec (once FFmpeg issues are resolved), only this module changes.

```python
# Source: soundfile docs + project conventions
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import soundfile as sf


@dataclass
class AudioMetadata:
    """Metadata for an audio file, extracted without loading full waveform."""
    path: Path
    sample_rate: int
    num_channels: int
    num_frames: int
    duration_seconds: float
    format: str           # e.g. "WAV", "FLAC", "MP3", "AIFF"
    subtype: str          # e.g. "PCM_24", "PCM_16", "FLOAT"


@dataclass
class AudioFile:
    """Loaded audio file with waveform tensor and metadata."""
    waveform: torch.Tensor    # Shape: [channels, samples], float32, normalized [-1, 1]
    sample_rate: int
    metadata: AudioMetadata


def get_metadata(path: Path) -> AudioMetadata:
    """Read audio file metadata without loading waveform data."""
    info = sf.info(str(path))
    return AudioMetadata(
        path=path,
        sample_rate=info.samplerate,
        num_channels=info.channels,
        num_frames=info.frames,
        duration_seconds=info.duration,
        format=info.format,
        subtype=info.subtype,
    )


def load_audio(path: Path, target_sample_rate: int = 48000) -> AudioFile:
    """Load audio file and return as AudioFile with torch tensor.

    Always returns float32 tensor in [channels, samples] format.
    Resamples to target_sample_rate if source differs.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns [samples, channels], we need [channels, samples]
    waveform = torch.from_numpy(data.T)

    # Resample if needed
    if sr != target_sample_rate:
        from torchaudio.transforms import Resample
        resampler = Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    metadata = get_metadata(path)
    return AudioFile(waveform=waveform, sample_rate=target_sample_rate, metadata=metadata)
```

### Pattern 2: Validation as Error Collection
**What:** Validation functions return lists of typed diagnostic objects (errors vs warnings), never raise exceptions.
**When to use:** Dataset integrity checks before training.
**Why:** Matches Phase 1 pattern from `validation/environment.py` -- collect all issues, report at once.

```python
# Follows Phase 1 pattern: validate_environment() -> (errors, warnings)
from dataclasses import dataclass
from pathlib import Path
from enum import Enum


class Severity(Enum):
    ERROR = "error"      # Cannot proceed (corrupt file)
    WARNING = "warning"  # Can proceed with caution (sample rate mismatch)


@dataclass
class ValidationIssue:
    severity: Severity
    file_path: Path | None
    message: str


def validate_dataset(files: list[Path], min_file_count: int = 5) -> list[ValidationIssue]:
    """Validate a collection of audio files for training readiness.

    Checks:
    - Minimum file count
    - File format support (WAV, AIFF, MP3, FLAC)
    - File readability (corrupt file detection)
    - Sample rate consistency across dataset
    """
    issues: list[ValidationIssue] = []
    # ... collect issues ...
    return issues
```

### Pattern 3: Augmentation Pipeline as Composable Transforms
**What:** Each augmentation is a callable that takes (waveform, sample_rate) and returns augmented (waveform, sample_rate). A pipeline composes them.
**When to use:** Before training, to expand small datasets.
**Why:** Composable, testable, configurable per-augmentation probability.

```python
# Using verified torchaudio.transforms
import torchaudio.transforms as T
import torch
import random


class AugmentationPipeline:
    """Applies random augmentations to expand training data."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.pitch_shift_range = (-2, 2)     # semitones
        self.speed_factors = [0.9, 0.95, 1.0, 1.0, 1.05, 1.1]
        self.noise_snr_range = (15, 40)       # dB
        self.volume_range = (0.7, 1.3)        # amplitude multiplier

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations. Each has independent probability."""
        if random.random() < 0.5:
            n_steps = random.uniform(*self.pitch_shift_range)
            waveform = T.PitchShift(self.sample_rate, n_steps)(waveform)

        if random.random() < 0.5:
            sp = T.SpeedPerturbation(self.sample_rate, self.speed_factors)
            waveform, _ = sp(waveform)

        if random.random() < 0.3:
            noise = torch.randn_like(waveform)
            snr = torch.tensor([random.uniform(*self.noise_snr_range)])
            waveform = T.AddNoise()(waveform, noise, snr)

        if random.random() < 0.5:
            gain = random.uniform(*self.volume_range)
            waveform = T.Vol(gain=gain, gain_type="amplitude")(waveform)

        return waveform
```

### Pattern 4: Lazy Import for Heavy Dependencies
**What:** Import torch, torchaudio, matplotlib inside function bodies, not at module top level.
**When to use:** In all audio modules.
**Why:** Matches Phase 1 pattern (see `hardware/device.py`, `validation/environment.py`). Allows config/validation code to run without heavy imports. Enables better error messages when dependencies are missing.

### Anti-Patterns to Avoid
- **Loading entire dataset into memory at once:** Audio files can be large. Use lazy loading -- load metadata for all files, load waveforms on demand or in batches.
- **Resampling during augmentation instead of preprocessing:** Resample once to 48kHz during import, not repeatedly during training. Store resampled versions.
- **Silently skipping corrupt files:** Always report corrupt files to the user. Never hide data quality issues.
- **Coupling I/O to a specific library:** Wrap soundfile in an abstraction layer (audio.io module). The torchaudio ecosystem is in active transition (torchcodec); abstraction protects against API churn.
- **Using torchaudio.load directly:** In torchaudio 2.10, this requires torchcodec + FFmpeg. Use soundfile instead.
- **Creating augmentation transforms per-call:** `torchaudio.transforms.PitchShift` and `Resample` have internal state (cached kernels). Create once, reuse.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio file decoding | Custom WAV/MP3 parsers | `soundfile.read()` | Handles dozens of formats, edge cases in headers, endianness, encoding subtypes |
| Sample rate conversion | Naive interpolation | `torchaudio.transforms.Resample` | Uses sinc interpolation with configurable filter; naive resampling creates aliasing artifacts |
| Pitch shifting | Manual STFT + phase vocoder | `torchaudio.transforms.PitchShift` | Phase vocoder is tricky; torchaudio handles windowing, overlap-add, phase coherence |
| Time stretching | Manual speed change + resample | `torchaudio.transforms.SpeedPerturbation` | Handles resampling internally, returns adjusted lengths |
| SNR-calibrated noise injection | Manual noise scaling | `torchaudio.transforms.AddNoise` | Correctly computes per-channel energy ratios for target SNR |
| Audio metadata extraction | Parsing WAV/AIFF headers manually | `soundfile.info()` | Handles all format variations, corrupt headers, unusual encodings |
| Loudness measurement | Manual RMS computation | `torchaudio.transforms.Loudness` | Implements ITU-R BS.1770-4 standard |

**Key insight:** Audio file formats are deceptively complex. WAV alone has dozens of encoding subtypes (PCM_16, PCM_24, PCM_32, FLOAT, DOUBLE, etc.), optional chunks, and vendor-specific extensions. MP3 has variable bitrate, ID3 tags, and Xing/LAME headers. Let soundfile/libsndfile handle this.

## Common Pitfalls

### Pitfall 1: torchaudio.load Requires TorchCodec in 2.10
**What goes wrong:** Code calls `torchaudio.load()` and gets `ImportError: TorchCodec is required for load_with_torchcodec`.
**Why it happens:** torchaudio 2.10 removed all legacy backends (sox, soundfile, ffmpeg). `load()` and `save()` are now aliases to `load_with_torchcodec()` / `save_with_torchcodec()`. TorchCodec is NOT a dependency of torchaudio -- it must be installed separately, and it requires FFmpeg as a system dependency.
**How to avoid:** Use soundfile for all audio I/O. Use torchaudio only for transforms (which work without TorchCodec).
**Warning signs:** `ImportError` on first audio load attempt.

### Pitfall 2: soundfile Returns [samples, channels], torchaudio Expects [channels, samples]
**What goes wrong:** Augmentation produces garbled audio or shape mismatch errors.
**Why it happens:** `sf.read()` returns numpy array with shape `(num_samples, num_channels)`. All torchaudio transforms expect shape `(num_channels, num_samples)` (channels-first).
**How to avoid:** Always transpose after loading: `torch.from_numpy(data.T)`. Use `always_2d=True` in sf.read() so mono files also return 2D arrays.
**Warning signs:** Waveform tensor has unexpected shape; first dimension is very large.

### Pitfall 3: Sample Rate Mismatch Across Dataset
**What goes wrong:** Model trains on mixed sample rates, producing artifacts or failing to converge.
**Why it happens:** Users import files from different sources (44.1kHz CDs, 48kHz DAW exports, 96kHz field recordings). Without validation, the pipeline treats all as the same rate.
**How to avoid:** (1) Detect and report mismatches during validation. (2) Resample everything to 48kHz (project baseline) during import. (3) Store original sample rate in metadata for reference.
**Warning signs:** Validation reports inconsistent sample rates; audio plays at wrong pitch.

### Pitfall 4: Corrupt Files Crash the Pipeline
**What goes wrong:** One corrupt file in a batch causes the entire import or training to fail.
**Why it happens:** `sf.read()` raises `RuntimeError` for truncated/corrupt files. If not caught per-file, the whole batch fails.
**How to avoid:** Wrap each file load in try/except. Collect errors per-file. Report all corrupt files at once, then continue with valid files.
**Warning signs:** RuntimeError during batch import; user sees crash instead of diagnostic.

### Pitfall 5: Augmentation Probability Too Aggressive
**What goes wrong:** Every training sample is heavily augmented, model never learns the original data distribution.
**Why it happens:** All augmentations applied to every sample at maximum intensity.
**How to avoid:** (1) Apply each augmentation with independent probability (0.3-0.5). (2) Keep original (unaugmented) copies in the dataset. (3) Use moderate parameter ranges (pitch: +/-2 semitones, speed: 0.9-1.1x, SNR: 15-40 dB, volume: 0.7-1.3x).
**Warning signs:** Model generates only heavily processed-sounding audio; validation loss does not decrease.

### Pitfall 6: PitchShift n_fft Too Small for Low Frequencies at 48kHz
**What goes wrong:** Pitch shifting introduces artifacts in bass-heavy content.
**Why it happens:** Default `n_fft=512` gives frequency resolution of ~94 Hz at 48kHz (48000/512). For 48kHz audio, this is coarse for low-frequency content.
**How to avoid:** Use `n_fft=2048` or `n_fft=4096` for 48kHz audio. This gives ~23 Hz or ~12 Hz resolution respectively.
**Warning signs:** Audible artifacts in augmented audio, especially in bass regions.

### Pitfall 7: matplotlib Import Without Agg Backend on Headless Server
**What goes wrong:** `import matplotlib.pyplot` tries to connect to a display, fails with `_tkinter.TclError: no display name`.
**Why it happens:** matplotlib defaults to an interactive backend. On headless servers or in background threads, there is no display.
**How to avoid:** Always call `matplotlib.use('Agg')` BEFORE importing pyplot. Or set env var `MPLBACKEND=Agg`.
**Warning signs:** TclError on server; works fine on dev machine with GUI.

## Code Examples

Verified patterns from official sources and live environment testing:

### Loading Audio with soundfile
```python
# Source: python-soundfile docs (python-soundfile.readthedocs.io)
import soundfile as sf
import torch
from pathlib import Path


def load_audio_as_tensor(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio file, return (waveform_tensor, sample_rate).

    Returns float32 tensor with shape [channels, samples].
    """
    # always_2d=True ensures mono returns shape (N, 1) not (N,)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Transpose: soundfile gives [samples, channels] -> [channels, samples]
    waveform = torch.from_numpy(data.T)
    return waveform, sr
```

### Getting Audio Metadata Without Loading Waveform
```python
# Source: python-soundfile docs
import soundfile as sf

info = sf.info("audio.wav")
print(f"Sample rate: {info.samplerate}")
print(f"Channels: {info.channels}")
print(f"Frames: {info.frames}")
print(f"Duration: {info.duration:.2f}s")
print(f"Format: {info.format}")      # e.g. "WAV"
print(f"Subtype: {info.subtype}")     # e.g. "PCM_24"
```

### Resampling to 48kHz
```python
# Source: torchaudio transforms docs (docs.pytorch.org/audio/stable/transforms.html)
# Verified working in current environment (torchaudio 2.10.0, no torchcodec needed)
import torchaudio.transforms as T

resampler = T.Resample(orig_freq=44100, new_freq=48000)
waveform_48k = resampler(waveform_44k)  # [channels, samples] -> [channels, new_samples]
```

### Pitch Shifting (for augmentation)
```python
# Source: torchaudio PitchShift docs
# Verified: T.PitchShift works without torchcodec
import torchaudio.transforms as T

# Use larger n_fft for 48kHz audio to avoid bass artifacts
pitch_shift = T.PitchShift(
    sample_rate=48000,
    n_steps=2,          # shift up 2 semitones
    n_fft=2048,         # better frequency resolution for 48kHz
)
shifted = pitch_shift(waveform)
```

### Speed Perturbation (time stretch without pitch change)
```python
# Source: torchaudio SpeedPerturbation docs
# Verified working in current environment
import torchaudio.transforms as T

speed_perturb = T.SpeedPerturbation(
    orig_freq=48000,
    factors=[0.9, 0.95, 1.0, 1.0, 1.05, 1.1],  # weighted toward 1.0
)
perturbed_waveform, lengths = speed_perturb(waveform)
```

### Volume/Loudness Variation
```python
# Source: torchaudio Vol docs
# Verified working in current environment
import torchaudio.transforms as T

# Reduce volume by 6dB
quieter = T.Vol(gain=-6.0, gain_type="db")(waveform)

# Increase volume by amplitude factor
louder = T.Vol(gain=1.3, gain_type="amplitude")(waveform)
```

### Noise Injection with SNR Control
```python
# Source: torchaudio AddNoise docs
# Verified working in current environment
import torchaudio.transforms as T
import torch

add_noise = T.AddNoise()
noise = torch.randn_like(waveform)          # Gaussian noise
snr_db = torch.tensor([20.0])               # 20 dB signal-to-noise ratio
noisy = add_noise(waveform, noise, snr_db)
```

### Waveform Thumbnail Generation
```python
# Source: matplotlib docs + community practice
import matplotlib
matplotlib.use('Agg')  # MUST be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_waveform_thumbnail(
    waveform: np.ndarray,  # shape: [channels, samples]
    output_path: Path,
    width: int = 800,
    height: int = 120,
    color: str = "#4A90D9",
) -> None:
    """Generate a compact waveform thumbnail PNG.

    Args:
        waveform: Audio data as numpy array [channels, samples].
        output_path: Where to save the PNG.
        width: Image width in pixels.
        height: Image height in pixels.
        color: Waveform color.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Mix to mono for display
    if waveform.ndim > 1:
        mono = waveform.mean(axis=0)
    else:
        mono = waveform

    # Downsample for display (no need for full resolution)
    if len(mono) > width * 2:
        indices = np.linspace(0, len(mono) - 1, width * 2, dtype=int)
        mono = mono[indices]

    time = np.linspace(0, 1, len(mono))
    ax.fill_between(time, mono, -mono, alpha=0.7, color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
```

### Corrupt File Detection
```python
# Source: soundfile error handling
import soundfile as sf
from pathlib import Path


def check_file_integrity(path: Path) -> tuple[bool, str]:
    """Check if an audio file can be read without errors.

    Returns (is_valid, message).
    """
    try:
        info = sf.info(str(path))
        if info.frames == 0:
            return False, f"Empty audio file (0 frames): {path.name}"
        # Try reading a small chunk to verify data integrity
        data, sr = sf.read(str(path), frames=min(info.frames, 1024), dtype="float32")
        if data.size == 0:
            return False, f"No audio data readable: {path.name}"
        return True, "OK"
    except Exception as e:
        return False, f"Cannot read file: {path.name} ({type(e).__name__}: {e})"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torchaudio.load()` with sox/soundfile/ffmpeg backends | `torchaudio.load()` delegates to TorchCodec | torchaudio 2.9 (late 2025) | Must install torchcodec + FFmpeg separately, or use soundfile directly |
| `torchaudio.info()` for metadata | Removed in 2.10 | torchaudio 2.9 | Use `soundfile.info()` or `torchcodec.AudioDecoder.metadata` |
| `torchaudio.backend` module for backend selection | Removed in 2.10 | torchaudio 2.9 | No backend selection; torchcodec is the only option |
| torchaudio transforms in C++ | Some converted to pure Python | torchaudio 2.9-2.10 | Most transforms preserved; `lfilter`, `RNNTLoss`, `forced_align`, `overdrive` kept after community feedback |
| librosa for everything audio | torchaudio.transforms + soundfile for I/O | Ongoing | Avoids heavy librosa dependency (scipy, numba); torchaudio transforms are GPU-acceleratable |

**Deprecated/outdated:**
- `torchaudio.info()`: Removed in 2.10. Use `soundfile.info()`.
- `torchaudio.load()` without torchcodec: No longer works. Either install torchcodec or use soundfile.
- `torchaudio.backend` module: Removed entirely.
- `torchaudio.sox_effects`: Removed. Use torchaudio.transforms or functional equivalents.
- WavAugment library: Relied on torchaudio's sox backend, which no longer exists.

## Open Questions

1. **AIFF support in soundfile**
   - What we know: soundfile/libsndfile lists AIFF as a supported format. The requirement explicitly includes AIFF.
   - What's unclear: Whether all AIFF subtypes (compressed AIFF-C with MACE, IMA ADPCM) are supported.
   - Recommendation: Test with AIFF files during implementation. Fall back to torchcodec (with FFmpeg) for any unsupported AIFF variants. This is an edge case since most professional AIFF files use PCM encoding.

2. **Augmentation intensity calibration for small datasets (5-50 files)**
   - What we know: The project targets datasets as small as 5 files. Augmentation should expand this significantly.
   - What's unclear: Optimal number of augmented copies per original file. Too few: not enough training data. Too many: augmented artifacts dominate.
   - Recommendation: Start with 10x expansion (each original produces 10 augmented variants). Make this configurable. Include the original (unaugmented) in every epoch.

3. **Dataset storage format (raw files vs preprocessed cache)**
   - What we know: Raw audio files are imported by the user. Training needs tensors at 48kHz.
   - What's unclear: Should we preprocess and cache resampled/chunked tensors, or resample on-the-fly?
   - Recommendation: Preprocess and cache. For small datasets (5-500 files), disk space is negligible. Preprocessing once avoids repeated resampling during training. Store as `.pt` files alongside originals.

4. **Waveform thumbnail storage and caching**
   - What we know: Thumbnails are needed for dataset summary display.
   - What's unclear: Where to store generated thumbnails. Regenerate each time or cache?
   - Recommendation: Cache thumbnails in a `.thumbnails/` subdirectory within the dataset directory. Regenerate only when source file is newer than thumbnail.

## Sources

### Primary (HIGH confidence)
- torchaudio 2.10.0 installed and tested in project environment -- transforms API verified working
- soundfile PyPI page and docs (python-soundfile.readthedocs.io) -- I/O API, format support, metadata
- [torchaudio 2.10.0 stable docs](https://docs.pytorch.org/audio/stable/transforms.html) -- transforms list, API signatures
- [torchaudio._torchcodec source](https://docs.pytorch.org/audio/stable/_modules/torchaudio/_torchcodec.html) -- verified load() requires torchcodec
- Live environment testing: all transforms (PitchShift, Vol, SpeedPerturbation, AddNoise, Resample) verified working without torchcodec

### Secondary (MEDIUM confidence)
- [TorchCodec AudioDecoder docs](https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html) -- AudioDecoder API
- [TorchAudio future update (GitHub issue #3902)](https://github.com/pytorch/audio/issues/3902) -- maintenance transition timeline
- [TorchCodec Homebrew FFmpeg issue #570](https://github.com/meta-pytorch/torchcodec/issues/570) -- macOS path resolution bug
- [torchaudio PitchShift docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.PitchShift.html) -- PitchShift API
- [torchaudio AddNoise docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.AddNoise.html) -- AddNoise API
- [torchaudio SpeedPerturbation docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.SpeedPerturbation.html) -- SpeedPerturbation API
- [TorchCodec DeepWiki](https://deepwiki.com/pytorch/torchcodec/3-audio-decoding) -- AudioDecoder full API

### Tertiary (LOW confidence)
- [Audio augmentation best practices (Towards Data Science)](https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52/) -- augmentation intensity recommendations
- Augmentation expansion ratio (10x) -- based on general ML community practice for small datasets, not specifically validated for this use case

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified in live environment; soundfile + torchaudio transforms confirmed working
- Architecture: HIGH -- patterns follow Phase 1 conventions; I/O abstraction is standard practice
- Pitfalls: HIGH -- torchaudio.load/torchcodec dependency chain verified empirically; format handling verified
- Augmentation parameters: MEDIUM -- ranges are community standard but not calibrated for this specific use case
- Augmentation expansion ratio: LOW -- 10x is a starting point; needs empirical tuning

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (stable -- soundfile and torchaudio.transforms are mature)
