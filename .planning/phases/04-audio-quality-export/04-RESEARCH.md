# Phase 4: Audio Quality & Export - Research

**Researched:** 2026-02-12
**Domain:** Audio generation pipeline, anti-aliasing, WAV export, quality metrics
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Generation duration & variations
- Freeform duration input (slider or numeric), not fixed presets
- Initial max duration: 60 seconds (architecture should support longer in the future)
- Chunk duration is configurable (currently 1.0s default), not locked
- For multi-chunk generation (>1 chunk), support BOTH concatenation modes:
  - Crossfade between chunks (reliable default)
  - Latent interpolation for evolving, continuous sound (experimental/smooth option)
- User selects which concatenation mode to use per generation

#### Stereo output behavior
- Default output is mono
- User can opt into stereo per generation
- When stereo is selected, user chooses the method:
  - Simple stereo widening (mid-side / Haas effect)
  - Dual generation (two different seeds for L/R channels)

#### Export defaults & configuration
- Default format: 48kHz / 24-bit WAV (professional production standard)
- One-click export with defaults; advanced settings (sample rate, bit depth, channel mode) available but tucked away
- Sidecar JSON alongside each exported .wav with full generation details (model name, parameters, seed, timestamp)
- No embedded WAV metadata tags (sidecar JSON is the metadata store)

#### Quality feedback
- Spectral analysis (spectrogram / frequency plot) available on demand, not always shown
- Audio preview player supports both waveform and spectrogram views (toggle between them)
- Automatic quality score shown after each generation, based on:
  - Signal-to-noise ratio
  - Clipping detection
- No spectral coverage or flatness metrics (keep it focused)

### Claude's Discretion
- Stereo width control implementation (slider vs presets)
- Export file destination (project output folder vs save-as)
- Exact quality score presentation (numeric, letter grade, visual indicator)
- Number of variations in batch generation (if implemented)
- Anti-aliasing filter implementation details
- Crossfade overlap duration for chunk concatenation

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

## Summary

Phase 4 transforms the existing VAE training previews into a proper audio generation pipeline. The current codebase already has the core mel-to-waveform conversion path (`AudioSpectrogram.mel_to_waveform` using `InverseMelScale` + `GriffinLim`) and basic WAV export (`_save_wav` in `training/preview.py` using soundfile with `PCM_16`). This phase extends that foundation with: (1) a generation engine that handles configurable duration via chunk-based synthesis with two concatenation modes, (2) anti-aliasing via low-pass filtering before final output, (3) stereo processing with mid-side widening and dual-seed generation, (4) professional WAV export with configurable sample rate / bit depth / sidecar JSON, and (5) quality metrics (SNR + clipping detection).

The technical risk is moderate. The mel inversion path (InverseMelScale + GriffinLim) is already proven in Phase 3 previews but has known quality limitations -- GriffinLim is lossy by nature (phase recovery is approximate). For this phase, GriffinLim is sufficient; neural vocoders (HiFi-GAN etc.) are out of scope but the architecture should not preclude them. The chunk concatenation with crossfade is straightforward; latent interpolation using SLERP is well-established but needs careful tuning. Anti-aliasing is standard DSP (low-pass filter at Nyquist before sample rate conversion).

**Primary recommendation:** Build a `GenerationPipeline` class in `src/small_dataset_audio/inference/` that orchestrates chunk generation, concatenation, anti-aliasing, stereo processing, and export. Use the existing `AudioSpectrogram` for mel-to-waveform conversion. Use `soundfile` for all WAV export with configurable subtypes. Use `scipy.signal` for anti-aliasing filters and `torchaudio.transforms.Resample` for sample rate conversion.

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| soundfile | >=0.13.0 | WAV export with PCM_16/PCM_24/FLOAT subtypes | Already in project; libsndfile handles all PCM formats natively; no FFmpeg needed |
| torchaudio | >=2.10.0 | InverseMelScale, GriffinLim, Resample transforms | Already in project; provides mel inversion + resampling with anti-aliasing |
| torch | >=2.10.0 | Tensor operations, model inference | Already in project; VAE model runs on torch |
| numpy | >=1.26 | Audio array manipulation, quality metrics | Already in project; needed for SNR/clipping calculations |
| matplotlib | >=3.9 | Spectrogram visualization, waveform plots | Already in project; used for spectral analysis display |

### Supporting (new dependency needed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | >=1.12 | Anti-aliasing low-pass filter (`scipy.signal.sosfiltfilt`, `butter`) | Before sample rate conversion and in final output stage |

**Note on scipy:** scipy is NOT currently in the project dependencies. It needs to be added. However, scipy is a standard scientific Python package, well-tested, and lightweight compared to alternatives. The alternative would be hand-rolling FIR/IIR filters with numpy, which is error-prone for audio.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| GriffinLim for mel inversion | HiFi-GAN neural vocoder | Much higher quality but requires training a separate vocoder model; out of scope for v1 |
| scipy for anti-aliasing | numpy-only FIR filter | Fewer dependencies but error-prone; scipy's butter + sosfiltfilt is battle-tested |
| torchaudio.Resample for sample rate conversion | scipy.signal.resample_poly | scipy is CPU-only but handles arbitrary ratios well; torchaudio Resample is already in project and handles the key rates (44.1k, 48k, 96k) |
| soundfile for WAV export | wave stdlib module | wave only supports PCM_16; no 24-bit or float support |

**Installation:**
```bash
uv add "scipy>=1.12"
```

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── inference/
│   ├── __init__.py           # Public API exports
│   ├── generation.py         # GenerationPipeline class (core orchestrator)
│   ├── chunking.py           # Chunk generation, crossfade, latent interpolation
│   ├── stereo.py             # Mid-side widening, Haas effect, dual-seed stereo
│   ├── quality.py            # SNR calculation, clipping detection, quality score
│   └── export.py             # WAV export with configurable format + sidecar JSON
├── audio/
│   ├── spectrogram.py        # [existing] mel-to-waveform (InverseMelScale + GriffinLim)
│   ├── filters.py            # [NEW] Anti-aliasing low-pass filter
│   └── ...                   # [existing modules unchanged]
```

### Pattern 1: GenerationPipeline (Orchestrator)
**What:** A single class that takes generation parameters and produces final audio output. Internally delegates to chunk generation, concatenation, stereo processing, anti-aliasing, and export.
**When to use:** Every generation request flows through this pipeline.
**Example:**
```python
@dataclass
class GenerationConfig:
    """Parameters for a single generation request."""
    duration_s: float = 1.0          # Desired output duration in seconds
    seed: int | None = None          # Random seed for reproducibility
    chunk_duration_s: float = 1.0    # Duration of each generated chunk
    concat_mode: str = "crossfade"   # "crossfade" or "latent_interpolation"
    stereo_mode: str = "mono"        # "mono", "mid_side", "dual_seed"
    stereo_width: float = 0.7        # Width parameter for mid-side (0.0-1.0)
    sample_rate: int = 48_000        # Output sample rate
    bit_depth: str = "PCM_24"        # soundfile subtype
    # ... export config


class GenerationPipeline:
    """Orchestrates audio generation from a trained VAE model."""

    def __init__(
        self,
        model: ConvVAE,
        spectrogram: AudioSpectrogram,
        device: torch.device,
    ) -> None:
        self.model = model
        self.spectrogram = spectrogram
        self.device = device

    def generate(self, config: GenerationConfig) -> GenerationResult:
        """Full generation pipeline: chunks -> concat -> stereo -> anti-alias -> export."""
        # 1. Compute chunk count
        # 2. Generate mel chunks from latent vectors
        # 3. Convert each chunk to waveform
        # 4. Concatenate chunks (crossfade or latent interpolation)
        # 5. Apply anti-aliasing filter
        # 6. Apply stereo processing (if not mono)
        # 7. Compute quality metrics
        # 8. Return GenerationResult with audio + metrics
```

### Pattern 2: Chunk-Based Generation with Configurable Concatenation
**What:** For audio longer than one chunk, generate multiple chunks and combine them. Two modes: crossfade (overlap-add in waveform domain) and latent interpolation (SLERP between latent vectors, decode each frame).
**When to use:** Any generation request where `duration_s > chunk_duration_s`.
**Key detail:** The VAE's `sample()` method currently generates 1-second chunks (default mel shape 128x94 at 48kHz). For multi-chunk generation, we generate N latent vectors and either crossfade the decoded waveforms or interpolate between latent vectors before decoding.

```python
def generate_chunks_crossfade(
    model: ConvVAE,
    spectrogram: AudioSpectrogram,
    num_chunks: int,
    device: torch.device,
    seed: int | None,
    overlap_samples: int = 2400,  # 50ms at 48kHz
) -> torch.Tensor:
    """Generate chunks and crossfade them together."""
    # Generate all latent vectors at once for efficiency
    z_vectors = _sample_latent_vectors(model, num_chunks, device, seed)

    waveforms = []
    for z in z_vectors:
        mel = model.decode(z.unsqueeze(0), target_shape=mel_shape)
        wav = spectrogram.mel_to_waveform(mel.cpu())
        waveforms.append(wav.squeeze())

    return _crossfade_chunks(waveforms, overlap_samples)


def generate_chunks_latent_interp(
    model: ConvVAE,
    spectrogram: AudioSpectrogram,
    num_chunks: int,
    device: torch.device,
    seed: int | None,
    steps_between: int = 10,
) -> torch.Tensor:
    """Generate audio via SLERP interpolation between latent vectors."""
    # Generate anchor latent vectors
    z_anchors = _sample_latent_vectors(model, num_chunks, device, seed)

    # Interpolate between anchors using SLERP
    all_z = []
    for i in range(len(z_anchors) - 1):
        for t in torch.linspace(0, 1, steps_between):
            z_interp = slerp(z_anchors[i], z_anchors[i + 1], t.item())
            all_z.append(z_interp)
    all_z.append(z_anchors[-1])

    # Decode all interpolated frames and concatenate
    # ...
```

### Pattern 3: Anti-Aliasing Before Sample Rate Conversion
**What:** Apply a low-pass Butterworth filter at the Nyquist frequency of the target sample rate before any downsampling operation.
**When to use:** Before exporting at a sample rate lower than the generation rate (e.g., 48kHz -> 44.1kHz), and as a final quality pass on all generated audio.
**Key detail:** Also apply anti-aliasing to the GriffinLim output to suppress artifacts above 20kHz.

```python
def apply_anti_alias_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 20_000.0,
    order: int = 8,
) -> np.ndarray:
    """Apply Butterworth low-pass filter for anti-aliasing."""
    from scipy.signal import butter, sosfiltfilt

    nyquist = sample_rate / 2.0
    # Clamp cutoff below Nyquist
    normalized_cutoff = min(cutoff_hz, nyquist * 0.95) / nyquist
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, audio, axis=-1)
```

### Pattern 4: Mid-Side Stereo Processing
**What:** Create stereo from mono using mid-side decomposition with width control, or Haas effect (delay-based widening).
**When to use:** When user selects stereo output mode.

```python
def apply_mid_side_widening(
    mono: np.ndarray,
    width: float = 0.7,
    sample_rate: int = 48_000,
    haas_delay_ms: float = 15.0,
) -> np.ndarray:
    """Create stereo from mono using Haas effect + mid-side width control.

    1. Create L/R by delaying one channel (Haas effect)
    2. Decompose to mid/side
    3. Scale side by width parameter
    4. Reconstruct L/R
    """
    delay_samples = int(haas_delay_ms * sample_rate / 1000.0)

    # Create delayed copy for right channel
    left = mono.copy()
    right = np.zeros_like(mono)
    right[delay_samples:] = mono[:-delay_samples] if delay_samples > 0 else mono

    # Mid-side decomposition
    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    # Apply width
    new_left = mid + width * side
    new_right = mid - width * side

    # Stack as [2, samples]
    return np.stack([new_left, new_right], axis=0)
```

### Anti-Patterns to Avoid
- **Generating full-length audio in one shot:** The VAE is trained on fixed-length chunks. Generating arbitrarily long mel spectrograms would require padding far beyond training distribution. Always use chunk-based generation.
- **Skipping anti-aliasing before sample rate conversion:** Downsampling without a low-pass filter creates audible aliasing artifacts. Always filter before downsampling.
- **Writing WAV with wrong subtype for bit depth:** Using `PCM_16` when user requests 24-bit silently degrades quality. Map bit depth choices to soundfile subtypes explicitly.
- **Peak normalizing without headroom:** Peak normalizing to exactly 1.0 leaves zero headroom for DAW processing. Normalize to -1 dBFS (approximately 0.891) for professional output.
- **Running InverseMelScale on MPS device:** The existing code already handles this (forces CPU), but any new code must maintain this pattern. `torch.linalg.lstsq` is broken on MPS.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Anti-aliasing filter | Custom FIR/IIR filter with numpy | `scipy.signal.butter` + `sosfiltfilt` | Butterworth filter design has well-known edge cases; scipy handles numerical stability, second-order sections, and zero-phase filtering |
| Sample rate conversion | Custom polyphase resampler | `torchaudio.transforms.Resample` | Resampling involves sinc interpolation with anti-aliasing; torchaudio's implementation handles rolloff and filter width correctly |
| WAV bit depth encoding | Manual PCM byte packing | `soundfile.write(subtype=...)` | PCM_24 encoding is non-trivial (3-byte packing); soundfile/libsndfile handles all formats correctly |
| SLERP interpolation | Linear interpolation (lerp) for latent vectors | Spherical interpolation (SLERP) | Linear interpolation in high-dimensional spaces changes magnitude; SLERP preserves vector norms which matters for latent space traversal |
| Spectrogram rendering | Custom FFT + plotting | `matplotlib.pyplot.specgram` or existing `AudioSpectrogram` + matplotlib | matplotlib's specgram handles windowing, overlap, colormap, and frequency axis labeling |

**Key insight:** The audio DSP domain has subtle gotchas (filter stability, phase response, sample rate ratio math, PCM encoding) that well-tested libraries handle correctly. Every hand-rolled solution is a potential source of audible artifacts.

## Common Pitfalls

### Pitfall 1: GriffinLim Phase Recovery Artifacts
**What goes wrong:** GriffinLim reconstructs phase iteratively. With insufficient iterations, the output has metallic/buzzy artifacts. With too many iterations, processing time increases significantly with diminishing returns.
**Why it happens:** The mel-to-linear spectrogram inversion (InverseMelScale) is approximate, and GriffinLim's phase estimation converges slowly for complex spectral shapes.
**How to avoid:** Use at least 64 iterations (current codebase uses 64, which is good). Use momentum=0.99 (fast GriffinLim). Accept that GriffinLim output will never be perfect -- quality improvements come from training better models, not more GriffinLim iterations.
**Warning signs:** Metallic/ringing quality in generated audio; spectral artifacts visible above 15kHz in spectrogram.

### Pitfall 2: Chunk Boundary Discontinuities
**What goes wrong:** When concatenating independently generated chunks, the waveform has audible clicks or thumps at chunk boundaries.
**Why it happens:** Each chunk is generated from a random latent vector. The waveforms at chunk boundaries have no phase or amplitude coherence.
**How to avoid:** For crossfade mode: use overlap-add with a smooth window (Hann). Overlap of 50ms (2400 samples at 48kHz) is a good default. For latent interpolation mode: generate densely interpolated frames so transitions are smooth. Ensure the first/last few milliseconds of each chunk are windowed.
**Warning signs:** Audible clicks when playing back multi-chunk audio; visible discontinuities in waveform display.

### Pitfall 3: Clipping After Stereo Processing
**What goes wrong:** Mid-side widening or Haas delay can push peak levels above 1.0, causing digital clipping.
**Why it happens:** When width > 1.0 or when the Haas delay creates constructive interference, sample values exceed the [-1.0, 1.0] range.
**How to avoid:** Always peak-normalize after stereo processing. Normalize to -1 dBFS (0.891) rather than 1.0 to leave headroom. The quality metrics should detect and report clipping before export.
**Warning signs:** Quality score reports clipping percentage > 0%.

### Pitfall 4: InverseMelScale Hangs on Bad Input
**What goes wrong:** InverseMelScale's `torch.linalg.lstsq` can take minutes or hang if input values are in an unexpected range.
**Why it happens:** The optimization tolerance is hardcoded relative to expected input magnitudes. If mel values are too large or too small, convergence is extremely slow.
**How to avoid:** Ensure mel spectrograms are in the expected normalized range (log1p-normalized, values typically 0-10). Clamp decoder output before mel inversion. The existing code uses `clamp(min=0)` which is correct.
**Warning signs:** Generation takes > 10 seconds for a 1-second chunk; CPU usage at 100% during mel inversion.

### Pitfall 5: Wrong Sample Rate Ratio for Resampling
**What goes wrong:** When resampling from 48kHz to 44.1kHz, the ratio 44100/48000 = 147/160 requires exact rational arithmetic. Using floating-point division introduces subtle pitch drift.
**Why it happens:** Floating-point imprecision in sample count calculations.
**How to avoid:** Use `torchaudio.transforms.Resample` which handles the rational ratio correctly internally. When computing expected sample counts, use integer arithmetic: `new_samples = int(original_samples * new_rate / original_rate)`.
**Warning signs:** Generated audio is slightly sharp or flat when exported at different sample rates.

### Pitfall 6: Sidecar JSON Not Written Atomically
**What goes wrong:** If the process crashes between writing the WAV and writing the JSON sidecar, the WAV exists without metadata.
**Why it happens:** Two separate file writes are not atomic.
**How to avoid:** Write JSON sidecar first (it's small and fast), then write WAV. Or write both to temp files and rename atomically. For v1, writing JSON first is sufficient.
**Warning signs:** Orphan WAV files without corresponding JSON sidecars.

## Code Examples

Verified patterns from official sources and existing codebase:

### WAV Export with Configurable Format
```python
# Source: soundfile official docs (https://python-soundfile.readthedocs.io/)
import numpy as np
import soundfile as sf

# Map user-facing bit depth to soundfile subtype
BIT_DEPTH_MAP = {
    "16-bit": "PCM_16",
    "24-bit": "PCM_24",
    "32-bit float": "FLOAT",
}

def export_wav(
    audio: np.ndarray,       # [channels, samples] or [samples] for mono
    path: str,
    sample_rate: int = 48_000,
    bit_depth: str = "24-bit",
) -> None:
    """Export audio as WAV with specified format."""
    subtype = BIT_DEPTH_MAP[bit_depth]

    # soundfile expects [samples, channels] for multichannel
    if audio.ndim == 2:
        audio = audio.T  # [channels, samples] -> [samples, channels]

    sf.write(path, audio, sample_rate, subtype=subtype)
```

### Quality Metrics: SNR Calculation
```python
# Source: Standard DSP practice
import numpy as np

def compute_snr_db(audio: np.ndarray, silence_threshold: float = 0.01) -> float:
    """Estimate signal-to-noise ratio in decibels.

    Segments below silence_threshold are classified as noise.
    Uses RMS power ratio.
    """
    # Frame-based analysis (10ms frames)
    frame_size = int(0.01 * 48_000)  # 480 samples per frame
    num_frames = len(audio) // frame_size

    signal_power = 0.0
    noise_power = 0.0
    signal_frames = 0
    noise_frames = 0

    for i in range(num_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        power = np.mean(frame ** 2)

        if rms > silence_threshold:
            signal_power += power
            signal_frames += 1
        else:
            noise_power += power
            noise_frames += 1

    if noise_frames == 0 or noise_power == 0:
        return float("inf")  # No detectable noise
    if signal_frames == 0:
        return 0.0  # All silence/noise

    avg_signal = signal_power / signal_frames
    avg_noise = noise_power / noise_frames
    return 10 * np.log10(avg_signal / avg_noise)
```

### Quality Metrics: Clipping Detection
```python
import numpy as np

def detect_clipping(
    audio: np.ndarray,
    threshold: float = 0.999,
    consecutive_samples: int = 3,
) -> dict:
    """Detect digital clipping in audio.

    Returns dict with clipping stats: clipped_samples, clipped_percentage,
    peak_value, max_consecutive_clipped.
    """
    abs_audio = np.abs(audio.flatten())
    clipped_mask = abs_audio >= threshold
    clipped_count = int(np.sum(clipped_mask))
    total_samples = len(abs_audio)

    # Find max consecutive clipped samples
    max_consecutive = 0
    current_run = 0
    for is_clipped in clipped_mask:
        if is_clipped:
            current_run += 1
            max_consecutive = max(max_consecutive, current_run)
        else:
            current_run = 0

    return {
        "clipped_samples": clipped_count,
        "clipped_percentage": (clipped_count / total_samples) * 100 if total_samples > 0 else 0.0,
        "peak_value": float(abs_audio.max()),
        "max_consecutive_clipped": max_consecutive,
        "has_clipping": clipped_count > 0,
    }
```

### SLERP for Latent Space Interpolation
```python
# Source: Birch-san/230ac46f99ec411ed5907b0a3d728efa (PyTorch SLERP gist)
import torch

def slerp(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: float,
    dot_threshold: float = 0.9995,
) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors.

    Falls back to linear interpolation when vectors are nearly parallel.
    """
    v0_norm = v0 / torch.linalg.norm(v0)
    v1_norm = v1 / torch.linalg.norm(v1)

    dot = torch.sum(v0_norm * v1_norm)

    # If nearly parallel, fall back to linear interpolation
    if torch.abs(dot) > dot_threshold:
        return torch.lerp(v0, v1, t)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    return s0 * v0 + s1 * v1
```

### Crossfade with Hann Window
```python
import numpy as np

def crossfade_chunks(
    chunks: list[np.ndarray],
    overlap_samples: int = 2400,  # 50ms at 48kHz
) -> np.ndarray:
    """Concatenate audio chunks with Hann-windowed crossfade."""
    if len(chunks) == 0:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    # Create Hann crossfade window
    window = np.hanning(2 * overlap_samples)
    fade_out = window[:overlap_samples]
    fade_in = window[overlap_samples:]

    # Calculate total length
    chunk_len = len(chunks[0])
    total = chunk_len + (len(chunks) - 1) * (chunk_len - overlap_samples)
    output = np.zeros(total, dtype=np.float32)

    pos = 0
    for i, chunk in enumerate(chunks):
        if i == 0:
            output[pos:pos + chunk_len] = chunk
        else:
            # Apply fade out to existing audio in overlap region
            output[pos:pos + overlap_samples] *= fade_out
            # Apply fade in to new chunk's overlap region
            chunk_faded = chunk.copy()
            chunk_faded[:overlap_samples] *= fade_in
            # Add (overlap-add)
            output[pos:pos + overlap_samples] += chunk_faded[:overlap_samples]
            output[pos + overlap_samples:pos + chunk_len] = chunk_faded[overlap_samples:]
        pos += chunk_len - overlap_samples

    return output[:pos + overlap_samples]  # Trim to actual length
```

### Sidecar JSON Export
```python
import json
from datetime import datetime, timezone
from pathlib import Path

def write_sidecar_json(
    wav_path: Path,
    model_name: str,
    generation_config: dict,
    seed: int,
    quality_metrics: dict,
) -> Path:
    """Write generation metadata alongside exported WAV."""
    sidecar_path = wav_path.with_suffix(".json")
    metadata = {
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "seed": seed,
        "generation": generation_config,
        "quality": quality_metrics,
        "audio": {
            "file": wav_path.name,
            "format": "WAV",
            "sample_rate": generation_config.get("sample_rate", 48000),
            "bit_depth": generation_config.get("bit_depth", "24-bit"),
            "channels": generation_config.get("channels", 1),
            "duration_s": generation_config.get("duration_s", 1.0),
        },
    }
    sidecar_path.write_text(json.dumps(metadata, indent=2))
    return sidecar_path
```

## Discretion Recommendations

These are areas marked as Claude's Discretion in CONTEXT.md. Research supports these recommendations:

### Stereo Width Control: Use a Slider (0.0 to 1.5)
**Recommendation:** A continuous slider from 0.0 (mono) to 1.5 (wide stereo) with 1.0 as "natural" width. This is more flexible than presets and matches the standard DAW paradigm. The slider maps directly to the `width` parameter in mid-side processing.
**Rationale:** Professional audio tools universally use continuous width controls. A slider gives the user fine-grained control without cognitive overhead.

### Export File Destination: Project Output Folder with Save-As Override
**Recommendation:** Default to `data/generated/{model_name}/` within the project. Files are auto-named with timestamp + seed (e.g., `gen_20260212_153042_seed42.wav`). A "Save As..." option allows the user to pick a custom destination.
**Rationale:** Auto-saving to a project folder prevents "where did my file go?" confusion. The save-as option handles the common case of exporting directly to a DAW project folder.

### Quality Score Presentation: Numeric + Traffic Light
**Recommendation:** Show a simple traffic light indicator (green/yellow/red) with numeric SNR value and clipping status on hover or in a detail panel.
- **Green:** SNR > 30 dB, no clipping
- **Yellow:** SNR 15-30 dB, or < 0.1% clipped samples
- **Red:** SNR < 15 dB, or > 0.1% clipped samples
**Rationale:** The traffic light gives instant feedback without requiring audio engineering knowledge. The numeric values are there for power users who want them.

### Anti-Aliasing Filter: 8th-Order Butterworth at 20kHz
**Recommendation:** Apply an 8th-order Butterworth low-pass filter with cutoff at min(20kHz, target_nyquist * 0.95) using `scipy.signal.butter` + `sosfiltfilt` (zero-phase filtering). Apply this:
1. After GriffinLim output (removes reconstruction artifacts above audible range)
2. Before any downsampling (standard anti-aliasing)
**Rationale:** Butterworth has maximally flat passband (preserves audio character). 8th order gives ~48 dB/octave rolloff which is steep enough to suppress aliasing while keeping the transition band narrow. Zero-phase filtering (`sosfiltfilt`) avoids phase distortion.

### Crossfade Overlap Duration: 50ms Default
**Recommendation:** 50 milliseconds (2400 samples at 48kHz) with Hann window. This is short enough to not audibly smear transients but long enough to eliminate clicks.
**Rationale:** Professional DAW crossfade defaults are typically 10-100ms. 50ms is the sweet spot for generative audio where chunks don't have rhythmic alignment. Users can adjust in future versions.

### Batch Variations: Not for v1
**Recommendation:** Defer batch generation to a later phase. For v1, generate one output at a time. The architecture should support generating with different seeds, but the UI/API for "generate 5 variations" is unnecessary complexity now.
**Rationale:** The generation pipeline naturally supports different seeds per call. Batch UI is a Phase 8/9 concern.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| GriffinLim (64 iters) | GriffinLim (64 iters, momentum 0.99) | torchaudio 0.10+ | Fast GriffinLim is ~2x faster than vanilla; momentum parameter added |
| scipy.signal.lfilter | scipy.signal.sosfiltfilt | SciPy 0.16+ | Second-order sections (SOS) avoid numerical instability with high-order filters |
| Manual PCM_24 byte packing | soundfile subtype="PCM_24" | soundfile 0.9+ | libsndfile handles all PCM formats natively; no manual encoding needed |
| Linear interpolation in latent space | SLERP for latent vectors | Research consensus ~2018+ | SLERP preserves vector magnitude; linear interpolation causes "dimming" in high-dimensional spaces |
| torchaudio.Resample (sinc_interp_hann) | sinc_interp_kaiser option available | torchaudio 0.12+ | Kaiser window gives better stopband attenuation; useful for professional audio resampling |

**Deprecated/outdated:**
- `scipy.stats.signaltonoise`: Removed from scipy. Compute SNR manually with numpy.
- `torchaudio.load()` / `torchaudio.info()`: Removed in torchaudio 2.10. Project already uses soundfile for I/O (correct).

## Open Questions

1. **GriffinLim quality ceiling for professional use**
   - What we know: GriffinLim is lossy; phase recovery introduces artifacts. 64 iterations is a good balance of quality vs speed.
   - What's unclear: Whether the quality is "good enough" for the user's professional production workflow, or whether a neural vocoder will be needed sooner.
   - Recommendation: Proceed with GriffinLim for v1. Add a `vocoder` parameter to `GenerationPipeline` to allow swapping in a neural vocoder later without architecture changes. Monitor user feedback on quality.

2. **Latent interpolation step density**
   - What we know: SLERP between anchor vectors produces smooth transitions. More interpolation steps = smoother but slower.
   - What's unclear: The optimal number of interpolation steps per second for the specific VAE latent space. This depends on the trained model.
   - Recommendation: Default to 10 steps per chunk boundary (yielding ~10 micro-chunks per transition). Make configurable. The user can tune after hearing results.

3. **Anti-aliasing at 96kHz target**
   - What we know: When upsampling from 48kHz to 96kHz, anti-aliasing is less critical (adding frequencies, not removing). The main concern is image frequencies from the upsampling.
   - What's unclear: Whether torchaudio.Resample handles 48kHz->96kHz upsampling cleanly or if additional filtering is needed.
   - Recommendation: Use torchaudio.Resample with `rolloff=0.99` for all conversions. Add spectral analysis verification in quality metrics to detect issues.

4. **Memory usage for long generation (60s)**
   - What we know: 60 seconds at 48kHz = 2,880,000 samples. Each chunk generates ~48,000 samples. That's ~60 chunks. The mel spectrogram for each chunk is 128x94 floats. InverseMelScale + GriffinLim runs on CPU.
   - What's unclear: Whether generating 60 chunks sequentially causes memory pressure from accumulated waveforms.
   - Recommendation: Generate and concatenate chunks incrementally (streaming approach). Don't hold all mel spectrograms in memory simultaneously. Process one chunk at a time, append to output buffer.

## Sources

### Primary (HIGH confidence)
- soundfile official docs: https://python-soundfile.readthedocs.io/ -- WAV subtypes (PCM_16, PCM_24, FLOAT), write() API
- torchaudio GriffinLim docs: https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html -- n_iter, momentum, power parameters
- torchaudio InverseMelScale docs: https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html -- lstsq solver, CPU requirement
- torchaudio Resample docs: https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.Resample.html -- rolloff, lowpass_filter_width, resampling_method
- SciPy signal docs: https://docs.scipy.org/doc/scipy/reference/signal.html -- butter, sosfiltfilt, resample_poly
- Existing codebase: `src/small_dataset_audio/audio/spectrogram.py`, `training/preview.py` -- current mel-to-waveform and WAV export patterns

### Secondary (MEDIUM confidence)
- PyTorch SLERP implementation (Birch-san gist): https://gist.github.com/Birch-san/230ac46f99ec411ed5907b0a3d728efa -- SLERP code pattern verified against mathematical definition
- Hack Audio stereo widening: https://www.hackaudio.com/digital-signal-processing/stereo-audio/stereo-image-widening/ -- mid-side processing math
- Stefan Behrens audio DSP: https://www.sbehrens4d.com/posts/python_dsp_1_panning.html -- stereo width implementation in Python/numpy
- Izotope on Haas effect: https://www.izotope.com/en/learn/what-is-the-haas-effect -- delay range 5-35ms for stereo widening
- MusicVAE on latent interpolation: https://magenta.tensorflow.org/music-vae -- SLERP for audio latent space traversal

### Tertiary (LOW confidence)
- Neural vocoder comparison (HiFi-GAN vs GriffinLim CLAP scores): General research consensus, not verified with specific benchmark -- flagged for future investigation if quality ceiling is hit

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project or well-established (soundfile, torchaudio, scipy)
- Architecture: HIGH -- chunk-based generation is proven pattern; pipeline architecture follows existing codebase patterns
- Anti-aliasing: HIGH -- Butterworth + sosfiltfilt is textbook DSP; scipy implementation is battle-tested
- Stereo processing: MEDIUM -- mid-side math is well-established; Haas effect delay values are from professional audio sources but implementation details need tuning
- Quality metrics: MEDIUM -- SNR calculation is standard but the specific thresholds for "good enough" generated audio are empirical
- Latent interpolation: MEDIUM -- SLERP is proven for latent spaces but optimal step density depends on this specific VAE's latent space structure
- Pitfalls: HIGH -- identified from existing codebase patterns and known audio DSP issues

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (stable domain; libraries unlikely to change significantly)
