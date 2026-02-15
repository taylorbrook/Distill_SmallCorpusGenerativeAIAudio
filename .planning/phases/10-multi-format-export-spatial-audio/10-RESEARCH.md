# Phase 10: Multi-Format Export & Spatial Audio - Research

**Researched:** 2026-02-14
**Domain:** Audio encoding/export, spatial audio DSP, multi-model inference blending
**Confidence:** HIGH (core stack) / MEDIUM (spatial audio) / MEDIUM (multi-model blending)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- All three formats (MP3, FLAC, OGG) are first-class citizens with equal priority
- MP3 default: 320 kbps CBR (maximum quality)
- FLAC compression: level 8 (max compression, smallest files)
- OGG quality: Claude's discretion for sensible default
- Format selection available everywhere audio is exported (generate tab, history, CLI) -- not a separate export step
- Replaces the existing Phase 4 stereo width parameter (0.0-1.5) entirely
- Two control dimensions: width + depth (front-back)
- Output mode selector: stereo, binaural, or mono -- spatial controls adapt to selected mode
- Binaural target: immersive headphone experience with full HRTF-based spatialization
- Up to 4 models loaded simultaneously
- Individual weight slider per model (0-100%), auto-normalized to sum to 100%
- User toggle between latent-space blending and audio-domain blending
- Union of all sliders from loaded models -- each slider maps to whichever models have that parameter
- Key info embedded in audio file tags: model name, seed, preset name
- Full provenance retained in sidecar JSON (complements, doesn't replace Phase 4 pattern)
- Default tags: Artist = "SDA Generator", Album = model name
- Metadata fields are editable before export -- user can override any tag

### Claude's Discretion
- OGG quality/bitrate default
- HRTF dataset selection for binaural rendering
- How union sliders handle models that don't share a parameter (zero-fill, skip, interpolate)
- Exact normalization behavior for blend weight sliders

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Summary

Phase 10 spans three distinct technical domains: multi-format audio export (MP3, FLAC, OGG), spatial audio with HRTF-based binaural rendering, and multi-model blending. Each has a mature ecosystem of Python libraries, but the specific requirements (320 kbps CBR MP3, embedded metadata, HRTF convolution) dictate careful library selection.

**Multi-format export** is best served by a two-library approach: soundfile (already a dependency) handles FLAC and OGG natively with full compression control, while `lameenc` provides direct 320 kbps CBR MP3 encoding without requiring FFmpeg as a system dependency. Metadata embedding across all three formats uses `mutagen`, which supports ID3v2 (MP3), Vorbis Comments (FLAC, OGG), and operates losslessly on audio data.

**Spatial audio** replaces the existing Phase 4 stereo width system entirely. The new spatial system adds a depth (front-back) dimension and a binaural output mode. Binaural rendering uses HRTF impulse response convolution via `scipy.signal.fftconvolve` (already a project dependency). The MIT KEMAR HRTF dataset is recommended -- it is freely available in SOFA format, well-documented, and widely used as a reference standard. The `sofar` library reads SOFA files into numpy arrays with minimal dependencies.

**Multi-model blending** is architecturally the most complex addition. Latent-space blending requires all models to share the same `latent_dim` (currently 64); audio-domain blending works universally. The union slider concept (single slider set mapping to whichever models have each parameter) requires a new slider resolution layer above the existing `SliderState` / `sliders_to_latent` pipeline.

**Primary recommendation:** Use soundfile for FLAC/OGG, lameenc for MP3 320 CBR, mutagen for metadata embedding, scipy.signal.fftconvolve with MIT KEMAR SOFA data for binaural, and numpy-level weighted averaging for both blending modes.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| soundfile | 0.13.x (already dep) | FLAC and OGG export with compression control | Already in project, libsndfile handles FLAC level 0-8 and OGG Vorbis quality natively |
| lameenc | >=1.7 | MP3 encoding with explicit kbps CBR control | Pre-built LAME binaries for all platforms, no FFmpeg needed, direct 320 kbps CBR via `set_bit_rate(320)` |
| mutagen | >=1.47 | Audio metadata tagging (ID3, Vorbis Comments) | Industry standard Python metadata library, no dependencies, supports MP3/FLAC/OGG losslessly |
| scipy | >=1.12 (already dep) | HRTF convolution via `scipy.signal.fftconvolve` | Already in project, FFT convolution is the standard approach for HRIR application |
| sofar | >=1.1 | Read SOFA-format HRTF datasets into numpy arrays | Clean API for SOFA files, actively maintained by pyfar team, minimal dependencies (netCDF4) |
| numpy | >=1.26 (already dep) | Audio-domain blending, spatial processing math | Already in project, all audio post-processing uses numpy arrays |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| netCDF4 | >=1.6 | Low-level SOFA file reading (sofar dependency) | Installed automatically with sofar |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| lameenc (MP3) | soundfile (libsndfile) | soundfile's MP3 bitrate control is broken in python-soundfile -- compression_level does not map to specific kbps values. GitHub issue #390 remains open. Cannot guarantee 320 kbps CBR. |
| lameenc (MP3) | pydub + ffmpeg | Requires FFmpeg as system dependency. Heavy for this use case. Project explicitly avoids FFmpeg (see audio/io.py design notes). |
| sofar (SOFA) | pysofaconventions | Older API, less actively maintained. sofar has cleaner numpy integration. |
| sofar (SOFA) | Bundled numpy arrays | Could pre-convert HRTF data to .npy files and skip sofar dependency. Viable but less flexible and harder to swap datasets. |

**Installation:**
```bash
uv add lameenc mutagen sofar
```

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── inference/
│   ├── export.py          # EXTEND: add export_mp3, export_flac, export_ogg, ExportFormat enum
│   ├── generation.py      # EXTEND: GenerationConfig gains spatial + format fields; new export dispatch
│   ├── stereo.py          # REPLACE: becomes spatial.py with width/depth/binaural modes
│   ├── quality.py         # unchanged
│   └── chunking.py        # unchanged
├── inference/
│   └── blending.py        # NEW: multi-model loading, weight normalization, blend dispatch
├── audio/
│   └── hrtf.py            # NEW: HRTF loading from SOFA, binaural convolution
├── audio/
│   └── metadata.py        # NEW: format-aware tag embedding via mutagen
├── data/
│   └── hrtf/              # NEW: bundled HRTF dataset (MIT KEMAR SOFA file, ~1.5 MB)
└── ui/tabs/
    └── generate_tab.py    # EXTEND: format selector, spatial controls, multi-model panel
```

### Pattern 1: Format-Agnostic Export Pipeline
**What:** Single `export_audio()` entry point that dispatches to format-specific encoders, then applies metadata.
**When to use:** Every export path (generate tab, history re-export, CLI).
**Example:**
```python
from enum import Enum
from pathlib import Path

class ExportFormat(Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"

def export_audio(
    audio: "np.ndarray",
    path: Path,
    sample_rate: int,
    format: ExportFormat,
    bit_depth: str = "24-bit",
    metadata: dict | None = None,
) -> Path:
    """Export audio in any supported format with optional metadata."""
    if format == ExportFormat.WAV:
        _export_wav(audio, path, sample_rate, bit_depth)
    elif format == ExportFormat.MP3:
        _export_mp3(audio, path, sample_rate)  # 320 CBR default
    elif format == ExportFormat.FLAC:
        _export_flac(audio, path, sample_rate)  # level 8 default
    elif format == ExportFormat.OGG:
        _export_ogg(audio, path, sample_rate)  # quality 6 default

    if metadata:
        _embed_metadata(path, format, metadata)

    return path
```

### Pattern 2: Two-Phase Export (Encode then Tag)
**What:** Separate audio encoding from metadata embedding. Encode first, then use mutagen to tag.
**When to use:** Always. Mutagen operates losslessly on existing files.
**Example:**
```python
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TXXX
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis

def _embed_metadata(path: Path, format: ExportFormat, metadata: dict) -> None:
    """Embed metadata tags into an already-encoded audio file."""
    if format == ExportFormat.MP3:
        audio = ID3()
        audio.add(TPE1(encoding=3, text=[metadata.get("artist", "SDA Generator")]))
        audio.add(TALB(encoding=3, text=[metadata.get("album", "")]))
        audio.add(TIT2(encoding=3, text=[metadata.get("title", "")]))
        # Custom fields for provenance
        if "seed" in metadata:
            audio.add(TXXX(encoding=3, desc="SDA_SEED", text=[str(metadata["seed"])]))
        if "model_name" in metadata:
            audio.add(TXXX(encoding=3, desc="SDA_MODEL", text=[metadata["model_name"]]))
        audio.save(str(path))

    elif format == ExportFormat.FLAC:
        audio = FLAC(str(path))
        audio["artist"] = metadata.get("artist", "SDA Generator")
        audio["album"] = metadata.get("album", "")
        audio["title"] = metadata.get("title", "")
        if "seed" in metadata:
            audio["sda_seed"] = str(metadata["seed"])
        audio.save()

    elif format == ExportFormat.OGG:
        audio = OggVorbis(str(path))
        audio["artist"] = [metadata.get("artist", "SDA Generator")]
        audio["album"] = [metadata.get("album", "")]
        audio["title"] = [metadata.get("title", "")]
        if "seed" in metadata:
            audio["sda_seed"] = [str(metadata["seed"])]
        audio.save()
    # WAV: no embedded tags (sidecar JSON only, existing pattern)
```

### Pattern 3: HRTF-Based Binaural Rendering
**What:** Convolve mono audio with left/right HRIR filters for binaural output.
**When to use:** Binaural output mode selection.
**Example:**
```python
import numpy as np
from scipy.signal import fftconvolve

def apply_binaural(
    mono: np.ndarray,
    hrir_left: np.ndarray,
    hrir_right: np.ndarray,
    width: float = 1.0,
    depth: float = 0.5,
) -> np.ndarray:
    """Apply HRTF-based binaural rendering to mono audio.

    Parameters
    ----------
    mono : np.ndarray
        1-D mono audio [samples].
    hrir_left, hrir_right : np.ndarray
        Left/right head-related impulse responses for the target position.
    width : float
        Spatial width (0.0=mono center, 1.0=natural).
    depth : float
        Front-back depth (0.0=very close, 1.0=distant).
    """
    # Convolve with left and right HRIRs
    left = fftconvolve(mono, hrir_left, mode='full')[:len(mono)]
    right = fftconvolve(mono, hrir_right, mode='full')[:len(mono)]

    # Width control: blend between mono (center) and binaural (wide)
    center = (left + right) * 0.5
    left_out = center + width * (left - center)
    right_out = center + width * (right - center)

    return np.stack([left_out, right_out], axis=0).astype(np.float32)
```

### Pattern 4: Normalized Blend Weights
**What:** Auto-normalize model weight sliders to sum to 100%.
**When to use:** Multi-model blending.
**Example:**
```python
def normalize_weights(raw_weights: list[float]) -> list[float]:
    """Normalize blend weights to sum to 1.0.

    If all weights are zero, distribute equally.
    """
    total = sum(raw_weights)
    if total == 0:
        n = len(raw_weights)
        return [1.0 / n] * n
    return [w / total for w in raw_weights]
```

### Pattern 5: Latent-Space vs Audio-Domain Blending
**What:** Two distinct blending strategies with user toggle.
**When to use:** Multi-model generation.
**Example:**
```python
def blend_latent_space(
    models: list["ConvVAE"],
    latent_vectors: list["np.ndarray"],
    weights: list[float],
) -> "np.ndarray":
    """Weighted average of latent vectors before decoding.

    Requires all models share the same latent_dim.
    """
    import numpy as np
    weights = normalize_weights(weights)
    blended = sum(w * z for w, z in zip(weights, latent_vectors))
    return blended

def blend_audio_domain(
    audio_outputs: list["np.ndarray"],
    weights: list[float],
) -> "np.ndarray":
    """Weighted average of audio waveforms after generation.

    Works with any models regardless of architecture.
    """
    import numpy as np
    weights = normalize_weights(weights)
    # Pad shorter outputs with zeros
    max_len = max(a.shape[-1] for a in audio_outputs)
    padded = []
    for a in audio_outputs:
        if a.shape[-1] < max_len:
            pad_width = max_len - a.shape[-1]
            if a.ndim == 1:
                a = np.pad(a, (0, pad_width))
            else:
                a = np.pad(a, ((0, 0), (0, pad_width)))
        padded.append(a)
    return sum(w * a for w, a in zip(weights, padded)).astype(np.float32)
```

### Anti-Patterns to Avoid
- **FFmpeg dependency for encoding:** The project explicitly avoids FFmpeg (see `audio/io.py` design notes). Use lameenc for MP3 instead of pydub/ffmpeg.
- **Inline format branching everywhere:** Don't scatter format-specific code across UI, CLI, and pipeline. Centralize in `export.py` behind `ExportFormat` enum.
- **Coupling spatial mode to stereo mode:** The new spatial system replaces `stereo_mode` and `stereo_width` entirely. Don't layer spatial on top of the old stereo system -- replace it.
- **Loading HRTF data on every render:** HRIRs should be loaded once at startup/model-load and cached. A typical SOFA file is ~1.5 MB, but convolution data should be pre-extracted to numpy arrays.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MP3 encoding with bitrate control | Custom LAME bindings | lameenc | Cross-platform binaries, trivial API, exact kbps control |
| Audio metadata tagging | Custom binary tag writers | mutagen | ID3v2, Vorbis Comments, and FLAC tags are complex binary formats with versioning |
| HRTF dataset parsing | Custom SOFA/MATLAB readers | sofar | SOFA format is netCDF-based with conventions; sofar handles all edge cases |
| FFT convolution | Manual numpy FFT + multiply + IFFT | scipy.signal.fftconvolve | Handles edge cases (padding, aliasing), optimized FFT sizes |
| OGG Vorbis encoding | Custom vorbis bindings | soundfile (libsndfile) | libsndfile bundles Vorbis encoder, no extra deps |

**Key insight:** Audio encoding formats (MP3, FLAC, OGG) and metadata schemas (ID3v2, Vorbis Comments) are spec-heavy domains with extensive edge cases. Any hand-rolled solution will have bugs that established libraries have already fixed over years of use.

## Common Pitfalls

### Pitfall 1: soundfile Cannot Set MP3 Bitrate Precisely
**What goes wrong:** Using `sf.write('output.mp3', ...)` produces MP3 files but at unpredictable bitrate (often 32-64 kbps). The `compression_level` parameter for MP3 is not properly exposed in python-soundfile.
**Why it happens:** python-soundfile issue #390 is still open. The CFFI bindings don't properly pass `SFC_SET_COMPRESSION_LEVEL` to libsndfile for MP3 format.
**How to avoid:** Use lameenc for MP3 encoding. Convert float32 numpy arrays to int16 PCM bytes, then encode with `encoder.set_bit_rate(320)`.
**Warning signs:** MP3 files that are unexpectedly small or sound low-quality.

### Pitfall 2: HRTF Sample Rate Mismatch
**What goes wrong:** Binaural output sounds metallic or has pitch artifacts.
**Why it happens:** HRIR data has its own sample rate (e.g., MIT KEMAR is 44.1 kHz). If the audio is at 48 kHz and the HRIR is at 44.1 kHz, convolution produces artifacts.
**How to avoid:** Resample the HRIR data to match the audio sample rate when loading the SOFA file. Cache resampled HRIRs keyed by sample rate (existing project pattern from `_resampler_cache`).
**Warning signs:** Subtle pitch shifting or metallic coloring in binaural output.

### Pitfall 3: Metadata Tag Differences Across Formats
**What goes wrong:** Tags set on MP3 don't appear, or FLAC/OGG tags use wrong field names.
**Why it happens:** MP3 uses ID3 frames (TIT2, TPE1, TALB), while FLAC/OGG use Vorbis Comments (case-insensitive string keys). They have completely different APIs in mutagen.
**How to avoid:** Abstract tag setting behind a single function that dispatches by format. Test metadata round-trips for each format.
**Warning signs:** Tags appearing blank in media players for some formats but not others.

### Pitfall 4: Latent Space Dimension Mismatch in Multi-Model Blending
**What goes wrong:** Latent-space blending crashes or produces garbage when models have different `latent_dim`.
**Why it happens:** Weighted average of vectors requires identical dimensions. Models trained with different configs may have different latent dimensions.
**How to avoid:** Validate `latent_dim` match before allowing latent-space blending. Fall back to audio-domain blending if dimensions differ. Show a clear UI warning.
**Warning signs:** Shape errors in numpy operations, or meaningless noise output.

### Pitfall 5: Export-Before-Tag Race Condition
**What goes wrong:** Metadata embedding fails because the file doesn't exist yet.
**Why it happens:** Audio encoding and metadata tagging are two separate steps. If encoding fails silently or writes to wrong path, mutagen fails.
**How to avoid:** Always verify file existence between encode and tag steps. Follow existing project pattern: sidecar JSON first, then audio file (research pitfall #6 from Phase 4).
**Warning signs:** FileNotFoundError in mutagen calls, or empty metadata on exported files.

### Pitfall 6: Multi-Model Memory Pressure
**What goes wrong:** Loading 4 models simultaneously causes OOM, especially on MPS devices.
**Why it happens:** Each ConvVAE is ~3.1M parameters (~12 MB), but with GPU memory fragmentation and spectrogram transforms, 4 models could use 200+ MB of GPU memory.
**How to avoid:** Load models to GPU only when actively generating. Keep inactive models on CPU. Use `model.to('cpu')` / `model.to(device)` transitions. Show memory usage in UI.
**Warning signs:** MPS/CUDA OOM errors, system slowdown with multiple models loaded.

### Pitfall 7: Stereo Width Parameter Migration
**What goes wrong:** Old saved presets, history entries, and GenerationConfig references break because `stereo_width` field is removed.
**Why it happens:** Phase 4's `stereo_width` (0.0-1.5) on `GenerationConfig` is being replaced by the new spatial system (width + depth + output mode).
**How to avoid:** Add backward compatibility: when loading old configs/presets, map `stereo_width` to the new `spatial_width` parameter. Keep `stereo_mode` values working during transition.
**Warning signs:** KeyError or AttributeError when loading old presets or history entries.

## Code Examples

### MP3 Export with lameenc (320 kbps CBR)
```python
# Source: lameenc PyPI docs + research verification
import numpy as np
import lameenc

def export_mp3(
    audio: np.ndarray,
    path: "Path",
    sample_rate: int = 48_000,
    bitrate: int = 320,
) -> "Path":
    """Export audio as MP3 with specified bitrate.

    Converts float32 numpy audio to int16 PCM, then encodes with LAME.
    Handles both mono [samples] and stereo [2, samples] arrays.
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Determine channels
    if audio.ndim == 2:
        channels = audio.shape[0]
        # Interleave: [2, samples] -> [samples, 2] -> flat
        interleaved = audio.T.flatten()
    else:
        channels = 1
        interleaved = audio

    # Convert float32 [-1, 1] to int16
    int16_data = (np.clip(interleaved, -1.0, 1.0) * 32767).astype(np.int16)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)  # 2=highest quality, 7=fastest

    mp3_data = encoder.encode(int16_data.tobytes())
    mp3_data += encoder.flush()

    path.write_bytes(mp3_data)
    return path
```

### FLAC Export with soundfile (Level 8)
```python
# Source: soundfile docs + libsndfile FLAC compression mapping
import numpy as np
import soundfile as sf

def export_flac(
    audio: np.ndarray,
    path: "Path",
    sample_rate: int = 48_000,
    compression_level: int = 8,
) -> "Path":
    """Export audio as FLAC with specified compression level (0-8).

    Level 8 = maximum compression (smallest files, lossless).
    libsndfile maps float 0.0-1.0 to FLAC levels 0-8.
    """
    audio_data = np.asarray(audio, dtype=np.float32)

    # soundfile expects [samples, channels]
    if audio_data.ndim == 2:
        audio_data = audio_data.T

    sf_compression = compression_level / 8.0  # Map 0-8 to 0.0-1.0
    sf.write(
        str(path),
        audio_data,
        sample_rate,
        format='FLAC',
        subtype='PCM_24',
        compression_level=sf_compression,
    )
    return path
```

### OGG Vorbis Export with soundfile
```python
# Source: soundfile docs + Vorbis quality research
import numpy as np
import soundfile as sf

def export_ogg(
    audio: np.ndarray,
    path: "Path",
    sample_rate: int = 48_000,
    quality: float = 0.6,
) -> "Path":
    """Export audio as OGG Vorbis with specified quality.

    Quality maps via libsndfile compression_level:
    0.0 = highest quality (~400 kbps), 1.0 = lowest quality (~64 kbps).
    Default 0.6 inverted = quality ~6 (~192 kbps) -- high quality.

    Note: OGG Vorbis is inherently VBR. The quality parameter
    controls the target quality level, not exact bitrate.
    """
    audio_data = np.asarray(audio, dtype=np.float32)

    if audio_data.ndim == 2:
        audio_data = audio_data.T

    # libsndfile: 0.0 = minimum compression (highest quality),
    #             1.0 = maximum compression (lowest quality).
    # We invert so user-facing quality 6/10 maps to compression 0.4.
    sf_compression = 1.0 - quality
    sf.write(
        str(path),
        audio_data,
        sample_rate,
        format='OGG',
        subtype='VORBIS',
        compression_level=sf_compression,
    )
    return path
```

### HRTF Loading from SOFA
```python
# Source: sofar docs + SOFA conventions
import numpy as np

def load_hrtf_from_sofa(
    sofa_path: "Path",
    target_sample_rate: int = 48_000,
    azimuth: float = 90.0,
    elevation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load left/right HRIR pair from a SOFA file.

    Parameters
    ----------
    sofa_path : Path
        Path to the .sofa HRTF file.
    target_sample_rate : int
        Desired sample rate. HRIRs are resampled if needed.
    azimuth : float
        Source azimuth in degrees (0=front, 90=right, -90=left).
    elevation : float
        Source elevation in degrees (0=ear level, 90=above).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (hrir_left, hrir_right) impulse response arrays.
    """
    import sofar

    sofa = sofar.read_sofa(str(sofa_path))

    # Find nearest source position
    # SOFA coordinates: [azimuth, elevation, distance]
    positions = sofa.SourcePosition  # shape [M, 3]
    target = np.array([azimuth, elevation, 1.0])
    distances = np.linalg.norm(positions[:, :2] - target[:2], axis=1)
    nearest_idx = np.argmin(distances)

    # Extract HRIR: shape [M, R, N] where R=2 (left, right), N=filter length
    hrir_left = sofa.Data_IR[nearest_idx, 0, :]
    hrir_right = sofa.Data_IR[nearest_idx, 1, :]

    # Resample if SOFA sample rate differs
    sofa_sr = int(sofa.Data_SamplingRate)
    if sofa_sr != target_sample_rate:
        from torchaudio.transforms import Resample
        import torch
        resampler = Resample(sofa_sr, target_sample_rate)
        hrir_left = resampler(torch.from_numpy(hrir_left).float().unsqueeze(0)).squeeze().numpy()
        hrir_right = resampler(torch.from_numpy(hrir_right).float().unsqueeze(0)).squeeze().numpy()

    return hrir_left.astype(np.float32), hrir_right.astype(np.float32)
```

## Discretion Recommendations

### OGG Quality Default: Quality 6 (~192 kbps)
**Recommendation:** Use OGG Vorbis quality level 6 (mapped to soundfile compression_level 0.4).
**Rationale:** Quality 5-6 is the community-recommended "near-CD-quality" range. Quality 6 produces ~192 kbps VBR, which is transparent for most listeners and balances file size with quality. This aligns with the project's ethos of defaulting to high quality (MP3 at 320 CBR, FLAC at max compression). Quality 6 maps to `compression_level=0.4` in libsndfile's inverted scale (0.0=max quality, 1.0=min quality).

### HRTF Dataset: MIT KEMAR (SOFA Format)
**Recommendation:** Bundle the MIT KEMAR dummy-head HRTF dataset in SOFA format.
**Rationale:**
- **Well-established reference:** Used as the standard HRTF reference in hundreds of publications since 1994.
- **Freely available:** Public domain, available in SOFA format from sofacoustics.org.
- **Compact:** SOFA file is ~1.5 MB, easily bundled with the application.
- **Good generic fit:** KEMAR is a standardized dummy head, providing a "median listener" HRTF that works reasonably well for most people without individualized measurement.
- **Rich measurement set:** 710 positions covering -40 to +90 degrees elevation at 44.1 kHz.

Download URL: `http://sofacoustics.org/data/database/mit/`

Alternative considered: CIPIC database (45 subjects, more positions per subject) -- larger files, and the extra subject-specific data isn't needed for a single-listener generative tool.

### Union Slider Handling for Non-Shared Parameters: Zero-Fill
**Recommendation:** When a slider maps to a parameter that a model doesn't have, treat it as zero (neutral position) for that model.
**Rationale:**
- **Zero = mean position:** In the existing PCA-based slider system, position 0 maps to the latent space mean. Treating absent parameters as zero means the model generates from its mean for that dimension.
- **Predictable behavior:** Users can reason about "this slider only affects models A and C" without surprising artifacts.
- **Simple implementation:** No special-case logic needed. Missing parameters simply don't contribute to that model's latent vector.
- **Alternative (skip/interpolate) rejected:** Skipping would change the weight distribution; interpolation from neighboring sliders would be confusing and unpredictable.

### Blend Weight Normalization: Soft Normalization with Dead Zone
**Recommendation:** Auto-normalize weights to sum to 100% using simple proportional normalization. If all weights are at 0%, distribute equally. Allow individual weights to be set to exactly 0% to exclude a model.
**Rationale:**
- Setting one weight to 100% while others are at 0% should produce pure output from that model.
- Proportional normalization is intuitive: if Model A is at 60% and Model B is at 40%, the blend is 60/40 regardless of absolute slider positions.
- The "all zero = equal" fallback prevents a state where no audio is generated.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ffmpeg/pydub for MP3 | lameenc (pure Python LAME) | lameenc 1.0+ (2020) | No system dependency for MP3 encoding |
| soundfile for all formats | soundfile (FLAC/OGG) + lameenc (MP3) | python-soundfile #390 still open | MP3 needs separate encoder for bitrate control |
| MATLAB HRTF files | SOFA format (AES69-2022) | AES standardized 2015, reaffirmed 2022 | Industry-standard spatially-oriented acoustic data format |
| Custom HRTF binary files | sofar Python library | sofar 1.0 (2023) | Clean Python API for SOFA file access |

**Deprecated/outdated:**
- `torchaudio.load()` / `torchaudio.info()`: Removed in torchaudio 2.10 (project already avoids these).
- CIPIC MATLAB format: Original UC Davis hosting offline since 2022. Use SOFA format mirror instead.

## Open Questions

1. **SOFA file bundling strategy**
   - What we know: MIT KEMAR SOFA file is ~1.5 MB, available from sofacoustics.org.
   - What's unclear: Should we bundle it in the Python package (increases install size) or download on first use? The project's `data/` directory pattern suggests bundling is fine.
   - Recommendation: Bundle in `data/hrtf/` alongside other data directories. 1.5 MB is negligible. Download-on-first-use adds complexity and failure modes.

2. **Binaural depth (front-back) implementation**
   - What we know: Width is straightforward (blend between center and binaural). Depth requires selecting different HRTF positions (closer positions have more pronounced high-frequency differences, distant positions are more diffuse).
   - What's unclear: Exact mapping from depth parameter (0.0-1.0) to HRTF position selection and amplitude scaling.
   - Recommendation: Depth controls two things: (1) HRTF position distance if available in the dataset, and (2) a simple proximity effect (bass boost for close, roll-off for far) using a low-shelf filter. Start simple, iterate.

3. **Multi-model pipeline flow**
   - What we know: Up to 4 models, two blending modes, union sliders.
   - What's unclear: Should generation be sequential (model A, then B, then blend) or parallel? Sequential is simpler but 4x slower for audio-domain blending.
   - Recommendation: Sequential generation for v1. The models are small (~3 MB each) and generation of 1-second audio takes <1 second. Parallelism can be added later if needed.

4. **Backward compatibility for stereo_width removal**
   - What we know: `GenerationConfig.stereo_width` and `stereo_mode` are used in saved presets, history entries, and CLI options.
   - What's unclear: How to handle loading old presets/history that reference `stereo_width`.
   - Recommendation: Keep `stereo_width` as a deprecated field in `GenerationConfig` with a migration path. When loading old data, map `stereo_width` to `spatial_width` and `stereo_mode` to `output_mode` ("mid_side" -> "stereo", "dual_seed" -> "stereo", "mono" -> "mono").

## Sources

### Primary (HIGH confidence)
- [python-soundfile docs](https://python-soundfile.readthedocs.io/en/0.13.1/) -- FLAC/OGG write capabilities, compression_level parameter
- [libsndfile FLAC compression mapping](https://github.com/libsndfile/libsndfile/issues/14) -- FLAC level 0-8 maps to float 0.0-1.0
- [libsndfile MP3 bitrate mapping](https://github.com/libsndfile/libsndfile/issues/1008) -- CBR 0.0=320kbps, 1.0=32kbps
- [python-soundfile MP3 bitrate issue #390](https://github.com/bastibe/python-soundfile/issues/390) -- MP3 bitrate control NOT working in python-soundfile
- [lameenc PyPI](https://pypi.org/project/lameenc/) -- Pre-built LAME binaries, set_bit_rate API
- [mutagen docs](https://mutagen.readthedocs.io/en/latest/user/gettingstarted.html) -- ID3, FLAC, OggVorbis tag APIs
- [mutagen ID3 docs](https://mutagen.readthedocs.io/en/latest/user/id3.html) -- TIT2, TPE1, TALB, TXXX frame classes
- [scipy.signal.fftconvolve docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html) -- FFT convolution API
- [SOFA conventions files](https://www.sofaconventions.org/mediawiki/index.php/Files) -- Available HRTF datasets in SOFA format
- [MIT KEMAR HRTF](https://sound.media.mit.edu/resources/KEMAR.html) -- Dataset details, sample rate, measurement positions

### Secondary (MEDIUM confidence)
- [sofar GitHub](https://github.com/pyfar/sofar) -- SOFA reading library for Python
- [OGG Vorbis quality guide (Audacity docs)](https://manual.audacityteam.org/man/ogg_vorbis_export_options.html) -- Quality-to-bitrate mapping
- [3D Audio Panner CIPIC implementation](https://github.com/franciscorotea/3D-Audio-Panner) -- HRTF convolution pattern with scipy
- [Binamix paper](https://arxiv.org/abs/2505.01369v1) -- Recent Python binaural rendering library using SADIE II

### Tertiary (LOW confidence)
- OGG Vorbis quality-to-kbps mapping is approximate and VBR-dependent -- actual bitrate varies with audio content.
- sofar SOFA reading API: confirmed from GitHub README but not verified with Context7.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- soundfile, lameenc, mutagen are well-documented and verified via official docs and GitHub issues
- Architecture: HIGH -- export patterns follow existing project conventions, formats dispatched via enum
- Spatial audio: MEDIUM -- HRTF convolution via scipy.signal.fftconvolve is standard but depth control mapping is custom
- Multi-model blending: MEDIUM -- latent-space blending is standard VAE technique; union slider concept is novel and not externally documented
- Pitfalls: HIGH -- most pitfalls verified via actual GitHub issues (soundfile #390) and project codebase analysis

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (30 days -- stable domain, libraries mature)
