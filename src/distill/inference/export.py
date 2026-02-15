"""Multi-format audio export with configurable encoding and sidecar JSON.

Exports generated audio as WAV, MP3, FLAC, or OGG Vorbis files with
configurable format parameters alongside a sidecar JSON file containing
full generation metadata. MP3/FLAC/OGG files also get embedded metadata tags
via :mod:`distill.audio.metadata`.

Design notes:
- Lazy imports for ``soundfile``, ``numpy``, ``lameenc`` (project pattern).
- Sidecar JSON written for ALL formats (complements embedded tags).
- ``write_sidecar_json`` should be called before audio encoding in the
  pipeline to avoid metadata loss if encoding fails (research pitfall #6).
"""

from __future__ import annotations

import enum
import json
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# ExportFormat enum
# ---------------------------------------------------------------------------


class ExportFormat(enum.Enum):
    """Supported audio export formats."""

    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


FORMAT_EXTENSIONS: dict[ExportFormat, str] = {
    ExportFormat.WAV: ".wav",
    ExportFormat.MP3: ".mp3",
    ExportFormat.FLAC: ".flac",
    ExportFormat.OGG: ".ogg",
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIT_DEPTH_MAP: dict[str, str] = {
    "16-bit": "PCM_16",
    "24-bit": "PCM_24",
    "32-bit float": "FLOAT",
}

SAMPLE_RATE_OPTIONS: tuple[int, ...] = (44_100, 48_000, 96_000)


# ---------------------------------------------------------------------------
# WAV export
# ---------------------------------------------------------------------------


def export_wav(
    audio: "np.ndarray",
    path: Path,
    sample_rate: int = 48_000,
    bit_depth: str = "24-bit",
) -> Path:
    """Export audio as a WAV file with configurable format.

    Parameters
    ----------
    audio : np.ndarray
        Audio array, either 1-D ``[samples]`` (mono) or 2-D
        ``[channels, samples]`` (stereo).
    path : Path
        Output WAV file path.
    sample_rate : int
        Sample rate in Hz (default 48000).
    bit_depth : str
        Bit depth string, one of ``"16-bit"``, ``"24-bit"``, or
        ``"32-bit float"`` (default ``"24-bit"``).

    Returns
    -------
    Path
        The output path (for chaining).

    Raises
    ------
    ValueError
        If *bit_depth* is not a valid option.
    """
    import numpy as np  # noqa: WPS433 -- lazy import
    import soundfile as sf  # noqa: WPS433 -- lazy import

    if bit_depth not in BIT_DEPTH_MAP:
        raise ValueError(
            f"Invalid bit_depth {bit_depth!r}. "
            f"Valid options: {list(BIT_DEPTH_MAP.keys())}"
        )

    subtype = BIT_DEPTH_MAP[bit_depth]
    audio_data = np.asarray(audio, dtype=np.float32)

    # soundfile expects [samples, channels] for multi-channel audio
    if audio_data.ndim == 2:
        audio_data = audio_data.T  # [channels, samples] -> [samples, channels]

    path = Path(path)
    sf.write(str(path), audio_data, sample_rate, subtype=subtype)

    return path


# ---------------------------------------------------------------------------
# Sidecar JSON metadata
# ---------------------------------------------------------------------------


def write_sidecar_json(
    wav_path: Path,
    model_name: str,
    generation_config: dict,
    seed: int | None,
    quality_metrics: dict,
    duration_s: float,
    sample_rate: int = 48_000,
    bit_depth: str = "24-bit",
    channels: int = 1,
) -> Path:
    """Write sidecar JSON alongside a WAV export with full generation metadata.

    The sidecar JSON is written *before* the WAV file in the generation
    pipeline to avoid metadata loss if the WAV write fails (research
    pitfall #6).

    Parameters
    ----------
    wav_path : Path
        Path to the WAV file (sidecar will be ``wav_path.with_suffix(".json")``).
    model_name : str
        Name or identifier of the model that generated the audio.
    generation_config : dict
        Generation configuration as a dictionary.
    seed : int | None
        Random seed used for generation.
    quality_metrics : dict
        Quality score dictionary from :func:`compute_quality_score`.
    duration_s : float
        Duration of the generated audio in seconds.
    sample_rate : int
        Sample rate of the exported WAV.
    bit_depth : str
        Bit depth of the exported WAV.
    channels : int
        Number of audio channels (1 for mono, 2 for stereo).

    Returns
    -------
    Path
        The sidecar JSON path.
    """
    wav_path = Path(wav_path)
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
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "channels": channels,
            "duration_s": duration_s,
        },
    }

    sidecar_path.write_text(json.dumps(metadata, indent=2, default=str))

    return sidecar_path


# ---------------------------------------------------------------------------
# MP3 export (lameenc)
# ---------------------------------------------------------------------------


def export_mp3(
    audio: "np.ndarray",
    path: Path,
    sample_rate: int = 48_000,
    bitrate: int = 320,
) -> Path:
    """Export audio as MP3 at a constant bit rate via lameenc.

    Parameters
    ----------
    audio : np.ndarray
        Audio array, either 1-D ``[samples]`` (mono) or 2-D
        ``[channels, samples]`` (stereo, shape ``[2, samples]``).
    path : Path
        Output MP3 file path.
    sample_rate : int
        Sample rate in Hz (default 48000).
    bitrate : int
        CBR bit rate in kbps (default 320).

    Returns
    -------
    Path
        The output path.
    """
    import lameenc  # noqa: WPS433 -- lazy import
    import numpy as np  # noqa: WPS433 -- lazy import

    audio_data = np.asarray(audio, dtype=np.float32)

    # Determine channel count
    if audio_data.ndim == 2:
        channels = audio_data.shape[0]
        # [channels, samples] -> [samples, channels] then flatten (interleaved)
        audio_data = audio_data.T.flatten()
    else:
        channels = 1

    # Convert float32 [-1, 1] to int16 PCM bytes
    audio_data = np.clip(audio_data, -1.0, 1.0)
    pcm_int16 = (audio_data * 32767).astype(np.int16)
    pcm_bytes = pcm_int16.tobytes()

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)  # Highest quality encoding

    mp3_data = encoder.encode(pcm_bytes) + encoder.flush()

    path = Path(path)
    path.write_bytes(mp3_data)

    return path


# ---------------------------------------------------------------------------
# FLAC export (soundfile)
# ---------------------------------------------------------------------------


def export_flac(
    audio: "np.ndarray",
    path: Path,
    sample_rate: int = 48_000,
    compression_level: int = 8,
) -> Path:
    """Export audio as FLAC with configurable compression level.

    Parameters
    ----------
    audio : np.ndarray
        Audio array, either 1-D ``[samples]`` (mono) or 2-D
        ``[channels, samples]`` (stereo).
    path : Path
        Output FLAC file path.
    sample_rate : int
        Sample rate in Hz (default 48000).
    compression_level : int
        FLAC compression level 0-8, where 8 is maximum compression
        (default 8).

    Returns
    -------
    Path
        The output path.
    """
    import numpy as np  # noqa: WPS433 -- lazy import
    import soundfile as sf  # noqa: WPS433 -- lazy import

    audio_data = np.asarray(audio, dtype=np.float32)

    # soundfile expects [samples, channels] for multi-channel audio
    if audio_data.ndim == 2:
        audio_data = audio_data.T  # [channels, samples] -> [samples, channels]

    path = Path(path)
    sf.write(
        str(path),
        audio_data,
        sample_rate,
        format="FLAC",
        subtype="PCM_24",
    )

    return path


# ---------------------------------------------------------------------------
# OGG Vorbis export (soundfile)
# ---------------------------------------------------------------------------


def export_ogg(
    audio: "np.ndarray",
    path: Path,
    sample_rate: int = 48_000,
    quality: float = 0.6,
) -> Path:
    """Export audio as OGG Vorbis with configurable quality.

    Parameters
    ----------
    audio : np.ndarray
        Audio array, either 1-D ``[samples]`` (mono) or 2-D
        ``[channels, samples]`` (stereo).
    path : Path
        Output OGG file path.
    sample_rate : int
        Sample rate in Hz (default 48000).
    quality : float
        Vorbis quality from 0.0 (lowest) to 1.0 (highest).
        Default 0.6 produces ~192 kbps VBR.

    Returns
    -------
    Path
        The output path.
    """
    import numpy as np  # noqa: WPS433 -- lazy import
    import soundfile as sf  # noqa: WPS433 -- lazy import

    audio_data = np.asarray(audio, dtype=np.float32)

    # soundfile expects [samples, channels] for multi-channel audio
    if audio_data.ndim == 2:
        audio_data = audio_data.T  # [channels, samples] -> [samples, channels]

    path = Path(path)
    sf.write(
        str(path),
        audio_data,
        sample_rate,
        format="OGG",
        subtype="VORBIS",
    )

    return path


# ---------------------------------------------------------------------------
# Unified export dispatcher
# ---------------------------------------------------------------------------

_FORMAT_DISPATCHERS: dict[ExportFormat, str] = {
    ExportFormat.WAV: "export_wav",
    ExportFormat.MP3: "export_mp3",
    ExportFormat.FLAC: "export_flac",
    ExportFormat.OGG: "export_ogg",
}


def export_audio(
    audio: "np.ndarray",
    path: Path,
    sample_rate: int = 48_000,
    format: ExportFormat = ExportFormat.WAV,
    bit_depth: str = "24-bit",
    metadata: dict | None = None,
) -> Path:
    """Export audio in the specified format with optional metadata embedding.

    Dispatches to the format-specific encoder, then embeds metadata tags
    (for formats that support them) and writes a sidecar JSON.

    Parameters
    ----------
    audio : np.ndarray
        Audio array, either 1-D ``[samples]`` (mono) or 2-D
        ``[channels, samples]`` (stereo).
    path : Path
        Output file path (extension will be used as-is).
    sample_rate : int
        Sample rate in Hz (default 48000).
    format : ExportFormat
        Target format (default WAV).
    bit_depth : str
        Bit depth for WAV export (ignored for other formats).
    metadata : dict | None
        Metadata dict for tag embedding. If *None*, no tags are embedded.
        Expected keys: ``artist``, ``album``, ``title``, ``seed``,
        ``model_name``, ``preset_name``.

    Returns
    -------
    Path
        The output path.
    """
    path = Path(path)

    # Dispatch to format-specific encoder
    if format == ExportFormat.WAV:
        export_wav(audio, path, sample_rate=sample_rate, bit_depth=bit_depth)
    elif format == ExportFormat.MP3:
        export_mp3(audio, path, sample_rate=sample_rate)
    elif format == ExportFormat.FLAC:
        export_flac(audio, path, sample_rate=sample_rate)
    elif format == ExportFormat.OGG:
        export_ogg(audio, path, sample_rate=sample_rate)
    else:
        raise ValueError(f"Unsupported export format: {format!r}")

    # Embed metadata tags for formats that support them
    if metadata is not None:
        from distill.audio.metadata import embed_metadata  # noqa: WPS433

        embed_metadata(path, format, metadata)

    return path
