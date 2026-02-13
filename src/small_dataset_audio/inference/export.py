"""WAV export with configurable sample rate, bit depth, and sidecar JSON.

Exports generated audio as WAV files with configurable format (sample rate,
bit depth) alongside a sidecar JSON file containing full generation metadata.

Design notes:
- Lazy import for ``soundfile`` (project pattern).
- Sidecar JSON written alongside the WAV (no embedded tags per user decision).
- ``write_sidecar_json`` should be called before ``export_wav`` in the
  pipeline to avoid metadata loss if WAV write fails (research pitfall #6).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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
