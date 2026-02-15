"""Audio I/O abstraction layer using soundfile.

All file reading goes through this module, isolating the I/O library
choice from the rest of the codebase.  If we later switch to TorchCodec
(once FFmpeg issues are resolved on macOS), only this module changes.

Design notes:
- Uses soundfile (libsndfile bundled in pip wheel, no FFmpeg needed)
- Does NOT use torchaudio.load() (requires TorchCodec/FFmpeg in 2.10)
- Does NOT use torchaudio.info() (removed in 2.10)
- Heavy dependencies (torch, torchaudio) are imported inside function
  bodies, matching Phase 1 lazy-import pattern (hardware/device.py,
  validation/environment.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS: set[str] = {"wav", "aiff", "mp3", "flac"}
"""Lowercase file extensions accepted by the audio pipeline."""

DEFAULT_SAMPLE_RATE: int = 48_000
"""Project baseline sample rate (professional audio production standard)."""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AudioMetadata:
    """Metadata for an audio file, extracted without loading the waveform."""

    path: Path
    sample_rate: int
    num_channels: int
    num_frames: int
    duration_seconds: float
    format: str    # e.g. "WAV", "FLAC", "MP3", "AIFF"
    subtype: str   # e.g. "PCM_24", "PCM_16", "FLOAT"


@dataclass
class AudioFile:
    """Loaded audio file with waveform tensor and metadata.

    ``waveform`` is always a float32 tensor with shape
    ``[channels, samples]``, normalised to [-1, 1].
    """

    waveform: "torch.Tensor"  # noqa: F821 -- lazy import
    sample_rate: int
    metadata: AudioMetadata


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_supported_format(path: str | Path) -> bool:
    """Return ``True`` if *path*'s extension is in :data:`SUPPORTED_FORMATS`."""
    return Path(path).suffix.lstrip(".").lower() in SUPPORTED_FORMATS


def get_metadata(path: Path) -> AudioMetadata:
    """Read audio file metadata without loading waveform data.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not in :data:`SUPPORTED_FORMATS`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not is_supported_format(path):
        raise ValueError(
            f"Unsupported audio format '{path.suffix}' for {path.name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

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


# Cache resamplers keyed by (orig_freq, new_freq) to avoid re-creating
# transform objects on every call.
_resamplers: dict[tuple[int, int], object] = {}


def load_audio(path: Path, target_sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioFile:
    """Load an audio file and return an :class:`AudioFile`.

    Always returns a float32 tensor with shape ``[channels, samples]``.
    Resamples to *target_sample_rate* if the source differs using a
    lazily imported ``torchaudio.transforms.Resample``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the format is unsupported.
    """
    import torch  # noqa: WPS433 -- lazy import

    path = Path(path)
    metadata = get_metadata(path)  # validates existence + format

    # soundfile returns [samples, channels]; always_2d ensures mono -> (N, 1)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # -> [channels, samples]

    if sr != target_sample_rate:
        from torchaudio.transforms import Resample  # noqa: WPS433

        key = (sr, target_sample_rate)
        if key not in _resamplers:
            _resamplers[key] = Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = _resamplers[key](waveform)

    return AudioFile(
        waveform=waveform,
        sample_rate=target_sample_rate,
        metadata=metadata,
    )


def check_file_integrity(path: Path) -> tuple[bool, str]:
    """Check whether an audio file can be read without errors.

    Tries ``sf.info()`` (header) then ``sf.read()`` of the first 1024
    frames (data).  Returns ``(True, "OK")`` on success or
    ``(False, description)`` on any error.
    """
    try:
        info = sf.info(str(path))
        if info.frames == 0:
            return False, f"Empty audio file (0 frames): {path.name}"
        # Read a small chunk to verify data integrity
        data, _sr = sf.read(
            str(path),
            frames=min(info.frames, 1024),
            dtype="float32",
        )
        if data.size == 0:
            return False, f"No audio data readable: {path.name}"
        return True, "OK"
    except Exception as exc:
        return False, (
            f"Cannot read file: {path.name} ({type(exc).__name__}: {exc})"
        )
