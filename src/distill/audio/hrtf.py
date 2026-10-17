"""HRTF loading from SOFA files and binaural convolution for spatial audio.

Loads Head-Related Transfer Function (HRTF) data from SOFA files, caches
the results, and renders stereo for headphones through a virtual speaker
pair.

Design notes:
- Lazy imports for sofar, scipy, numpy, torch/torchaudio (project pattern).
- HRTF data cached by (sofa_path, sample_rate, azimuth, elevation).
- Positions follow the SOFA convention: azimuth counter-clockwise from the
  front, so 90 deg is the listener's LEFT and 270 deg the right.
- Binaural convolution uses scipy.signal.fftconvolve for efficiency.
- Depth applies gentle high-frequency rolloff for distance simulation.

Status: binaural output is hidden in the UI and CLI until a SOFA file is
shipped or downloaded (see below).

HRTF file:
    The MIT KEMAR SOFA file should be placed at:
        <package>/data/hrtf/mit_kemar.sofa

    Download from: https://sofacoustics.org/data/database/mit/
    File: MIT_KEMAR_normal_pinna.sofa (~1.4 MB)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level cache: (sofa_path_str, sample_rate, azimuth, elevation) -> HRTFData
_hrtf_cache: dict[tuple[str, int, float, float], "HRTFData"] = {}

VIRTUAL_SPEAKER_AZIMUTH = 30.0
"""Virtual speakers sit at +/-30 deg, the standard stereo triangle."""


@dataclass
class HRTFData:
    """Loaded HRTF impulse response pair for binaural rendering.

    Attributes
    ----------
    hrir_left : np.ndarray
        Left ear impulse response [filter_length].
    hrir_right : np.ndarray
        Right ear impulse response [filter_length].
    sample_rate : int
        Sample rate of the HRIRs.
    source_azimuth : float
        Azimuth of the selected source position (degrees).
    source_elevation : float
        Elevation of the selected source position (degrees).
    """

    hrir_left: "np.ndarray"
    hrir_right: "np.ndarray"
    sample_rate: int
    source_azimuth: float
    source_elevation: float


def get_default_hrtf_path() -> Path:
    """Return path to the bundled MIT KEMAR SOFA file.

    Returns
    -------
    Path
        Path to ``<package>/data/hrtf/mit_kemar.sofa``.

    Raises
    ------
    FileNotFoundError
        If the SOFA file is not found at the expected location.
    """
    hrtf_path = Path(__file__).parent.parent / "data" / "hrtf" / "mit_kemar.sofa"
    if not hrtf_path.exists():
        raise FileNotFoundError(
            f"MIT KEMAR HRTF file not found at {hrtf_path}. "
            "Download MIT_KEMAR_normal_pinna.sofa from "
            "https://sofacoustics.org/data/database/mit/ "
            "and place it at the above path."
        )
    return hrtf_path


def load_hrtf(
    sofa_path: Path | None = None,
    target_sample_rate: int = 48_000,
    azimuth: float = 0.0,
    elevation: float = 0.0,
) -> HRTFData:
    """Load HRTF data from a SOFA file with caching and optional resampling.

    Parameters
    ----------
    sofa_path : Path or None
        Path to a SOFA file.  If ``None``, uses the bundled MIT KEMAR file.
    target_sample_rate : int
        Desired sample rate for the HRIRs.  Resamples if SOFA rate differs.
    azimuth : float
        Target source azimuth in degrees, counter-clockwise from the front
        (SOFA convention: ``90`` is left, ``270`` is right).  Default ``0``.
    elevation : float
        Target source elevation in degrees.  Default ``0.0`` (ear level).

    Returns
    -------
    HRTFData
        Loaded and optionally resampled HRTF data.

    Raises
    ------
    FileNotFoundError
        If the SOFA file does not exist.
    """
    import numpy as np  # noqa: WPS433

    if sofa_path is None:
        sofa_path = get_default_hrtf_path()

    cache_key = (str(sofa_path), target_sample_rate, float(azimuth), float(elevation))
    if cache_key in _hrtf_cache:
        return _hrtf_cache[cache_key]

    import sofar  # noqa: WPS433

    logger.info("Loading HRTF from %s", sofa_path)
    sofa_obj = sofar.read_sofa(str(sofa_path))

    # Find nearest source position to requested (azimuth, elevation)
    positions = np.array(sofa_obj.SourcePosition, dtype=np.float64)
    if str(getattr(sofa_obj, "SourcePosition_Type", "spherical")).lower() == "cartesian":
        directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        az_el = np.stack([
            np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) % 360.0,
            np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))),
        ], axis=1)
    else:
        # Spherical: columns (azimuth, elevation, distance) in degrees
        directions = _unit_vectors(positions[:, 0], positions[:, 1])
        az_el = positions[:, :2]
    target = _unit_vectors(np.array([azimuth]), np.array([elevation]))[0]
    # Largest dot product = smallest angle on the sphere, which wraps
    # around 0/360 deg (a plain azimuth difference does not)
    nearest_idx = int(np.argmax(directions @ target))

    actual_az = float(az_el[nearest_idx, 0])
    actual_el = float(az_el[nearest_idx, 1])
    logger.debug(
        "Nearest HRTF position: azimuth=%.1f, elevation=%.1f (requested %.1f, %.1f)",
        actual_az,
        actual_el,
        azimuth,
        elevation,
    )

    # Extract left and right HRIRs
    hrir_left = np.array(sofa_obj.Data_IR[nearest_idx, 0, :], dtype=np.float32)
    hrir_right = np.array(sofa_obj.Data_IR[nearest_idx, 1, :], dtype=np.float32)

    # Resample if SOFA sample rate differs from target
    sofa_sr = int(sofa_obj.Data_SamplingRate)
    if sofa_sr != target_sample_rate:
        import torch  # noqa: WPS433
        import torchaudio  # noqa: WPS433

        logger.debug("Resampling HRIR from %d to %d Hz", sofa_sr, target_sample_rate)
        resampler = torchaudio.transforms.Resample(sofa_sr, target_sample_rate)
        hrir_left = resampler(torch.from_numpy(hrir_left).unsqueeze(0)).squeeze(0).numpy()
        hrir_right = resampler(torch.from_numpy(hrir_right).unsqueeze(0)).squeeze(0).numpy()

    hrtf_data = HRTFData(
        hrir_left=hrir_left,
        hrir_right=hrir_right,
        sample_rate=target_sample_rate,
        source_azimuth=actual_az,
        source_elevation=actual_el,
    )

    _hrtf_cache[cache_key] = hrtf_data
    logger.info(
        "HRTF loaded: %d-tap filter at %d Hz (azimuth=%.1f, elevation=%.1f)",
        len(hrir_left),
        target_sample_rate,
        actual_az,
        actual_el,
    )
    return hrtf_data


def _unit_vectors(azimuth_deg: "np.ndarray", elevation_deg: "np.ndarray") -> "np.ndarray":
    """Direction vectors ``[N, 3]`` for SOFA spherical angles in degrees."""
    import numpy as np  # noqa: WPS433

    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)


def load_speaker_pair(
    sofa_path: Path | None = None,
    target_sample_rate: int = 48_000,
) -> tuple[HRTFData, HRTFData]:
    """HRTFs for a virtual stereo speaker pair at +/-30 deg (left, right)."""
    left = load_hrtf(sofa_path, target_sample_rate, azimuth=VIRTUAL_SPEAKER_AZIMUTH)
    right = load_hrtf(sofa_path, target_sample_rate, azimuth=360.0 - VIRTUAL_SPEAKER_AZIMUTH)
    return left, right


def apply_binaural(
    stereo: "np.ndarray",
    left_speaker: HRTFData,
    right_speaker: HRTFData,
    depth: float = 0.5,
) -> "np.ndarray":
    """Render stereo audio for headphones through a virtual speaker pair.

    Each channel is played from its own virtual speaker; each ear hears
    both speakers through the measured HRIRs.  A mono signal (equal
    channels) therefore stays centred instead of sitting in one ear, and
    the stereo width of the input carries over to the binaural image.

    Parameters
    ----------
    stereo : np.ndarray
        ``[2, samples]`` audio (left, right).  1-D input is treated as
        mono and sent to both speakers.
    left_speaker, right_speaker : HRTFData
        HRIRs for the left and right virtual speaker positions
        (see :func:`load_speaker_pair`).
    depth : float
        Front-back depth.  ``0.0`` = close/intimate (full bandwidth),
        ``1.0`` = distant/diffuse (high-frequency rolloff).  Default ``0.5``.

    Returns
    -------
    np.ndarray
        Binaural audio ``[2, samples]`` (left ear, right ear) as float32.
    """
    import numpy as np  # noqa: WPS433
    from scipy.signal import fftconvolve, butter, sosfiltfilt  # noqa: WPS433

    stereo = np.asarray(stereo, dtype=np.float32)
    if stereo.ndim == 1:
        stereo = np.stack([stereo, stereo])
    left_in, right_in = stereo[0], stereo[1]
    n_samples = stereo.shape[1]

    def _conv(signal: "np.ndarray", hrir: "np.ndarray") -> "np.ndarray":
        return fftconvolve(signal, hrir, mode="full")[:n_samples]

    left_out = _conv(left_in, left_speaker.hrir_left) + _conv(right_in, right_speaker.hrir_left)
    right_out = _conv(left_in, left_speaker.hrir_right) + _conv(right_in, right_speaker.hrir_right)

    # Depth control: gentle high-frequency rolloff proportional to depth
    # Close (depth=0.0) = no rolloff (full bandwidth)
    # Far (depth=1.0) = cutoff at ~8 kHz
    if depth > 0.01:
        # Cutoff frequency scales from 20 kHz (depth=0) to 8 kHz (depth=1)
        nyquist = left_speaker.sample_rate / 2.0
        cutoff_hz = 20_000.0 - depth * 12_000.0  # 20k -> 8k
        # Clamp to valid range for Butterworth (must be < Nyquist)
        cutoff_hz = min(cutoff_hz, nyquist * 0.95)
        cutoff_hz = max(cutoff_hz, 1000.0)

        normalized_cutoff = cutoff_hz / nyquist
        sos = butter(1, normalized_cutoff, btype="low", output="sos")
        left_out = sosfiltfilt(sos, left_out).astype(np.float32)
        right_out = sosfiltfilt(sos, right_out).astype(np.float32)

    return np.stack([left_out, right_out], axis=0).astype(np.float32)


def clear_hrtf_cache() -> None:
    """Clear the module-level HRTF cache."""
    _hrtf_cache.clear()
    logger.debug("HRTF cache cleared")
