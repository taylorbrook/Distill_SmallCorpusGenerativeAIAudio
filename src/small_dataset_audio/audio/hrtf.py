"""HRTF loading from SOFA files and binaural convolution for spatial audio.

Loads Head-Related Transfer Function (HRTF) data from SOFA files, caches
the results, and provides binaural convolution with width and depth controls
for immersive headphone listening.

Design notes:
- Lazy imports for sofar, scipy, numpy, torch/torchaudio (project pattern).
- HRTF data cached by (sofa_path, sample_rate) to avoid repeated I/O.
- Binaural convolution uses scipy.signal.fftconvolve for efficiency.
- Width blends between center and full binaural image.
- Depth applies gentle high-frequency rolloff for distance simulation.

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

# Module-level cache: (sofa_path_str, sample_rate) -> HRTFData
_hrtf_cache: dict[tuple[str, int], "HRTFData"] = {}


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
    azimuth: float = 90.0,
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
        Target source azimuth in degrees (0-360).  Default ``90.0`` (right).
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

    cache_key = (str(sofa_path), target_sample_rate)
    if cache_key in _hrtf_cache:
        return _hrtf_cache[cache_key]

    import sofar  # noqa: WPS433

    logger.info("Loading HRTF from %s", sofa_path)
    sofa_obj = sofar.read_sofa(str(sofa_path))

    # Find nearest source position to requested (azimuth, elevation)
    # SourcePosition is [N, 3] with columns (azimuth, elevation, distance)
    positions = np.array(sofa_obj.SourcePosition)
    target = np.array([azimuth, elevation])
    distances = np.sqrt(np.sum((positions[:, :2] - target) ** 2, axis=1))
    nearest_idx = int(np.argmin(distances))

    actual_az = float(positions[nearest_idx, 0])
    actual_el = float(positions[nearest_idx, 1])
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


def apply_binaural(
    mono: "np.ndarray",
    hrtf: HRTFData,
    width: float = 1.0,
    depth: float = 0.5,
) -> "np.ndarray":
    """Apply binaural spatialization to mono audio using HRTF convolution.

    Parameters
    ----------
    mono : np.ndarray
        1-D mono audio array ``[samples]``.
    hrtf : HRTFData
        Loaded HRTF data with left/right impulse responses.
    width : float
        Spatial width.  ``0.0`` = mono center, ``1.0`` = natural binaural,
        up to ``1.5`` = exaggerated.  Default ``1.0``.
    depth : float
        Front-back depth.  ``0.0`` = close/intimate (full bandwidth),
        ``1.0`` = distant/diffuse (high-frequency rolloff).  Default ``0.5``.

    Returns
    -------
    np.ndarray
        Stereo audio ``[2, samples]`` as float32.
    """
    import numpy as np  # noqa: WPS433
    from scipy.signal import fftconvolve, butter, sosfiltfilt  # noqa: WPS433

    mono = np.asarray(mono, dtype=np.float32).ravel()
    n_samples = len(mono)

    # Convolve with left and right HRIRs
    left = fftconvolve(mono, hrtf.hrir_left, mode="full")[:n_samples]
    right = fftconvolve(mono, hrtf.hrir_right, mode="full")[:n_samples]

    # Width control: blend between center and full binaural
    center = (left + right) * 0.5
    left_out = center + width * (left - center)
    right_out = center + width * (right - center)

    # Depth control: gentle high-frequency rolloff proportional to depth
    # Close (depth=0.0) = no rolloff (full bandwidth)
    # Far (depth=1.0) = cutoff at ~8 kHz
    if depth > 0.01:
        # Cutoff frequency scales from 20 kHz (depth=0) to 8 kHz (depth=1)
        nyquist = hrtf.sample_rate / 2.0
        cutoff_hz = 20_000.0 - depth * 12_000.0  # 20k -> 8k
        # Clamp to valid range for Butterworth (must be < Nyquist)
        cutoff_hz = min(cutoff_hz, nyquist * 0.95)
        cutoff_hz = max(cutoff_hz, 1000.0)

        normalized_cutoff = cutoff_hz / nyquist
        sos = butter(1, normalized_cutoff, btype="low", output="sos")
        left_out = sosfiltfilt(sos, left_out).astype(np.float32)
        right_out = sosfiltfilt(sos, right_out).astype(np.float32)

    stereo = np.stack([left_out, right_out], axis=0).astype(np.float32)
    return stereo


def clear_hrtf_cache() -> None:
    """Clear the module-level HRTF cache."""
    _hrtf_cache.clear()
    logger.debug("HRTF cache cleared")
