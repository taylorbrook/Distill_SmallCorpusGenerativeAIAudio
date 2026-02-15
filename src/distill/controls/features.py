"""Audio feature extraction for latent space analysis.

Computes scalar acoustic features from raw waveforms for correlating
PCA components with perceptually meaningful audio properties.

Uses numpy and scipy only (no librosa) to avoid numba dependency.
All features are scalar summaries of the entire waveform (not frame-level).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "spectral_centroid",
    "rms_energy",
    "zero_crossing_rate",
    "spectral_rolloff",
    "spectral_flatness",
]
"""Ordered list of audio feature names computed by this module."""


def compute_audio_features(
    waveform: "np.ndarray",
    sample_rate: int = 48_000,
) -> dict[str, float]:
    """Compute scalar audio features from a 1-D waveform.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 audio waveform.
    sample_rate : int
        Sample rate in Hz (default 48000).

    Returns
    -------
    dict[str, float]
        Maps feature name to scalar float value.
    """
    import numpy as np  # noqa: WPS433 -- lazy import
    from scipy.stats import gmean  # noqa: WPS433 -- lazy import

    waveform = np.asarray(waveform, dtype=np.float32).ravel()

    # --- Shared spectral data ---
    magnitude = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), 1.0 / sample_rate)
    power = magnitude ** 2
    mag_sum = magnitude.sum()

    # --- Spectral centroid ---
    if mag_sum > 0:
        spectral_centroid = float(np.sum(freqs * magnitude) / mag_sum)
    else:
        spectral_centroid = 0.0

    # --- RMS energy ---
    rms_energy = float(np.sqrt(np.mean(waveform ** 2)))

    # --- Zero crossing rate ---
    sign_changes = np.abs(np.diff(np.signbit(waveform).astype(int)))
    zero_crossing_rate = float(np.mean(sign_changes))

    # --- Spectral rolloff (85% energy) ---
    if mag_sum > 0:
        cumulative = np.cumsum(magnitude)
        rolloff_threshold = 0.85 * cumulative[-1]
        rolloff_idx = np.searchsorted(cumulative, rolloff_threshold)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        spectral_rolloff = float(freqs[rolloff_idx])
    else:
        spectral_rolloff = 0.0

    # --- Spectral flatness (geometric mean / arithmetic mean of power) ---
    eps = 1e-10
    power_safe = power + eps
    arithmetic_mean = float(np.mean(power_safe))
    if arithmetic_mean > eps:
        geometric_mean = float(gmean(power_safe))
        spectral_flatness = geometric_mean / arithmetic_mean
    else:
        spectral_flatness = 0.0

    return {
        "spectral_centroid": spectral_centroid,
        "rms_energy": rms_energy,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_rolloff": spectral_rolloff,
        "spectral_flatness": spectral_flatness,
    }


def compute_features_batch(
    waveforms: "list[np.ndarray]",
    sample_rate: int = 48_000,
) -> "list[dict[str, float]]":
    """Compute audio features for a list of waveforms.

    Per-item try/except isolates failures -- a broken waveform returns
    a dict of zeros rather than crashing the batch.

    Parameters
    ----------
    waveforms : list[np.ndarray]
        List of 1-D float32 audio waveforms.
    sample_rate : int
        Sample rate in Hz (default 48000).

    Returns
    -------
    list[dict[str, float]]
        One feature dict per waveform (zeros on failure).
    """
    results: list[dict[str, float]] = []
    zeros = {name: 0.0 for name in FEATURE_NAMES}

    for i, waveform in enumerate(waveforms):
        try:
            results.append(compute_audio_features(waveform, sample_rate))
        except Exception:
            logger.warning("Feature extraction failed for waveform %d, returning zeros", i)
            results.append(dict(zeros))

    return results
