"""Stereo processing for generated audio.

Provides two stereo creation methods:

1. **Mid-side widening** -- creates stereo from mono using a Haas-effect
   delay to produce left/right differentiation, then adjusts the
   stereo image via mid-side width control.
2. **Dual-seed combiner** -- stacks two independently generated mono
   signals (from different random seeds) as left and right channels.

Also includes peak normalisation with professional headroom (-1 dBFS).

Design notes:
- Lazy-imports numpy inside functions (project pattern).
- All functions operate on numpy arrays (CPU-only post-generation).
- Stereo arrays use shape ``[2, samples]`` (channels-first).
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)


def apply_mid_side_widening(
    mono: "np.ndarray",
    width: float = 0.7,
    sample_rate: int = 48_000,
    haas_delay_ms: float = 15.0,
) -> "np.ndarray":
    """Create stereo from mono using Haas effect and mid-side width control.

    The Haas effect introduces a small delay on the right channel to create
    a perceived spatial separation.  Mid-side decomposition then controls the
    stereo width: 0.0 collapses to mono, 1.0 is natural width, and values
    above 1.0 exaggerate the stereo image.

    Parameters
    ----------
    mono : np.ndarray
        1-D mono audio array ``[samples]``.
    width : float
        Stereo width, clamped to ``[0.0, 1.5]``.  Default ``0.7``.
    sample_rate : int
        Sample rate in Hz.  Used to compute delay in samples.
    haas_delay_ms : float
        Haas-effect delay for the right channel in milliseconds.

    Returns
    -------
    np.ndarray
        Stereo audio ``[2, samples]`` as float32.
    """
    import numpy as np  # noqa: WPS433

    # Validate and clamp width
    if width < 0.0 or width > 1.5:
        warnings.warn(
            f"Stereo width {width} outside [0.0, 1.5] range; clamping.",
            stacklevel=2,
        )
        width = float(np.clip(width, 0.0, 1.5))

    mono = np.asarray(mono, dtype=np.float32).ravel()
    delay_samples = int(haas_delay_ms * sample_rate / 1000.0)

    # Create left (original) and right (delayed) channels
    left = mono.copy()
    right = np.zeros_like(mono)

    if delay_samples > 0 and delay_samples < len(mono):
        right[delay_samples:] = mono[: -delay_samples]
    elif delay_samples == 0:
        right[:] = mono

    # Mid-side decomposition
    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    # Apply width control
    new_left = mid + width * side
    new_right = mid - width * side

    return np.stack([new_left, new_right], axis=0).astype(np.float32)


def create_dual_seed_stereo(
    left_mono: "np.ndarray",
    right_mono: "np.ndarray",
) -> "np.ndarray":
    """Combine two mono signals into a stereo pair.

    The two signals are typically generated with different random seeds
    by the generation pipeline to produce natural stereo variation.

    Parameters
    ----------
    left_mono : np.ndarray
        1-D mono array for the left channel.
    right_mono : np.ndarray
        1-D mono array for the right channel.

    Returns
    -------
    np.ndarray
        Stereo audio ``[2, samples]`` as float32.  If inputs differ in
        length the shorter channel is zero-padded.
    """
    import numpy as np  # noqa: WPS433

    left = np.asarray(left_mono, dtype=np.float32).ravel()
    right = np.asarray(right_mono, dtype=np.float32).ravel()

    # Pad shorter channel with zeros
    if len(left) != len(right):
        max_len = max(len(left), len(right))
        if len(left) < max_len:
            left = np.pad(left, (0, max_len - len(left)))
        if len(right) < max_len:
            right = np.pad(right, (0, max_len - len(right)))

    return np.stack([left, right], axis=0).astype(np.float32)


def peak_normalize(
    audio: "np.ndarray",
    target_peak: float = 0.891,
) -> "np.ndarray":
    """Normalize peak amplitude to a target level.

    Default target is -1 dBFS (0.891) which leaves professional headroom
    for downstream DAW processing.  Do **not** normalise to 1.0.

    Parameters
    ----------
    audio : np.ndarray
        Audio array, either 1-D ``[samples]`` or 2-D ``[channels, samples]``.
    target_peak : float
        Target peak absolute value.  Default ``0.891`` (-1 dBFS).

    Returns
    -------
    np.ndarray
        Peak-normalised audio as float32 with the same shape as input.
    """
    import numpy as np  # noqa: WPS433

    audio = np.asarray(audio, dtype=np.float32)
    current_peak = np.max(np.abs(audio))

    if current_peak == 0.0:
        return audio

    return (audio * (target_peak / current_peak)).astype(np.float32)
