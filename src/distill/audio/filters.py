"""Anti-aliasing filters for generated audio.

Applies a Butterworth low-pass filter to remove frequencies above the
audible range (default 20 kHz) from generated audio.  Used after
GriffinLim output and before sample rate conversion.

Design notes:
- Lazy imports for ``numpy`` and ``scipy`` (project pattern).
- ``sosfiltfilt`` provides zero-phase filtering (no phase distortion).
- Second-order sections (SOS) representation avoids numerical instability
  with high-order filters.
"""

from __future__ import annotations


def apply_anti_alias_filter(
    audio: "np.ndarray",
    sample_rate: int,
    cutoff_hz: float = 20_000.0,
    order: int = 8,
) -> "np.ndarray":
    """Apply Butterworth low-pass filter for anti-aliasing.

    Removes frequencies above ``cutoff_hz`` using an Nth-order Butterworth
    filter with zero-phase filtering (``sosfiltfilt``).  Handles both 1-D
    ``[samples]`` and 2-D ``[channels, samples]`` input shapes via
    ``axis=-1``.

    Parameters
    ----------
    audio : np.ndarray
        Audio samples as float32.  Shape ``[samples]`` (mono) or
        ``[channels, samples]`` (multi-channel).
    sample_rate : int
        Sample rate in Hz (e.g. 48000).
    cutoff_hz : float
        Low-pass cutoff frequency in Hz (default 20000).
    order : int
        Butterworth filter order (default 8, giving ~48 dB/octave rolloff).

    Returns
    -------
    np.ndarray
        Filtered audio as float32, same shape as input.
    """
    import numpy as np  # noqa: WPS433 -- lazy import
    from scipy.signal import butter, sosfiltfilt  # noqa: WPS433 -- lazy import

    nyquist = sample_rate / 2.0

    # If cutoff is at or above Nyquist, no filtering needed
    if cutoff_hz >= nyquist:
        return audio.astype(np.float32)

    # Clamp normalized cutoff below Nyquist to prevent filter instability
    normalized_cutoff = min(cutoff_hz, nyquist * 0.95) / nyquist

    # Design Butterworth low-pass filter using second-order sections
    sos = butter(order, normalized_cutoff, btype="low", output="sos")

    # Apply zero-phase filtering along the last axis (works for 1-D and 2-D)
    filtered = sosfiltfilt(sos, audio, axis=-1)

    return filtered.astype(np.float32)
