"""Audio quality metrics for generated output.

Provides practical quality feedback after each generation:

- **SNR** -- frame-based signal-to-noise ratio in dB.
- **Clipping detection** -- counts clipped samples, consecutive runs,
  and peak value.
- **Quality score** -- combines SNR and clipping into a traffic-light
  rating (green / yellow / red) for immediate user feedback.

Design notes:
- Lazy-imports numpy inside functions (project pattern).
- All functions operate on numpy arrays (CPU-only post-generation).
- Per user decision: quality is practical (SNR + clipping), not academic
  (no spectral coverage or flatness metrics).
"""

from __future__ import annotations


def compute_snr_db(
    audio: "np.ndarray",
    sample_rate: int = 48_000,
    silence_threshold: float = 0.01,
) -> float:
    """Estimate signal-to-noise ratio in decibels.

    Uses frame-based analysis: frames whose RMS exceeds
    *silence_threshold* are classified as signal; the rest as noise.

    Parameters
    ----------
    audio : np.ndarray
        Audio array (1-D or 2-D).  Flattened to 1-D before analysis.
    sample_rate : int
        Sample rate in Hz.  Used to compute the 10 ms frame size.
    silence_threshold : float
        RMS threshold separating signal frames from noise frames.

    Returns
    -------
    float
        SNR in dB.  Returns ``float('inf')`` when no noise frames are
        detected and ``0.0`` when no signal frames are found.
    """
    import numpy as np  # noqa: WPS433

    audio = np.asarray(audio, dtype=np.float32).ravel()

    frame_size = int(0.01 * sample_rate)  # 10 ms frames
    if frame_size == 0:
        frame_size = 1
    num_frames = len(audio) // frame_size

    if num_frames == 0:
        return 0.0

    # Reshape into frames for vectorised RMS
    frames = audio[: num_frames * frame_size].reshape(num_frames, frame_size)
    power = np.mean(frames ** 2, axis=1)
    rms = np.sqrt(power)

    signal_mask = rms > silence_threshold
    noise_mask = ~signal_mask

    signal_count = int(np.sum(signal_mask))
    noise_count = int(np.sum(noise_mask))

    if noise_count == 0:
        return float("inf")
    if signal_count == 0:
        return 0.0

    avg_signal = float(np.mean(power[signal_mask]))
    avg_noise = float(np.mean(power[noise_mask]))

    if avg_noise == 0.0:
        return float("inf")

    return float(10.0 * np.log10(avg_signal / avg_noise))


def detect_clipping(
    audio: "np.ndarray",
    threshold: float = 0.999,
    consecutive_threshold: int = 3,
) -> dict:
    """Detect digital clipping in audio.

    Samples whose absolute value meets or exceeds *threshold* are
    considered clipped.  Consecutive-run detection uses vectorised
    numpy operations (no Python for-loop).

    Parameters
    ----------
    audio : np.ndarray
        Audio array (1-D or 2-D).  Flattened to 1-D before analysis.
    threshold : float
        Absolute sample value at or above which clipping is detected.
    consecutive_threshold : int
        Minimum consecutive clipped samples to consider a clipping event
        (informational; ``has_clipping`` is based on any clipped sample).

    Returns
    -------
    dict
        Keys: ``clipped_samples`` (int), ``clipped_percentage`` (float,
        0-100), ``peak_value`` (float), ``max_consecutive_clipped`` (int),
        ``has_clipping`` (bool).
    """
    import numpy as np  # noqa: WPS433

    audio = np.asarray(audio, dtype=np.float32).ravel()
    abs_audio = np.abs(audio)

    clipped_mask = abs_audio >= threshold
    clipped_count = int(np.sum(clipped_mask))
    total_samples = len(abs_audio)

    # Vectorised max-consecutive-clipped using diff on clipped indices
    max_consecutive = 0
    if clipped_count > 0:
        clipped_indices = np.where(clipped_mask)[0]
        if len(clipped_indices) == 1:
            max_consecutive = 1
        else:
            # Consecutive indices have diff == 1; split runs where diff > 1
            diffs = np.diff(clipped_indices)
            # Run lengths: split at gaps, count elements in each run
            gaps = np.where(diffs > 1)[0]
            # Run boundaries
            starts = np.concatenate([[0], gaps + 1])
            ends = np.concatenate([gaps + 1, [len(clipped_indices)]])
            run_lengths = ends - starts
            max_consecutive = int(np.max(run_lengths))

    return {
        "clipped_samples": clipped_count,
        "clipped_percentage": (clipped_count / total_samples) * 100.0 if total_samples > 0 else 0.0,
        "peak_value": float(np.max(abs_audio)) if total_samples > 0 else 0.0,
        "max_consecutive_clipped": max_consecutive,
        "has_clipping": clipped_count > 0,
    }


def compute_quality_score(
    audio: "np.ndarray",
    sample_rate: int = 48_000,
) -> dict:
    """Compute a combined quality report with traffic-light rating.

    Orchestrates :func:`compute_snr_db` and :func:`detect_clipping`
    into a single assessment.

    Rating thresholds:

    - **green**: SNR > 30 dB AND no clipping.
    - **yellow**: SNR 15--30 dB OR < 0.1 % clipped samples.
    - **red**: SNR < 15 dB OR > 0.1 % clipped samples.

    Parameters
    ----------
    audio : np.ndarray
        Audio array (1-D or 2-D).
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    dict
        Keys: ``snr_db`` (float), ``clipping`` (dict from
        :func:`detect_clipping`), ``rating`` (str), ``rating_reason`` (str).
    """
    snr = compute_snr_db(audio, sample_rate=sample_rate)
    clipping = detect_clipping(audio)

    clipped_pct = clipping["clipped_percentage"]

    # Determine rating -- evaluate red conditions first
    if snr < 15.0 or clipped_pct > 0.1:
        rating = "red"
        reasons = []
        if snr < 15.0:
            reasons.append(f"SNR too low ({snr:.1f} dB < 15 dB)")
        if clipped_pct > 0.1:
            reasons.append(f"Significant clipping ({clipped_pct:.2f}% > 0.1%)")
        rating_reason = "; ".join(reasons)
    elif snr <= 30.0 or (clipping["has_clipping"] and clipped_pct <= 0.1):
        rating = "yellow"
        reasons = []
        if snr <= 30.0:
            reasons.append(f"Moderate SNR ({snr:.1f} dB)")
        if clipping["has_clipping"]:
            reasons.append(f"Minor clipping ({clipped_pct:.3f}%)")
        rating_reason = "; ".join(reasons)
    else:
        rating = "green"
        rating_reason = f"Good quality (SNR {snr:.1f} dB, no clipping)"

    return {
        "snr_db": snr,
        "clipping": clipping,
        "rating": rating,
        "rating_reason": rating_reason,
    }
