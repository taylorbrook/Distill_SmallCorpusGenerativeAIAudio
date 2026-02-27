"""Continuous audio generation via overlap-add mel synthesis.

For audio longer than one chunk, latent vectors are decoded to
overlapping mel windows and combined via Hann-windowed overlap-add
for continuous audio.  Overlap-add mel synthesis produces seamless
audio with no chunk boundaries, clicks, or windowing artifacts.

Two generation modes:

- **Crossfade:** Random anchor vectors interpolated via SLERP, decoded
  through continuous overlap-add synthesis.
- **Latent interpolation:** Same approach with denser anchor sampling.

Design notes:
- Lazy imports for ``torch`` and ``numpy`` (project pattern).
- ``model.decode(z, target_shape=mel_shape)`` passes mel shape to decoder.
- 50% Hann overlap satisfies the COLA (Constant Overlap-Add) condition,
  guaranteeing amplitude consistency across the synthesized mel.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# SLERP
# ---------------------------------------------------------------------------


def slerp(
    v0: "torch.Tensor",
    v1: "torch.Tensor",
    t: float,
    dot_threshold: float = 0.9995,
) -> "torch.Tensor":
    """Spherical linear interpolation between two latent vectors.

    Interpolates on the unit sphere, preserving vector magnitudes.
    Falls back to ``torch.lerp`` when vectors are nearly parallel
    (dot product exceeds *dot_threshold*).

    Parameters
    ----------
    v0 : torch.Tensor
        Start vector (any shape, treated as flat).
    v1 : torch.Tensor
        End vector (same shape as *v0*).
    t : float
        Interpolation factor in ``[0, 1]``.
    dot_threshold : float
        When the cosine similarity exceeds this value, fall back to
        linear interpolation to avoid numerical issues (default 0.9995).

    Returns
    -------
    torch.Tensor
        Interpolated vector, same shape as *v0*.
    """
    import torch  # noqa: WPS433 -- lazy import

    # Flatten for computation, restore shape at end
    original_shape = v0.shape
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    # Normalise to unit vectors
    v0_norm = v0_flat / torch.linalg.norm(v0_flat)
    v1_norm = v1_flat / torch.linalg.norm(v1_flat)

    # Compute cosine similarity
    dot = torch.sum(v0_norm * v1_norm).clamp(-1.0, 1.0)

    # Fall back to linear interpolation when nearly parallel
    if torch.abs(dot) > dot_threshold:
        return torch.lerp(v0_flat, v1_flat, t).reshape(original_shape)

    # SLERP: interpolate on the unit sphere, scale by original magnitudes
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0

    result = s0 * v0_flat + s1 * v1_flat
    return result.reshape(original_shape)


# ---------------------------------------------------------------------------
# Waveform crossfade (legacy, kept for backward compat)
# ---------------------------------------------------------------------------


def crossfade_chunks(
    chunks: "list[np.ndarray]",
    overlap_samples: int = 2400,
) -> "np.ndarray":
    """Concatenate audio chunks with Hann-windowed overlap-add crossfade.

    Legacy function kept for backward compatibility.  New code should
    use :func:`synthesize_continuous_mel` instead.

    Parameters
    ----------
    chunks : list[np.ndarray]
        List of 1-D float32 audio arrays (each the same length).
    overlap_samples : int
        Number of samples to overlap between adjacent chunks.
        Default 2400 (50 ms at 48 kHz).

    Returns
    -------
    np.ndarray
        Concatenated audio as float32.
    """
    import numpy as np  # noqa: WPS433 -- lazy import

    if len(chunks) == 0:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32)

    # Create Hann crossfade window
    window = np.hanning(2 * overlap_samples)
    fade_out = window[:overlap_samples]
    fade_in = window[overlap_samples:]

    # Calculate total output length
    chunk_len = len(chunks[0])
    total_length = chunk_len + (len(chunks) - 1) * (chunk_len - overlap_samples)
    output = np.zeros(total_length, dtype=np.float32)

    pos = 0
    for i, chunk in enumerate(chunks):
        chunk = chunk.astype(np.float32)
        if i == 0:
            output[pos : pos + chunk_len] = chunk
        else:
            # Apply fade-out to existing audio in overlap region
            output[pos : pos + overlap_samples] *= fade_out
            # Apply fade-in to new chunk's overlap region
            chunk_faded = chunk.copy()
            chunk_faded[:overlap_samples] *= fade_in
            # Overlap-add
            output[pos : pos + overlap_samples] += chunk_faded[:overlap_samples]
            output[pos + overlap_samples : pos + chunk_len] = chunk_faded[overlap_samples:]
        pos += chunk_len - overlap_samples

    # Trim to actual content length
    actual_length = pos + overlap_samples
    return output[:actual_length]


# ---------------------------------------------------------------------------
# Trajectory interpolation
# ---------------------------------------------------------------------------


def _interpolate_trajectory(
    anchors: "list[torch.Tensor]",
    num_steps: int,
) -> "list[torch.Tensor]":
    """Build a dense latent trajectory by SLERP-interpolating between anchors.

    Parameters
    ----------
    anchors : list[torch.Tensor]
        Anchor latent vectors, each shape ``[1, latent_dim]``.
    num_steps : int
        Total number of trajectory points to produce.

    Returns
    -------
    list[torch.Tensor]
        Dense trajectory of ``num_steps`` vectors, each ``[1, latent_dim]``.
    """
    if len(anchors) == 1 or num_steps <= 1:
        return [anchors[0]] * max(num_steps, 1)

    num_segments = len(anchors) - 1
    trajectory: list[torch.Tensor] = []

    for step_idx in range(num_steps):
        t_global = step_idx / max(num_steps - 1, 1)
        segment_float = t_global * num_segments
        segment_idx = min(int(segment_float), num_segments - 1)
        t_local = segment_float - segment_idx

        z = slerp(
            anchors[segment_idx].squeeze(0),
            anchors[segment_idx + 1].squeeze(0),
            t_local,
        )
        trajectory.append(z.unsqueeze(0))

    return trajectory


# ---------------------------------------------------------------------------
# Continuous overlap-add mel synthesis
# ---------------------------------------------------------------------------


def synthesize_continuous_mel(
    model: "ConvVAE",
    spectrogram: "AudioSpectrogram",
    latent_trajectory: "list[torch.Tensor]",
    chunk_samples: int = 48_000,
) -> "torch.Tensor":
    """Synthesize a continuous mel spectrogram via overlap-add decoding.

    Each latent vector in the trajectory is decoded to a mel chunk,
    Hann-windowed, and accumulated with 50% overlap.  The Hann window
    at 50% hop satisfies the COLA condition, producing a smooth
    continuous mel spectrogram with no chunk boundaries or artifacts.

    Parameters
    ----------
    model : ConvVAE
        Trained VAE model.
    spectrogram : AudioSpectrogram
        Spectrogram converter (for ``get_mel_shape``).
    latent_trajectory : list[torch.Tensor]
        Dense trajectory of latent vectors, each ``[1, latent_dim]``.
    chunk_samples : int
        Number of audio samples per decoded chunk (default 48000 = 1 s).

    Returns
    -------
    torch.Tensor
        Continuous mel spectrogram, shape ``[1, 1, n_mels, total_frames]``.
    """
    import torch  # noqa: WPS433 -- lazy import

    mel_shape = spectrogram.get_mel_shape(chunk_samples)
    n_mels, chunk_frames = mel_shape
    hop_frames = chunk_frames // 2  # 50% overlap (COLA for Hann)
    num_steps = len(latent_trajectory)

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            # Single step: no overlap needed
            if num_steps == 1:
                if model.decoder.fc is None:
                    pad_h = (16 - n_mels % 16) % 16
                    pad_w = (16 - chunk_frames % 16) % 16
                    spatial = (
                        (n_mels + pad_h) // 16,
                        (chunk_frames + pad_w) // 16,
                    )
                    model.decoder._init_linear(spatial)
                return model.decode(
                    latent_trajectory[0], target_shape=mel_shape
                ).cpu()

            total_frames = (num_steps - 1) * hop_frames + chunk_frames
            window = torch.hann_window(chunk_frames)
            output = torch.zeros(n_mels, total_frames)
            norm = torch.zeros(total_frames)

            for i, z in enumerate(latent_trajectory):
                if model.decoder.fc is None:
                    pad_h = (16 - n_mels % 16) % 16
                    pad_w = (16 - chunk_frames % 16) % 16
                    spatial = (
                        (n_mels + pad_h) // 16,
                        (chunk_frames + pad_w) // 16,
                    )
                    model.decoder._init_linear(spatial)

                mel = model.decode(z, target_shape=mel_shape)
                mel_2d = mel.squeeze(0).squeeze(0).cpu()

                start = i * hop_frames
                end = start + chunk_frames
                output[:, start:end] += mel_2d * window.unsqueeze(0)
                norm[start:end] += window

            norm = norm.clamp(min=1e-8)
            output = output / norm.unsqueeze(0)

    finally:
        if was_training:
            model.train()

    return output.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Latent vector sampling
# ---------------------------------------------------------------------------


def _sample_latent_vectors(
    model: "ConvVAE",
    num_vectors: int,
    device: "torch.device",
    seed: int | None,
) -> "list[torch.Tensor]":
    """Generate random latent vectors from the VAE's prior (standard normal).

    Parameters
    ----------
    model : ConvVAE
        VAE model (used to read ``latent_dim``).
    num_vectors : int
        Number of latent vectors to generate.
    device : torch.device
        Device to place tensors on.
    seed : int | None
        If not ``None``, set ``torch.manual_seed`` for reproducibility.

    Returns
    -------
    list[torch.Tensor]
        Each tensor has shape ``[1, latent_dim]`` on the specified device.
    """
    import torch  # noqa: WPS433 -- lazy import

    if seed is not None:
        torch.manual_seed(seed)

    latent_dim = model.latent_dim
    return [
        torch.randn(1, latent_dim, device=device)
        for _ in range(num_vectors)
    ]


# ---------------------------------------------------------------------------
# Decode step calculation
# ---------------------------------------------------------------------------


def _compute_num_decode_steps(
    spectrogram: "AudioSpectrogram",
    num_chunks: int,
    chunk_samples: int,
) -> tuple[int, int, int]:
    """Compute the number of overlap-add decode steps for a target duration.

    Parameters
    ----------
    spectrogram : AudioSpectrogram
        Spectrogram converter.
    num_chunks : int
        Number of original 1-second chunks (i.e. target seconds).
    chunk_samples : int
        Audio samples per decoded chunk.

    Returns
    -------
    tuple[int, int, int]
        ``(num_steps, chunk_frames, hop_frames)``
    """
    import math  # noqa: WPS433

    mel_shape = spectrogram.get_mel_shape(chunk_samples)
    _, chunk_frames = mel_shape
    hop_frames = chunk_frames // 2

    total_target_samples = num_chunks * chunk_samples
    total_frames_needed = spectrogram.get_mel_shape(total_target_samples)[1]
    num_steps = max(
        1,
        math.ceil((total_frames_needed - chunk_frames) / hop_frames) + 1,
    )
    return num_steps, chunk_frames, hop_frames


# ---------------------------------------------------------------------------
# Generation: crossfade mode
# ---------------------------------------------------------------------------


def generate_chunks_crossfade(
    model: "ConvVAE",
    spectrogram: "AudioSpectrogram",
    num_chunks: int,
    device: "torch.device",
    seed: int | None,
    chunk_samples: int = 48_000,
    overlap_samples: int = 2400,
) -> "np.ndarray":
    """Generate audio via continuous overlap-add synthesis.

    Random anchor vectors are SLERP-interpolated to produce a dense
    latent trajectory, then decoded via 50%-overlap Hann-windowed
    overlap-add for seamless, click-free audio.

    Parameters
    ----------
    model : ConvVAE
        Trained VAE model.
    spectrogram : AudioSpectrogram
        Spectrogram converter (for mel-to-waveform).
    num_chunks : int
        Target duration in chunks (each chunk_samples long).
    device : torch.device
        Device the model is on.
    seed : int | None
        Random seed for reproducibility (``None`` for random).
    chunk_samples : int
        Number of audio samples per chunk (default 48000 = 1 s at 48 kHz).
    overlap_samples : int
        Kept for API compatibility (overlap is now 50% mel frames).

    Returns
    -------
    np.ndarray
        Audio as float32.
    """
    import numpy as np  # noqa: WPS433 -- lazy import

    num_steps, _, _ = _compute_num_decode_steps(
        spectrogram, num_chunks, chunk_samples,
    )

    # Random anchors + dense SLERP trajectory
    num_anchors = max(2, num_chunks)
    z_anchors = _sample_latent_vectors(model, num_anchors, device, seed)
    trajectory = _interpolate_trajectory(z_anchors, num_steps)

    combined_mel = synthesize_continuous_mel(
        model, spectrogram, trajectory, chunk_samples,
    )
    # TODO(Phase 16): Replace with complex_mel_to_waveform ISTFT path
    #   wav = spectrogram.mel_to_waveform(combined_mel)
    #   return wav.squeeze().numpy().astype(np.float32)
    raise NotImplementedError(
        "mel_to_waveform removed (v1.0). Phase 16 will wire complex_mel_to_waveform."
    )


# ---------------------------------------------------------------------------
# Generation: latent interpolation mode
# ---------------------------------------------------------------------------


def generate_chunks_latent_interp(
    model: "ConvVAE",
    spectrogram: "AudioSpectrogram",
    num_chunks: int,
    device: "torch.device",
    seed: int | None,
    chunk_samples: int = 48_000,
    steps_between: int = 10,
) -> "np.ndarray":
    """Generate audio via SLERP interpolation with continuous synthesis.

    Anchor vectors are generated at random, then connected by dense
    SLERP interpolation.  Decoded via overlap-add for seamless audio.

    Parameters
    ----------
    model : ConvVAE
        Trained VAE model.
    spectrogram : AudioSpectrogram
        Spectrogram converter (for mel-to-waveform).
    num_chunks : int
        Number of anchor latent vectors / target duration in seconds.
    device : torch.device
        Device the model is on.
    seed : int | None
        Random seed for reproducibility (``None`` for random).
    chunk_samples : int
        Audio samples per decoded chunk (default 48000 = 1 s at 48 kHz).
    steps_between : int
        Kept for API compatibility.

    Returns
    -------
    np.ndarray
        Audio as float32.
    """
    import numpy as np  # noqa: WPS433 -- lazy import

    num_steps, _, _ = _compute_num_decode_steps(
        spectrogram, num_chunks, chunk_samples,
    )

    # More anchors for exploration mode
    num_anchors = max(2, num_chunks)
    z_anchors = _sample_latent_vectors(model, num_anchors, device, seed)
    trajectory = _interpolate_trajectory(z_anchors, num_steps)

    combined_mel = synthesize_continuous_mel(
        model, spectrogram, trajectory, chunk_samples,
    )
    # TODO(Phase 16): Replace with complex_mel_to_waveform ISTFT path
    #   wav = spectrogram.mel_to_waveform(combined_mel)
    #   return wav.squeeze().numpy().astype(np.float32)
    raise NotImplementedError(
        "mel_to_waveform removed (v1.0). Phase 16 will wire complex_mel_to_waveform."
    )
