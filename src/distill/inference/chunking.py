"""Chunk-based audio generation with crossfade and latent interpolation.

For audio longer than one chunk, multiple chunks are generated from the
VAE and concatenated.  Two concatenation modes are supported:

- **Crossfade:** Independent chunks decoded to waveform, then combined via
  Hann-windowed overlap-add.  Reliable default.
- **Latent interpolation:** Anchor latent vectors connected by SLERP
  interpolation, producing smoothly evolving sound.  Experimental.

Design notes:
- Lazy imports for ``torch`` and ``numpy`` (project pattern).
- Chunks processed one at a time to limit memory (research open question #4).
- ``model.decode(z, target_shape=mel_shape)`` passes mel shape to decoder.
- ``spectrogram.mel_to_waveform()`` forces CPU (existing InverseMelScale
  pattern from ``audio/spectrogram.py``).
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
# Crossfade
# ---------------------------------------------------------------------------


def crossfade_chunks(
    chunks: "list[np.ndarray]",
    overlap_samples: int = 2400,
) -> "np.ndarray":
    """Concatenate audio chunks with Hann-windowed overlap-add crossfade.

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
    """Generate audio by decoding independent chunks and crossfading.

    Each chunk gets an independent latent vector.  Decoded waveforms are
    concatenated via Hann-windowed overlap-add crossfade.

    Parameters
    ----------
    model : ConvVAE
        Trained VAE model.
    spectrogram : AudioSpectrogram
        Spectrogram converter (for mel-to-waveform).
    num_chunks : int
        Number of audio chunks to generate.
    device : torch.device
        Device the model is on.
    seed : int | None
        Random seed for reproducibility (``None`` for random).
    chunk_samples : int
        Number of audio samples per chunk (default 48000 = 1 s at 48 kHz).
    overlap_samples : int
        Overlap for crossfade in samples (default 2400 = 50 ms at 48 kHz).

    Returns
    -------
    np.ndarray
        Concatenated audio as float32.
    """
    import torch  # noqa: WPS433 -- lazy import
    import numpy as np  # noqa: WPS433 -- lazy import

    mel_shape = spectrogram.get_mel_shape(chunk_samples)
    z_vectors = _sample_latent_vectors(model, num_chunks, device, seed)

    was_training = model.training
    model.eval()

    waveforms: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for z in z_vectors:
                # Ensure decoder is initialised
                if model.decoder.fc is None:
                    n_mels, time_frames = mel_shape
                    pad_h = (16 - n_mels % 16) % 16
                    pad_w = (16 - time_frames % 16) % 16
                    spatial = ((n_mels + pad_h) // 16, (time_frames + pad_w) // 16)
                    model.decoder._init_linear(spatial)

                # Decode latent vector to mel spectrogram
                mel = model.decode(z, target_shape=mel_shape)

                # Convert mel to waveform on CPU (InverseMelScale requirement)
                wav = spectrogram.mel_to_waveform(mel.cpu())
                waveforms.append(wav.squeeze().numpy().astype(np.float32))
    finally:
        if was_training:
            model.train()

    return crossfade_chunks(waveforms, overlap_samples)


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
    """Generate audio via SLERP interpolation between latent anchor vectors.

    Anchor vectors are generated at random, then connected by spherical
    interpolation.  Each interpolated latent is decoded to a waveform and
    all waveforms are concatenated sequentially (no crossfade needed --
    interpolation ensures smooth transitions).

    Parameters
    ----------
    model : ConvVAE
        Trained VAE model.
    spectrogram : AudioSpectrogram
        Spectrogram converter (for mel-to-waveform).
    num_chunks : int
        Number of anchor latent vectors.
    device : torch.device
        Device the model is on.
    seed : int | None
        Random seed for reproducibility (``None`` for random).
    chunk_samples : int
        Audio samples per decoded chunk (default 48000 = 1 s at 48 kHz).
    steps_between : int
        Number of intermediate interpolation steps between each pair of
        anchors (default 10).

    Returns
    -------
    np.ndarray
        Concatenated audio as float32.
    """
    import torch  # noqa: WPS433 -- lazy import
    import numpy as np  # noqa: WPS433 -- lazy import

    mel_shape = spectrogram.get_mel_shape(chunk_samples)
    z_anchors = _sample_latent_vectors(model, num_chunks, device, seed)

    # Build list of all latent vectors: anchors + interpolated points
    all_z: list[torch.Tensor] = []
    for i in range(len(z_anchors) - 1):
        for t_val in torch.linspace(0, 1, steps_between, device=device):
            z_interp = slerp(z_anchors[i].squeeze(0), z_anchors[i + 1].squeeze(0), t_val.item())
            all_z.append(z_interp.unsqueeze(0))
    # Append final anchor (not included by linspace ending at 1.0 of last pair)
    all_z.append(z_anchors[-1])

    was_training = model.training
    model.eval()

    waveform_parts: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for z in all_z:
                # Ensure decoder is initialised
                if model.decoder.fc is None:
                    n_mels, time_frames = mel_shape
                    pad_h = (16 - n_mels % 16) % 16
                    pad_w = (16 - time_frames % 16) % 16
                    spatial = ((n_mels + pad_h) // 16, (time_frames + pad_w) // 16)
                    model.decoder._init_linear(spatial)

                mel = model.decode(z, target_shape=mel_shape)
                wav = spectrogram.mel_to_waveform(mel.cpu())
                waveform_parts.append(wav.squeeze().numpy().astype(np.float32))
    finally:
        if was_training:
            model.train()

    return np.concatenate(waveform_parts).astype(np.float32)
