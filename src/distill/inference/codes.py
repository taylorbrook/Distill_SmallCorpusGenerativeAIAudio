"""Encode/decode/preview pipeline for VQ-VAE code inspection.

Provides functions to encode audio files into discrete VQ-VAE code indices,
decode code grids back to audio, and preview individual codebook entries,
time slices, or full rows.  These are the computational backend for the
Phase 16 Codes tab.

Design notes:
- Lazy imports for torch, numpy, load_audio (project pattern from generation.py).
- TYPE_CHECKING imports for heavy modules.
- All audio at 48 kHz internally.
- torch.no_grad() around all inference calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

    from distill.models.persistence import LoadedVQModel


def encode_audio_file(
    audio_path: Path,
    loaded: "LoadedVQModel",
) -> dict:
    """Encode an audio file to VQ-VAE code indices.

    Parameters
    ----------
    audio_path:
        Path to the audio file (WAV, FLAC, MP3, AIFF).
    loaded:
        A loaded VQ-VAE model bundle.

    Returns
    -------
    dict
        Keys: ``indices`` (cpu tensor [1, H*W, num_quantizers]),
        ``spatial_shape`` (H, W), ``mel_shape`` (n_mels, time),
        ``num_quantizers``, ``codebook_size``, ``duration_s``.
    """
    import torch  # noqa: WPS433 -- lazy import

    from distill.audio.io import load_audio  # noqa: WPS433

    # Load and prepare audio
    audio_file = load_audio(Path(audio_path), target_sample_rate=48000)
    waveform = audio_file.waveform  # [channels, samples]

    # Mono mixdown if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert to mel spectrogram
    # waveform_to_mel expects [B, 1, samples]
    mel = loaded.spectrogram.waveform_to_mel(
        waveform.unsqueeze(0).to(loaded.device),
    )  # [1, 1, n_mels, time]

    # Store mel shape BEFORE forward pass (Pitfall 1 from research)
    mel_shape = (mel.shape[2], mel.shape[3])

    # Run forward pass to get code indices
    loaded.model.eval()
    with torch.no_grad():
        _recon, indices, _commit_loss = loaded.model(mel)

    # Capture spatial shape IMMEDIATELY after forward
    spatial_shape = loaded.model._spatial_shape

    # Compute duration from original waveform
    duration_s = waveform.shape[-1] / 48000.0

    return {
        "indices": indices.cpu(),  # [1, H*W, num_quantizers]
        "spatial_shape": spatial_shape,
        "mel_shape": mel_shape,
        "num_quantizers": loaded.model.num_quantizers,
        "codebook_size": loaded.model.codebook_size,
        "duration_s": duration_s,
    }


def decode_code_grid(
    indices: "torch.Tensor",
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
    loaded: "LoadedVQModel",
) -> "np.ndarray":
    """Decode code indices back to audio waveform.

    Parameters
    ----------
    indices:
        Shape ``[1, seq_len, num_quantizers]`` code indices.
    spatial_shape:
        ``(H, W)`` spatial dimensions from the encoder.
    mel_shape:
        ``(n_mels, time)`` original mel spectrogram dimensions.
    loaded:
        A loaded VQ-VAE model bundle.

    Returns
    -------
    np.ndarray
        1-D float32 numpy array at 48 kHz.
    """
    import numpy as np  # noqa: WPS433 -- lazy import
    import torch  # noqa: WPS433 -- lazy import

    loaded.model.eval()
    with torch.no_grad():
        quantized = loaded.model.codes_to_embeddings(
            indices.to(loaded.device), spatial_shape,
        )
        mel = loaded.model.decode(quantized, target_shape=mel_shape)
        wav = loaded.spectrogram.mel_to_waveform(mel)

    return wav.squeeze().cpu().numpy().astype(np.float32)


def preview_single_code(
    level: int,
    code_index: int,
    loaded: "LoadedVQModel",
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
) -> "np.ndarray":
    """Produce audio for a single codebook entry.

    Creates a code grid with zeros everywhere except the target level,
    which is filled with ``code_index`` at every position.

    Parameters
    ----------
    level:
        Quantizer level (0-based).
    code_index:
        Codebook entry index to preview.
    loaded:
        A loaded VQ-VAE model bundle.
    spatial_shape:
        ``(H, W)`` spatial dimensions.
    mel_shape:
        ``(n_mels, time)`` mel spectrogram dimensions.

    Returns
    -------
    np.ndarray
        1-D float32 numpy array at 48 kHz.
    """
    import torch  # noqa: WPS433 -- lazy import

    H, W = spatial_shape
    seq_len = H * W
    num_q = loaded.model.num_quantizers

    # Create zeroed indices tensor
    indices = torch.zeros(1, seq_len, num_q, dtype=torch.long)
    # Set target code at all positions for this level
    indices[0, :, level] = code_index

    return decode_code_grid(indices, spatial_shape, mel_shape, loaded)


def preview_time_slice(
    position: int,
    full_indices: "torch.Tensor",
    loaded: "LoadedVQModel",
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
) -> "np.ndarray":
    """Produce audio for all levels at one time position.

    Extracts the codes at ``position`` and broadcasts them to all positions.

    Parameters
    ----------
    position:
        Time position index (column in the code grid).
    full_indices:
        Shape ``[1, seq_len, num_quantizers]`` full code grid.
    loaded:
        A loaded VQ-VAE model bundle.
    spatial_shape:
        ``(H, W)`` spatial dimensions.
    mel_shape:
        ``(n_mels, time)`` mel spectrogram dimensions.

    Returns
    -------
    np.ndarray
        1-D float32 numpy array at 48 kHz.
    """
    import torch  # noqa: WPS433 -- lazy import

    H, W = spatial_shape
    seq_len = H * W
    num_q = loaded.model.num_quantizers

    # Extract codes at this position
    pos_codes = full_indices[0, position, :]  # [num_quantizers]

    # Broadcast to all positions
    indices = torch.zeros(1, seq_len, num_q, dtype=torch.long)
    indices[0, :, :] = pos_codes.unsqueeze(0).expand(seq_len, -1)

    return decode_code_grid(indices, spatial_shape, mel_shape, loaded)


def play_row_audio(
    level: int,
    full_indices: "torch.Tensor",
    loaded: "LoadedVQModel",
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
) -> "np.ndarray":
    """Concatenate decoded audio for one level across all time positions.

    For each position, creates a single-position preview (the code at
    that level, zeros elsewhere), decodes to audio, and concatenates
    all position audios into one continuous array.

    Parameters
    ----------
    level:
        Quantizer level (0-based).
    full_indices:
        Shape ``[1, seq_len, num_quantizers]`` full code grid.
    loaded:
        A loaded VQ-VAE model bundle.
    spatial_shape:
        ``(H, W)`` spatial dimensions.
    mel_shape:
        ``(n_mels, time)`` mel spectrogram dimensions.

    Returns
    -------
    np.ndarray
        1-D float32 numpy array at 48 kHz.
    """
    import numpy as np  # noqa: WPS433 -- lazy import

    H, W = spatial_shape
    seq_len = H * W
    chunks: list[np.ndarray] = []

    for pos in range(seq_len):
        code_index = int(full_indices[0, pos, level].item())
        chunk = preview_single_code(
            level, code_index, loaded, spatial_shape, mel_shape,
        )
        chunks.append(chunk)

    return np.concatenate(chunks).astype(np.float32)
