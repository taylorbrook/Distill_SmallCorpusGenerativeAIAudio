"""Audio preview generation from VAE decoder.

Generates WAV files from random latent vectors (or reconstructions) so
users can hear the model improve over time.  Previews are listed by epoch
for a scrollable timeline in the UI.

Design notes:
- ``InverseMelScale`` + ``GriffinLim`` run on CPU (project pattern from
  ``audio/spectrogram.py``).
- Per-file try/except: one failed preview does not stop training.
- Peak normalization prevents clipping in 16-bit WAV output.
- Lazy imports for ``torch``, ``soundfile``, ``numpy`` (project pattern).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Naming conventions
_PREVIEW_PATTERN = "preview_epoch*.wav"
_RECON_ORIG_PATTERN = "recon_epoch*_orig_*.wav"
_RECON_RECON_PATTERN = "recon_epoch*_recon_*.wav"


# ---------------------------------------------------------------------------
# Random-sample previews
# ---------------------------------------------------------------------------


def generate_preview(
    model: "torch.nn.Module",
    spectrogram: "AudioSpectrogram",
    output_dir: Path,
    epoch: int,
    device: "torch.device",
    num_samples: int = 1,
    sample_rate: int = 48_000,
) -> list[Path]:
    """Generate WAV preview files from random latent vectors.

    Sets the model to eval mode, generates samples, then restores train mode.
    Each sample is decoded through the VAE decoder and converted to a waveform
    via ``spectrogram.mel_to_waveform`` (CPU-based InverseMelScale + GriffinLim).

    Parameters
    ----------
    model : torch.nn.Module
        The VAE model (must have ``sample`` or ``decode`` method).
    spectrogram : AudioSpectrogram
        Spectrogram converter for mel-to-waveform.
    output_dir : Path
        Directory to save WAV files (created if needed).
    epoch : int
        Current epoch number (used in filename).
    device : torch.device
        Device the model is on.
    num_samples : int
        Number of preview samples to generate (default 1).
    sample_rate : int
        Output sample rate in Hz (default 48000).

    Returns
    -------
    list[Path]
        Paths to successfully saved WAV files.
    """
    import torch  # noqa: WPS433 -- lazy import

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    saved_paths: list[Path] = []

    try:
        with torch.no_grad():
            # Generate mel spectrograms from random latent vectors
            mel_recon = model.sample(num_samples, device)

            # Convert to waveform on CPU (InverseMelScale requirement)
            waveforms = spectrogram.mel_to_waveform(mel_recon.cpu())

            # Save each sample as WAV
            for i in range(waveforms.shape[0]):
                try:
                    wav_path = output_dir / f"preview_epoch{epoch:04d}_{i:02d}.wav"
                    audio = waveforms[i, 0]  # [samples] -- mono
                    _save_wav(audio, wav_path, sample_rate)
                    saved_paths.append(wav_path)
                except Exception:
                    logger.warning(
                        "Failed to save preview %d for epoch %d",
                        i, epoch, exc_info=True,
                    )
    except Exception:
        logger.warning("Failed to generate previews for epoch %d", epoch, exc_info=True)
    finally:
        if was_training:
            model.train()

    if saved_paths:
        logger.info("Saved %d preview(s) for epoch %d", len(saved_paths), epoch)

    return saved_paths


# ---------------------------------------------------------------------------
# Reconstruction previews
# ---------------------------------------------------------------------------


def generate_reconstruction_preview(
    model: "torch.nn.Module",
    spectrogram: "AudioSpectrogram",
    sample_batch: "torch.Tensor",
    output_dir: Path,
    epoch: int,
    device: "torch.device",
    sample_rate: int = 48_000,
) -> list[Path]:
    """Generate original vs. reconstruction WAV pairs for quality monitoring.

    Takes a batch of real mel spectrograms, encodes and decodes through the
    VAE, then saves both original and reconstruction as WAV files.  Only the
    first ``min(2, batch_size)`` items are saved to limit disk usage.

    Parameters
    ----------
    model : torch.nn.Module
        The VAE model (must have ``forward`` returning ``(recon, mu, logvar)``).
    spectrogram : AudioSpectrogram
        Spectrogram converter for mel-to-waveform.
    sample_batch : torch.Tensor
        Batch of mel spectrograms ``[B, 1, n_mels, time]``.
    output_dir : Path
        Directory to save WAV files (created if needed).
    epoch : int
        Current epoch number (used in filename).
    device : torch.device
        Device the model is on.
    sample_rate : int
        Output sample rate in Hz (default 48000).

    Returns
    -------
    list[Path]
        Paths to all successfully saved WAV files (originals + reconstructions).
    """
    import torch  # noqa: WPS433 -- lazy import

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    saved_paths: list[Path] = []

    n_items = min(2, sample_batch.shape[0])

    try:
        with torch.no_grad():
            sample_batch = sample_batch[:n_items].to(device)
            recon, _mu, _logvar = model(sample_batch)

            # Convert both to waveform on CPU
            orig_waveforms = spectrogram.mel_to_waveform(sample_batch.cpu())
            recon_waveforms = spectrogram.mel_to_waveform(recon.cpu())

            for i in range(n_items):
                # Original
                try:
                    orig_path = output_dir / f"recon_epoch{epoch:04d}_orig_{i:02d}.wav"
                    _save_wav(orig_waveforms[i, 0], orig_path, sample_rate)
                    saved_paths.append(orig_path)
                except Exception:
                    logger.warning(
                        "Failed to save original %d for epoch %d",
                        i, epoch, exc_info=True,
                    )

                # Reconstruction
                try:
                    recon_path = output_dir / f"recon_epoch{epoch:04d}_recon_{i:02d}.wav"
                    _save_wav(recon_waveforms[i, 0], recon_path, sample_rate)
                    saved_paths.append(recon_path)
                except Exception:
                    logger.warning(
                        "Failed to save reconstruction %d for epoch %d",
                        i, epoch, exc_info=True,
                    )
    except Exception:
        logger.warning(
            "Failed to generate reconstruction previews for epoch %d",
            epoch, exc_info=True,
        )
    finally:
        if was_training:
            model.train()

    if saved_paths:
        logger.info("Saved %d reconstruction preview(s) for epoch %d", len(saved_paths), epoch)

    return saved_paths


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def list_previews(preview_dir: Path) -> list[dict]:
    """List all preview WAV files sorted by epoch.

    Scans for ``preview_epoch*.wav`` files and returns metadata dicts
    supporting the scrollable timeline UI.

    Parameters
    ----------
    preview_dir : Path
        Directory containing preview WAV files.

    Returns
    -------
    list[dict]
        Sorted by epoch: ``[{epoch, path, filename}, ...]``.
    """
    preview_dir = Path(preview_dir)
    if not preview_dir.exists():
        return []

    results: list[dict] = []
    for wav_path in sorted(preview_dir.glob(_PREVIEW_PATTERN)):
        epoch = _epoch_from_preview_name(wav_path)
        results.append({
            "epoch": epoch,
            "path": wav_path,
            "filename": wav_path.name,
        })

    results.sort(key=lambda p: (p["epoch"], p["filename"]))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VQ-VAE reconstruction previews (v1.1)
# ---------------------------------------------------------------------------


def generate_vqvae_reconstruction_preview(
    model: "torch.nn.Module",
    spectrogram: "AudioSpectrogram",
    sample_batch: "torch.Tensor",
    output_dir: Path,
    epoch: int,
    device: "torch.device",
    sample_rate: int = 48_000,
) -> list[Path]:
    """Generate original vs. reconstruction WAV pairs for VQ-VAE quality monitoring.

    Takes a batch of waveform tensors, converts to mel internally, encodes
    through the VQ-VAE (encode-quantize-decode), then saves both original
    and reconstruction as WAV files.  Only the first ``min(2, batch_size)``
    items are saved to limit disk usage.

    Unlike :func:`generate_reconstruction_preview` which takes mel
    spectrograms directly, this function accepts waveform batches
    ``[B, 1, samples]`` from the DataLoader and handles mel conversion.

    The VQ-VAE forward pass returns ``(recon, indices, commit_loss)``
    instead of the VAE's ``(recon, mu, logvar)``.

    Parameters
    ----------
    model : torch.nn.Module
        The VQ-VAE model (must have ``forward`` returning
        ``(recon, indices, commit_loss)``).
    spectrogram : AudioSpectrogram
        Spectrogram converter for waveform-to-mel and mel-to-waveform.
    sample_batch : torch.Tensor
        Batch of waveform tensors ``[B, 1, samples]`` from the DataLoader.
    output_dir : Path
        Directory to save WAV files (created if needed).
    epoch : int
        Current epoch number (used in filename).
    device : torch.device
        Device the model is on.
    sample_rate : int
        Output sample rate in Hz (default 48000).

    Returns
    -------
    list[Path]
        Paths to all successfully saved WAV files (originals + reconstructions).
    """
    import torch  # noqa: WPS433 -- lazy import

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    saved_paths: list[Path] = []

    n_items = min(2, sample_batch.shape[0])

    try:
        with torch.no_grad():
            # Move waveform to device and convert to mel
            waveform_batch = sample_batch[:n_items].to(device)
            mel_batch = spectrogram.waveform_to_mel(waveform_batch)

            # VQ-VAE forward pass: returns (recon, indices, commit_loss)
            recon, _indices, _commit_loss = model(mel_batch)

            # Convert both to waveform on CPU
            orig_waveforms = spectrogram.mel_to_waveform(mel_batch.cpu())
            recon_waveforms = spectrogram.mel_to_waveform(recon.cpu())

            for i in range(n_items):
                # Original
                try:
                    orig_path = output_dir / f"recon_epoch{epoch:04d}_orig_{i:02d}.wav"
                    _save_wav(orig_waveforms[i, 0], orig_path, sample_rate)
                    saved_paths.append(orig_path)
                except Exception:
                    logger.warning(
                        "Failed to save VQ-VAE original %d for epoch %d",
                        i, epoch, exc_info=True,
                    )

                # Reconstruction
                try:
                    recon_path = output_dir / f"recon_epoch{epoch:04d}_recon_{i:02d}.wav"
                    _save_wav(recon_waveforms[i, 0], recon_path, sample_rate)
                    saved_paths.append(recon_path)
                except Exception:
                    logger.warning(
                        "Failed to save VQ-VAE reconstruction %d for epoch %d",
                        i, epoch, exc_info=True,
                    )
    except Exception:
        logger.warning(
            "Failed to generate VQ-VAE reconstruction previews for epoch %d",
            epoch, exc_info=True,
        )
    finally:
        if was_training:
            model.train()

    if saved_paths:
        logger.info("Saved %d VQ-VAE reconstruction preview(s) for epoch %d", len(saved_paths), epoch)

    return saved_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_wav(
    audio_tensor: "torch.Tensor",
    path: Path,
    sample_rate: int,
) -> None:
    """Peak-normalize a 1-D audio tensor and save as 16-bit WAV."""
    import numpy as np  # noqa: WPS433 -- lazy import
    import soundfile as sf  # noqa: WPS433 -- lazy import

    audio_np = audio_tensor.numpy().astype(np.float32)

    # Peak normalize to prevent clipping
    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak

    sf.write(str(path), audio_np, sample_rate, subtype="PCM_16")


def _epoch_from_preview_name(wav_path: Path) -> int:
    """Extract epoch number from ``preview_epochNNNN_NN.wav``."""
    stem = wav_path.stem  # preview_epoch0042_00
    try:
        # Split on 'epoch', take second part, split on '_', take first part
        epoch_part = stem.split("epoch")[1].split("_")[0]
        return int(epoch_part)
    except (IndexError, ValueError):
        return -1
