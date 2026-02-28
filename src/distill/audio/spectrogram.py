"""Mel spectrogram representation layer.

Converts raw waveforms to normalized log-mel spectrograms (forward-only).
All audio entering the VAE passes through this module, ensuring
consistent mel parameters (n_fft, n_mels, hop_length, sample_rate).

Mel-to-waveform reconstruction is handled by neural vocoders (BigVGAN
universal or per-model HiFi-GAN) -- this module is intentionally
forward-only.

Design notes:
- Lazy-imports torchaudio.transforms inside ``__init__`` (project pattern).
- ``log1p`` / ``expm1`` normalisation compresses dynamic range and handles zeros.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpectrogramConfig:
    """Mel spectrogram parameters -- must be consistent across training and inference.

    Default values target professional audio production (48 kHz, 128 mels).
    One second of audio at 48 kHz produces a 128x94 mel spectrogram.
    """

    sample_rate: int = 48_000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None  # defaults to sample_rate / 2
    power: float = 2.0


class AudioSpectrogram:
    """Convert waveforms to normalized log-mel spectrograms (forward-only).

    Instantiate once and reuse across batches -- internal torchaudio transforms
    allocate buffers that are expensive to re-create.

    Parameters
    ----------
    config : SpectrogramConfig | None
        Mel parameters.  ``None`` uses project defaults (48 kHz, 128 mels).
    """

    def __init__(self, config: SpectrogramConfig | None = None) -> None:
        import torch  # noqa: WPS433 -- lazy import
        from torchaudio.transforms import MelSpectrogram

        self.config = config or SpectrogramConfig()
        c = self.config

        self.mel_transform = MelSpectrogram(
            sample_rate=c.sample_rate,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            n_mels=c.n_mels,
            f_min=c.f_min,
            f_max=c.f_max,
            power=c.power,
        )

        # Keep a reference to torch for use in methods without re-importing
        self._torch = torch

    def to(self, device: "torch.device") -> "AudioSpectrogram":
        """Move the mel transform to the given device.

        Returns ``self`` for chaining.
        """
        self.mel_transform = self.mel_transform.to(device)
        return self

    # ------------------------------------------------------------------
    # Forward: waveform -> mel
    # ------------------------------------------------------------------

    def waveform_to_mel(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Convert waveform to normalised log-mel spectrogram.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``[B, 1, samples]`` -- batch of mono waveforms.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, n_mels, time]`` -- normalised log-mel spectrogram.
        """
        torch = self._torch
        mel = self.mel_transform(waveform.squeeze(1))  # [B, n_mels, time]
        mel_log = torch.log1p(mel)  # log(1 + x), handles zeros gracefully
        return mel_log.unsqueeze(1)  # [B, 1, n_mels, time]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_mel_shape(self, num_samples: int) -> tuple[int, int]:
        """Return ``(n_mels, time_frames)`` for a given number of audio samples.

        Useful for sizing the VAE architecture before any data is loaded.
        """
        c = self.config
        # torchaudio pads by n_fft // 2 on each side, then applies hop_length
        time_frames = num_samples // c.hop_length + 1
        return (c.n_mels, time_frames)
