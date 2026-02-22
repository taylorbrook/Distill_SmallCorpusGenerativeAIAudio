"""Abstract vocoder interface for mel-to-waveform conversion.

Defines the contract that all vocoder implementations must satisfy.
Currently planned implementations: BigVGAN (universal, Phase 12) and
HiFi-GAN V2 (per-model, Phase 16).

Torch is imported inside method signatures via TYPE_CHECKING to allow
this module to be imported for introspection even if torch is not yet
available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class VocoderBase(ABC):
    """Abstract vocoder interface for mel-to-waveform conversion.

    All vocoders accept VAE-format mel spectrograms (log1p, HTK, 48kHz)
    and handle internal conversion to their expected format.
    """

    @abstractmethod
    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        Parameters
        ----------
        mel : torch.Tensor
            VAE output mel spectrogram in log1p format.
            Shape: [B, 1, n_mels, time]

        Returns
        -------
        torch.Tensor
            Waveform at vocoder's native sample rate.
            Shape: [B, 1, samples]
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native output sample rate of this vocoder."""
        ...

    @abstractmethod
    def to(self, device: torch.device) -> VocoderBase:
        """Move vocoder to device. Returns self for chaining."""
        ...
