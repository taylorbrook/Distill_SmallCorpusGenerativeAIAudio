"""HiFi-GAN V2 inference wrapper implementing VocoderBase.

Provides :class:`HiFiGANVocoder` which loads a trained per-model
generator from a ``vocoder_state`` dict and produces waveforms from
VAE-format mel spectrograms.

The generator is trained on the VAE's own mel format (log1p, HTK,
48kHz) so no MelAdapter is needed -- unlike BigVGAN which requires
format conversion.

Usage::

    from distill.vocoder.hifigan import HiFiGANVocoder

    vocoder = HiFiGANVocoder(loaded_model.vocoder_state, device="auto")
    wav = vocoder.mel_to_waveform(mel)  # [B, 1, 128, T] -> [B, 1, samples]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from distill.vocoder.base import VocoderBase

if TYPE_CHECKING:
    pass


class HiFiGANVocoder(VocoderBase):
    """Per-model HiFi-GAN V2 vocoder for inference.

    Loads a trained generator from a ``vocoder_state`` dict (as stored
    in ``.distillgan`` model files) and provides the standard
    :class:`VocoderBase` interface.

    The generator accepts VAE mel spectrograms directly (log1p
    normalized, HTK filterbank, 48kHz parameters). No MelAdapter or
    format conversion is required because the generator was trained on
    exactly this mel format.

    Parameters
    ----------
    vocoder_state : dict
        Vocoder state dict containing ``"config"``,
        ``"generator_state_dict"``, and ``"training_metadata"``.
    device : str
        Device preference: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """

    def __init__(self, vocoder_state: dict, device: str = "auto") -> None:
        from distill.hardware.device import select_device
        from distill.vocoder.hifigan.config import HiFiGANConfig
        from distill.vocoder.hifigan.generator import HiFiGANGenerator

        # Extract config and build generator
        config = HiFiGANConfig(**vocoder_state["config"])
        self._generator = HiFiGANGenerator(config)
        self._generator.load_state_dict(vocoder_state["generator_state_dict"])
        self._generator.eval()
        self._generator.remove_weight_norm()  # Fuse for inference speed
        self._sample_rate = config.sample_rate

        # Resolve and move to device
        self._device = select_device(device)
        self._generator.to(self._device)

        # Store training metadata for diagnostics
        self._training_metadata = vocoder_state.get("training_metadata", {})

    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert VAE mel spectrogram to waveform.

        Parameters
        ----------
        mel : torch.Tensor
            VAE output mel spectrogram in log1p format.
            Shape: ``[B, 1, 128, T]``

        Returns
        -------
        torch.Tensor
            Generated waveform at 48kHz.
            Shape: ``[B, 1, samples]`` where ``samples = T * 512``.

        Notes
        -----
        The input mel is in log1p format (as produced by
        :meth:`AudioSpectrogram.waveform_to_mel`). The generator was
        trained on exactly this format, so no conversion is needed.

        The log1p mel is undone via ``expm1`` to recover linear power
        mel before passing to the generator, which expects linear input.
        """
        mel = mel.to(self._device)

        # [B, 1, 128, T] -> [B, 128, T]
        mel_input = mel.squeeze(1)

        # Undo log1p: expm1(clamp(x, min=0)) -> linear power mel
        mel_input = torch.expm1(mel_input.clamp(min=0.0))

        with torch.inference_mode():
            wav = self._generator(mel_input)  # [B, 1, T * hop_size]

        return wav

    @property
    def sample_rate(self) -> int:
        """Native output sample rate (48000 Hz)."""
        return self._sample_rate

    def to(self, device: torch.device) -> HiFiGANVocoder:
        """Move vocoder to device. Returns self for chaining."""
        self._generator.to(device)
        self._device = device
        return self

    @property
    def training_epochs(self) -> int:
        """Number of epochs the generator was trained for."""
        return self._training_metadata.get("epochs", 0)
