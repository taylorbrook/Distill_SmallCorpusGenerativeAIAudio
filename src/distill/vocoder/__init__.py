"""Vocoder package for mel-to-waveform conversion.

Public API
----------
VocoderBase : Abstract base class for vocoder implementations.
BigVGANVocoder : BigVGAN-v2 universal vocoder (122M params, 44.1kHz).
MelAdapter : Converts VAE mel format to BigVGAN mel format.
get_vocoder : Factory function to obtain a ready-to-use vocoder.
"""

from __future__ import annotations

from distill.vocoder.base import VocoderBase

__all__ = ["VocoderBase", "BigVGANVocoder", "MelAdapter", "get_vocoder"]


def __getattr__(name: str):
    """Lazy import for BigVGANVocoder and MelAdapter to avoid loading torch at package import."""
    if name == "BigVGANVocoder":
        from distill.vocoder.bigvgan_vocoder import BigVGANVocoder

        return BigVGANVocoder
    if name == "MelAdapter":
        from distill.vocoder.mel_adapter import MelAdapter

        return MelAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_vocoder(
    vocoder_type: str = "bigvgan", device: str = "auto"
) -> VocoderBase:
    """Get a vocoder instance by type.

    Parameters
    ----------
    vocoder_type : str
        One of "bigvgan" (universal) or "hifigan" (per-model, Phase 16).
    device : str
        Device preference: "auto", "cuda", "mps", or "cpu".

    Returns
    -------
    VocoderBase
        Ready-to-use vocoder instance on the selected device.
    """
    if vocoder_type == "bigvgan":
        # Lazy import to avoid loading torch at module import time
        from distill.vocoder.bigvgan_vocoder import BigVGANVocoder

        return BigVGANVocoder(device=device)
    elif vocoder_type == "hifigan":
        raise NotImplementedError(
            "Per-model HiFi-GAN vocoder is planned for Phase 16."
        )
    else:
        raise ValueError(
            f"Unknown vocoder type: {vocoder_type!r}. Use 'bigvgan' or 'hifigan'."
        )
