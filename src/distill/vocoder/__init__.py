"""Vocoder package for mel-to-waveform conversion.

Public API
----------
VocoderBase : Abstract base class for vocoder implementations.
get_vocoder : Factory function to obtain a ready-to-use vocoder.
"""

from __future__ import annotations

from distill.vocoder.base import VocoderBase

__all__ = ["VocoderBase", "get_vocoder"]


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
    raise NotImplementedError(
        f"Vocoder type '{vocoder_type}' not yet implemented. "
        "BigVGAN implementation coming in plan 02."
    )
