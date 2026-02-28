"""HiFi-GAN V2 per-model vocoder package.

Public API
----------
HiFiGANConfig : Configuration dataclass for HiFi-GAN V2.
HiFiGANGenerator : V2 generator (mel -> waveform).
MultiPeriodDiscriminator : Multi-period discriminator (MPD).
MultiScaleDiscriminator : Multi-scale discriminator (MSD).
generator_loss : Least-squares GAN generator loss.
discriminator_loss : Least-squares GAN discriminator loss.
feature_loss : L1 feature matching loss.
"""

from __future__ import annotations

__all__ = [
    "HiFiGANConfig",
    "HiFiGANGenerator",
    "MultiPeriodDiscriminator",
    "MultiScaleDiscriminator",
    "generator_loss",
    "discriminator_loss",
    "feature_loss",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading torch at package import time."""
    if name == "HiFiGANConfig":
        from distill.vocoder.hifigan.config import HiFiGANConfig

        return HiFiGANConfig
    if name == "HiFiGANGenerator":
        from distill.vocoder.hifigan.generator import HiFiGANGenerator

        return HiFiGANGenerator
    if name == "MultiPeriodDiscriminator":
        from distill.vocoder.hifigan.discriminator import MultiPeriodDiscriminator

        return MultiPeriodDiscriminator
    if name == "MultiScaleDiscriminator":
        from distill.vocoder.hifigan.discriminator import MultiScaleDiscriminator

        return MultiScaleDiscriminator
    if name in ("generator_loss", "discriminator_loss", "feature_loss"):
        from distill.vocoder.hifigan import losses

        return getattr(losses, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
