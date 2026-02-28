"""HiFi-GAN V2 configuration adapted for 128-band 48kHz mel spectrograms.

Provides sensible defaults for per-model vocoder training on small
datasets (5-50 files). The critical constraint is that the product of
``upsample_rates`` must equal ``hop_size`` (512).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HiFiGANConfig:
    """HiFi-GAN V2 configuration adapted for this project's mel parameters.

    The generator upsamples mel frames to waveform samples. The product
    of ``upsample_rates`` **must** equal ``hop_size`` so that each mel
    frame maps to exactly ``hop_size`` waveform samples.

    Default upsample_rates [8, 8, 4, 2] -> product = 512 = hop_size.
    """

    # Generator architecture
    resblock_type: int = 1  # ResBlock1 (HiFi-GAN V2)
    upsample_rates: list[int] = field(
        default_factory=lambda: [8, 8, 4, 2],
    )
    upsample_kernel_sizes: list[int] = field(
        default_factory=lambda: [16, 16, 8, 4],
    )
    upsample_initial_channel: int = 128
    resblock_kernel_sizes: list[int] = field(
        default_factory=lambda: [3, 7, 11],
    )
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    )

    # Audio parameters (must match SpectrogramConfig)
    num_mels: int = 128  # Project uses 128, not the original 80
    sample_rate: int = 48_000
    hop_size: int = 512  # Must match SpectrogramConfig.hop_length

    # Training defaults (tuned for small datasets)
    learning_rate: float = 0.0002
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    segment_size: int = 24_576  # 0.512s at 48kHz, multiple of hop_size
    batch_size: int = 8

    # Discriminator
    mpd_periods: list[int] = field(
        default_factory=lambda: [2, 3, 5, 7, 11],
    )

    def __post_init__(self) -> None:
        """Validate that upsample_rates product equals hop_size."""
        product = 1
        for r in self.upsample_rates:
            product *= r
        if product != self.hop_size:
            msg = (
                f"Product of upsample_rates {self.upsample_rates} is "
                f"{product}, but hop_size is {self.hop_size}. "
                f"These must be equal."
            )
            raise ValueError(msg)
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            msg = (
                f"upsample_rates ({len(self.upsample_rates)}) and "
                f"upsample_kernel_sizes ({len(self.upsample_kernel_sizes)}) "
                f"must have the same length."
            )
            raise ValueError(msg)
