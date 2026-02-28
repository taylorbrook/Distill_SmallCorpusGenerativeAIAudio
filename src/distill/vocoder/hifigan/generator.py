"""HiFi-GAN V2 generator for mel-to-waveform synthesis.

Implements the V2 generator architecture adapted for 128-band 48kHz mel
spectrograms. The generator upsamples mel frames through a series of
transposed convolutions interleaved with residual blocks.

Reference: https://arxiv.org/abs/2010.05646
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

if TYPE_CHECKING:
    from distill.vocoder.hifigan.config import HiFiGANConfig

LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    """HiFi-GAN V2 residual block with three dilated convolution layers.

    Each dilated conv is followed by a conv with dilation=1. All
    convolutions use weight normalization.
    """

    def __init__(self, channels: int, kernel_size: int, dilations: list[int]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=self._get_padding(kernel_size, d),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=self._get_padding(kernel_size, 1),
                    )
                )
            )

    @staticmethod
    def _get_padding(kernel_size: int, dilation: int) -> int:
        return (kernel_size * dilation - dilation) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class HiFiGANGenerator(nn.Module):
    """HiFi-GAN V2 generator.

    Takes mel spectrogram input ``[B, num_mels, T]`` and produces
    waveform output ``[B, 1, T * hop_size]``.

    Parameters
    ----------
    config : HiFiGANConfig
        Generator configuration with upsample rates, kernel sizes, and
        residual block parameters.
    """

    def __init__(self, config: HiFiGANConfig) -> None:
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # Pre-convolution: project mels to initial channel count
        self.conv_pre = weight_norm(
            nn.Conv1d(config.num_mels, config.upsample_initial_channel, 7, padding=3)
        )

        # Upsampling layers with interleaved residual blocks
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = config.upsample_initial_channel
        for i, (u, k) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        ch,
                        ch // 2,
                        k,
                        stride=u,
                        padding=(k - u) // 2,
                    )
                )
            )
            ch_out = ch // 2
            for j, (rk, rd) in enumerate(
                zip(
                    config.resblock_kernel_sizes,
                    config.resblock_dilation_sizes,
                )
            ):
                self.resblocks.append(ResBlock1(ch_out, rk, rd))
            ch = ch_out

        # Post-convolution: project to single-channel waveform
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Parameters
        ----------
        x : torch.Tensor
            Mel spectrogram, shape ``[B, num_mels, T]``.

        Returns
        -------
        torch.Tensor
            Generated waveform, shape ``[B, 1, T * hop_size]``.
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            # Sum outputs of all residual blocks for this upsampling stage
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from all layers (for inference)."""
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)
