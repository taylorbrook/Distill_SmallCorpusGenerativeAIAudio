"""HiFi-GAN Multi-Period and Multi-Scale discriminators.

Implements the two discriminator families used in HiFi-GAN training:

- **Multi-Period Discriminator (MPD):** Reshapes 1D audio into 2D
  sub-sequences of various periods and applies 2D convolutions to
  capture periodic patterns.
- **Multi-Scale Discriminator (MSD):** Operates on the raw waveform at
  multiple temporal resolutions via average-pooling downsampling.

Reference: https://arxiv.org/abs/2010.05646
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

if TYPE_CHECKING:
    from distill.vocoder.hifigan.config import HiFiGANConfig

LRELU_SLOPE = 0.1


# ---------------------------------------------------------------------------
# Multi-Period Discriminator
# ---------------------------------------------------------------------------


class PeriodDiscriminator(nn.Module):
    """Single-period sub-discriminator for MPD.

    Reshapes the 1D waveform into a 2D grid with shape
    ``[B, 1, T // period, period]``, then applies a stack of 2D
    convolutions to classify real vs generated audio.

    Parameters
    ----------
    period : int
        Reshaping period (e.g. 2, 3, 5, 7, or 11).
    """

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period

        # Channel progression: 1 -> 32 -> 128 -> 512 -> 1024 -> 1024 -> 1
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(1, 32, (5, 1), stride=(3, 1), padding=(2, 0))
                ),
                weight_norm(
                    nn.Conv2d(32, 128, (5, 1), stride=(3, 1), padding=(2, 0))
                ),
                weight_norm(
                    nn.Conv2d(128, 512, (5, 1), stride=(3, 1), padding=(2, 0))
                ),
                weight_norm(
                    nn.Conv2d(512, 1024, (5, 1), stride=(3, 1), padding=(2, 0))
                ),
                weight_norm(
                    nn.Conv2d(1024, 1024, (5, 1), stride=1, padding=(2, 0))
                ),
            ]
        )
        self.conv_post = weight_norm(
            nn.Conv2d(1024, 1, (3, 1), stride=1, padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Classify waveform segment.

        Parameters
        ----------
        x : torch.Tensor
            Waveform, shape ``[B, 1, T]``.

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor]]
            ``(output, feature_maps)`` where *output* is the
            discriminator score and *feature_maps* is a list of
            intermediate activations for feature matching loss.
        """
        fmap: list[torch.Tensor] = []

        # Reshape 1D -> 2D: [B, 1, T] -> [B, 1, T//period, period]
        b, c, t = x.shape
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            t = t + pad_len
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD).

    Wraps multiple :class:`PeriodDiscriminator` instances, one for each
    period in ``config.mpd_periods`` (default ``[2, 3, 5, 7, 11]``).

    Parameters
    ----------
    config : HiFiGANConfig
        Configuration providing ``mpd_periods``.
    """

    def __init__(self, config: HiFiGANConfig) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in config.mpd_periods]
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:
        """Run all period discriminators on real and generated audio.

        Parameters
        ----------
        y : torch.Tensor
            Real waveform, shape ``[B, 1, T]``.
        y_hat : torch.Tensor
            Generated waveform, shape ``[B, 1, T]``.

        Returns
        -------
        tuple
            ``(real_outputs, fake_outputs, real_fmaps, fake_fmaps)``
        """
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []
        real_fmaps: list[list[torch.Tensor]] = []
        fake_fmaps: list[list[torch.Tensor]] = []

        for d in self.discriminators:
            r, r_fmap = d(y)
            f, f_fmap = d(y_hat)
            real_outputs.append(r)
            fake_outputs.append(f)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)

        return real_outputs, fake_outputs, real_fmaps, fake_fmaps


# ---------------------------------------------------------------------------
# Multi-Scale Discriminator
# ---------------------------------------------------------------------------


class ScaleDiscriminator(nn.Module):
    """Single-scale sub-discriminator for MSD.

    A 1D convolutional classifier operating at one temporal resolution.

    Parameters
    ----------
    use_spectral_norm : bool
        If ``True``, use spectral normalization instead of weight
        normalization. The first scale (raw audio) typically uses
        spectral norm.
    """

    def __init__(self, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, stride=1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, stride=2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, stride=2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, stride=4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, stride=4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, stride=1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Classify waveform at this scale.

        Parameters
        ----------
        x : torch.Tensor
            Waveform, shape ``[B, 1, T]``.

        Returns
        -------
        tuple[torch.Tensor, list[torch.Tensor]]
            ``(output, feature_maps)``
        """
        fmap: list[torch.Tensor] = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator (MSD).

    Operates at 3 temporal scales:
    - Scale 0: raw waveform (spectral norm)
    - Scale 1: 2x downsampled (weight norm)
    - Scale 2: 4x downsampled (weight norm)

    Downsampling uses ``AvgPool1d(4, 2, padding=2)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(use_spectral_norm=True),
                ScaleDiscriminator(),
                ScaleDiscriminator(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[list[torch.Tensor]],
        list[list[torch.Tensor]],
    ]:
        """Run all scale discriminators on real and generated audio.

        Parameters
        ----------
        y : torch.Tensor
            Real waveform, shape ``[B, 1, T]``.
        y_hat : torch.Tensor
            Generated waveform, shape ``[B, 1, T]``.

        Returns
        -------
        tuple
            ``(real_outputs, fake_outputs, real_fmaps, fake_fmaps)``
        """
        real_outputs: list[torch.Tensor] = []
        fake_outputs: list[torch.Tensor] = []
        real_fmaps: list[list[torch.Tensor]] = []
        fake_fmaps: list[list[torch.Tensor]] = []

        for i, d in enumerate(self.discriminators):
            if i > 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            r, r_fmap = d(y)
            f, f_fmap = d(y_hat)
            real_outputs.append(r)
            fake_outputs.append(f)
            real_fmaps.append(r_fmap)
            fake_fmaps.append(f_fmap)

        return real_outputs, fake_outputs, real_fmaps, fake_fmaps
