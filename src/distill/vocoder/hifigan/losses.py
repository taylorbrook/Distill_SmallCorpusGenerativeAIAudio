"""HiFi-GAN loss functions for GAN training.

Implements the three standard HiFi-GAN loss components:

- **Generator loss:** Least-squares GAN loss encouraging the generator
  to produce outputs that the discriminator classifies as real.
- **Discriminator loss:** Least-squares GAN loss training the
  discriminator to distinguish real from generated audio.
- **Feature matching loss:** L1 distance between discriminator
  intermediate feature maps of real and generated audio, weighted by 2.

All functions are pure (not ``nn.Module``), accepting lists of tensors
from the discriminator forward passes.

Reference: https://arxiv.org/abs/2010.05646
"""

from __future__ import annotations

import torch


def generator_loss(
    disc_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Least-squares GAN generator loss.

    Encourages discriminator outputs for generated audio to be close
    to 1 (classified as real).

    Parameters
    ----------
    disc_outputs : list[torch.Tensor]
        Discriminator outputs for generated audio, one per
        sub-discriminator.

    Returns
    -------
    tuple[torch.Tensor, list[torch.Tensor]]
        ``(total_loss, per_discriminator_losses)``
    """
    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    gen_losses: list[torch.Tensor] = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l
    return loss, gen_losses


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Least-squares GAN discriminator loss.

    Encourages discriminator to output 1 for real audio and 0 for
    generated audio.

    Parameters
    ----------
    disc_real_outputs : list[torch.Tensor]
        Discriminator outputs for real audio.
    disc_generated_outputs : list[torch.Tensor]
        Discriminator outputs for generated audio.

    Returns
    -------
    tuple[torch.Tensor, list[float], list[float]]
        ``(total_loss, real_losses, generated_losses)``
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    r_losses: list[float] = []
    g_losses: list[float] = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss = loss + r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def feature_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    """L1 feature matching loss between discriminator feature maps.

    Computes the mean absolute difference between intermediate feature
    maps of real and generated audio across all discriminator layers,
    multiplied by 2 (weight factor from original HiFi-GAN).

    Parameters
    ----------
    fmap_r : list[list[torch.Tensor]]
        Feature maps from discriminator on real audio.
        Outer list: sub-discriminators. Inner list: layers.
    fmap_g : list[list[torch.Tensor]]
        Feature maps from discriminator on generated audio.

    Returns
    -------
    torch.Tensor
        Scalar feature matching loss.
    """
    loss = torch.tensor(0.0, device=fmap_r[0][0].device)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + torch.mean(torch.abs(rl - gl))
    return loss * 2
