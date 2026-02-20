"""VAE loss functions with KL annealing and free bits.

Provides the core VAE loss (reconstruction + KL divergence) with two
mechanisms to prevent posterior collapse:

1. **KL annealing:** Linearly increase the KL weight from 0 to
   ``kl_weight_max`` (beta) over a warmup fraction of training.
2. **Free bits:** Per-dimension KL floor that ensures each latent
   dimension encodes some information.

Design notes:
- Reconstruction loss uses MSE on log-mel spectrograms (already normalised
  by ``log1p``).
- KL is summed over latent dimensions and averaged over the batch.
- ``compute_kl_divergence`` provides raw KL for monitoring posterior collapse
  independent of free bits or annealing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    free_bits: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss with free bits and KL annealing weight.

    Parameters
    ----------
    recon : torch.Tensor
        Reconstructed mel spectrogram ``[B, 1, n_mels, time]``.
    target : torch.Tensor
        Original mel spectrogram ``[B, 1, n_mels, time]``.
    mu : torch.Tensor
        Latent mean ``[B, latent_dim]``.
    logvar : torch.Tensor
        Latent log-variance ``[B, latent_dim]``.
    kl_weight : float
        Annealing weight for KL term (0.0 to 1.0).
    free_bits : float
        Minimum KL per dimension (prevents posterior collapse).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(total_loss, recon_loss, kl_loss)`` -- all finite scalars.
    """
    # Reconstruction loss: MSE on log-mel spectrograms
    recon_loss = F.mse_loss(recon, target, reduction="mean")

    # KL divergence per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Free bits: clamp each dimension to minimum of free_bits
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Sum over latent dims, mean over batch
    kl_loss = kl_per_dim.sum(dim=1).mean()

    # Total loss with annealing weight
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def get_kl_weight(
    epoch: int,
    total_epochs: int,
    warmup_fraction: float = 0.3,
    kl_weight_max: float = 1.0,
) -> float:
    """Linear KL annealing: weight from 0 to *kl_weight_max* over warmup fraction.

    Parameters
    ----------
    epoch : int
        Current epoch (0-indexed).
    total_epochs : int
        Total number of training epochs.
    warmup_fraction : float
        Fraction of training over which to anneal (default 0.3 = 30%).
    kl_weight_max : float
        Maximum KL weight (beta). Values < 1.0 create a beta-VAE that
        prioritises reconstruction over KL regularisation.

    Returns
    -------
    float
        KL weight between 0.0 and *kl_weight_max*.
    """
    warmup_epochs = int(total_epochs * warmup_fraction)
    if warmup_epochs == 0:
        return kl_weight_max
    return min(kl_weight_max, kl_weight_max * epoch / warmup_epochs)


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> float:
    """Compute raw KL divergence (without free bits) for monitoring.

    Useful for detecting posterior collapse: if this value drops below
    0.5 across all latent dimensions, the model may be collapsing.

    Parameters
    ----------
    mu : torch.Tensor
        Latent mean ``[B, latent_dim]``.
    logvar : torch.Tensor
        Latent log-variance ``[B, latent_dim]``.

    Returns
    -------
    float
        Scalar mean KL divergence value.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_per_dim.sum(dim=1).mean().item()
