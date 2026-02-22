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
    free_bits: float = 0.1,
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

    # Total loss with annealing weight (guard against 0.0 * inf = NaN)
    if kl_weight > 0:
        total_loss = recon_loss + kl_weight * kl_loss
    else:
        total_loss = recon_loss

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


# ---------------------------------------------------------------------------
# VQ-VAE loss functions (v1.1)
# ---------------------------------------------------------------------------


def multi_scale_mel_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Multi-scale MSE loss on mel spectrograms at multiple resolutions.

    Captures both fine-grained spectral detail and coarse structural
    reconstruction quality by comparing at three scales:

    1. **Full resolution:** Pixel-level MSE between recon and target.
    2. **2x downsampled:** ``avg_pool2d(kernel=2)`` smooths out fine detail,
       focusing on medium-scale spectral structure.
    3. **4x downsampled:** ``avg_pool2d(kernel=4)`` captures broad energy
       distribution and overall spectral shape.

    Averaging across scales prevents the loss from being dominated by
    high-frequency noise while still penalising blurry reconstructions.

    Parameters
    ----------
    recon : torch.Tensor
        Reconstructed mel spectrogram ``[B, 1, n_mels, time]``.
    target : torch.Tensor
        Original mel spectrogram ``[B, 1, n_mels, time]``.

    Returns
    -------
    torch.Tensor
        Scalar loss averaged across all three scales.
    """
    # Full resolution
    loss_full = F.mse_loss(recon, target, reduction="mean")

    # 2x downsampled
    loss_2x = F.mse_loss(
        F.avg_pool2d(recon, 2),
        F.avg_pool2d(target, 2),
        reduction="mean",
    )

    # 4x downsampled
    loss_4x = F.mse_loss(
        F.avg_pool2d(recon, 4),
        F.avg_pool2d(target, 4),
        reduction="mean",
    )

    return (loss_full + loss_2x + loss_4x) / 3.0


def vqvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    commit_loss: torch.Tensor,
    commitment_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VQ-VAE loss combining multi-scale spectral reconstruction with commitment loss.

    Replaces :func:`vae_loss` for VQ-VAE training (v1.1).  There is **no KL
    divergence**, no free bits, and no annealing schedule.  The only tunable
    parameter is ``commitment_weight`` which balances reconstruction quality
    against codebook commitment (encoder-to-codebook alignment).

    The reconstruction term uses :func:`multi_scale_mel_loss` which compares
    mel spectrograms at full, 2x, and 4x downsampled resolutions for both
    fine-grained and structural reconstruction quality.

    Parameters
    ----------
    recon : torch.Tensor
        Reconstructed mel spectrogram ``[B, 1, n_mels, time]``.
    target : torch.Tensor
        Original mel spectrogram ``[B, 1, n_mels, time]``.
    commit_loss : torch.Tensor
        Commitment loss from :class:`QuantizerWrapper` (scalar or
        multi-element tensor from per-quantizer losses).
    commitment_weight : float
        Weight for commitment loss term (default 0.25).  This is the single
        tunable hyperparameter for VQ-VAE loss -- per user decision, no other
        scheduling or weighting mechanisms are used.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(total_loss, recon_loss, weighted_commit)`` -- all finite scalar
        tensors suitable for ``.backward()``.
    """
    # Multi-scale spectral reconstruction loss
    recon_loss = multi_scale_mel_loss(recon, target)

    # Commitment term: handle both scalar and multi-element commit_loss
    if commit_loss.dim() > 0:
        commit_loss = commit_loss.sum()
    weighted_commit = commitment_weight * commit_loss

    # Total: reconstruction + commitment (no KL, no annealing)
    total_loss = recon_loss + weighted_commit

    return total_loss, recon_loss, weighted_commit
