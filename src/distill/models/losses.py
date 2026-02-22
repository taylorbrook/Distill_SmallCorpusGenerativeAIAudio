"""VAE loss functions with KL annealing, free bits, and combined multi-resolution loss.

Provides two loss APIs:

1. **Legacy ``vae_loss``:** Simple MSE reconstruction + KL divergence.
   Kept for backward compatibility.
2. **``compute_combined_loss``:** Multi-resolution STFT loss (auraloss) +
   per-channel L1 reconstruction with magnitude-weighted IF + KL divergence.
   This is the v2.0 loss function for 2-channel complex spectrogram training.

Both share the same KL machinery:

- **KL annealing:** Linearly increase the KL weight from 0 to
  ``kl_weight_max`` (beta) over a warmup fraction of training.
- **Free bits:** Per-dimension KL floor that ensures each latent
  dimension encodes some information.

Design notes:
- ``compute_combined_loss`` returns a dict of individual loss components for
  granular logging (STFT, magnitude recon, IF recon, KL).
- Multi-resolution STFT loss operates on flattened spectrogram rows via
  auraloss, applied to the magnitude channel only.
- IF reconstruction uses magnitude-weighted L1 so errors in low-energy
  (inaudible) regions contribute less.
- ``compute_kl_divergence`` provides raw KL for monitoring posterior collapse
  independent of free bits or annealing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from auraloss.freq import MultiResolutionSTFTLoss

    from distill.training.config import LossConfig

logger = logging.getLogger(__name__)


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
        Reconstructed spectrogram ``[B, C, n_mels, time]`` (C=2 for v2.0).
    target : torch.Tensor
        Original spectrogram ``[B, C, n_mels, time]`` (C=2 for v2.0).
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


def create_stft_loss(
    loss_config: LossConfig, device: torch.device
) -> MultiResolutionSTFTLoss:
    """Create a MultiResolutionSTFTLoss from config, moved to device.

    Parameters
    ----------
    loss_config:
        Combined loss configuration containing STFT parameters.
    device:
        Target device for the loss module.

    Returns
    -------
    MultiResolutionSTFTLoss
        Configured multi-resolution STFT loss module on *device*.
    """
    from auraloss.freq import MultiResolutionSTFTLoss

    return MultiResolutionSTFTLoss(
        fft_sizes=list(loss_config.stft.fft_sizes),
        hop_sizes=list(loss_config.stft.hop_sizes),
        win_lengths=list(loss_config.stft.win_lengths),
    ).to(device)


def compute_combined_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    loss_config: LossConfig,
    kl_weight: float,
    _stft_loss_fn: MultiResolutionSTFTLoss | None = None,
) -> dict[str, torch.Tensor]:
    """Compute combined multi-resolution loss for 2-channel VAE training.

    Combines three loss components:
    1. **Multi-resolution STFT loss** (auraloss) on magnitude channel
    2. **Per-channel L1 reconstruction** with magnitude-weighted IF
    3. **KL divergence** with free bits

    Parameters
    ----------
    recon:
        Reconstructed spectrogram ``[B, 2, n_mels, time]``.
    target:
        Original spectrogram ``[B, 2, n_mels, time]``.
    mu:
        Latent mean ``[B, latent_dim]``.
    logvar:
        Latent log-variance ``[B, latent_dim]``.
    loss_config:
        Combined loss configuration with nested STFT, reconstruction,
        and KL sub-configs.
    kl_weight:
        Current KL annealing weight (from ``get_kl_weight``).
    _stft_loss_fn:
        Pre-created MultiResolutionSTFTLoss instance. If *None*, one is
        created from *loss_config* (lazy initialization).

    Returns
    -------
    dict[str, torch.Tensor]
        Keys: ``total_loss``, ``stft_loss``, ``mag_recon_loss``,
        ``if_recon_loss``, ``kl_loss``, ``recon_loss``.
        All values are scalar tensors.
    """
    # --- 1. Per-channel reconstruction loss (L1) ---
    recon_mag = recon[:, 0:1, :, :]  # [B, 1, n_mels, time]
    target_mag = target[:, 0:1, :, :]
    recon_if = recon[:, 1:2, :, :]  # [B, 1, n_mels, time]
    target_if = target[:, 1:2, :, :]

    # Magnitude channel: plain L1
    mag_loss = F.l1_loss(recon_mag, target_mag)

    # IF channel: magnitude-weighted L1
    # Normalize weights so mean weight ~= 1 (preserves loss scale)
    mag_weights = target_mag / (target_mag.mean() + 1e-8)
    if_error = (recon_if - target_if).abs()
    weighted_if_loss = (if_error * mag_weights).mean()

    # Combined reconstruction term
    recon_term = loss_config.reconstruction.weight * (
        loss_config.reconstruction.magnitude_weight * mag_loss
        + loss_config.reconstruction.if_weight * weighted_if_loss
    )

    # --- 2. Multi-resolution STFT loss (magnitude channel only) ---
    if _stft_loss_fn is None:
        _stft_loss_fn = create_stft_loss(loss_config, recon.device)

    # auraloss expects [B, 1, T] -- flatten spectrogram to 1D signal
    B = recon_mag.shape[0]
    recon_mag_flat = recon_mag.reshape(B, 1, -1)  # [B, 1, n_mels * time]
    target_mag_flat = target_mag.reshape(B, 1, -1)

    stft_loss_raw = _stft_loss_fn(recon_mag_flat, target_mag_flat)
    stft_loss_raw = stft_loss_raw.clamp(min=0.0)  # epsilon guard

    stft_term = loss_config.stft.weight * stft_loss_raw

    # --- 3. KL divergence (reuse existing KL computation) ---
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=loss_config.kl.free_bits)
    kl_loss = kl_per_dim.sum(dim=1).mean()

    kl_term = kl_weight * kl_loss if kl_weight > 0 else torch.zeros_like(kl_loss)

    # --- 4. Total loss ---
    total = stft_term + recon_term + kl_term

    # NaN stability guard
    if not torch.isfinite(total):
        logger.warning(
            "compute_combined_loss: total_loss is NaN/Inf, "
            "falling back to recon_loss only"
        )
        total = recon_term

    return {
        "total_loss": total,
        "stft_loss": stft_loss_raw,
        "mag_recon_loss": mag_loss,
        "if_recon_loss": weighted_if_loss,
        "kl_loss": kl_loss,
        "recon_loss": recon_term,
    }
