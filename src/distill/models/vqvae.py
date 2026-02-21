"""Convolutional VQ-VAE with Residual Vector Quantization for mel spectrograms.

Replaces :class:`ConvVAE` (v1.0 continuous VAE) with a discrete bottleneck.
The forward pass returns ``(recon, indices, commit_loss)`` instead of
``(recon, mu, logvar)`` -- there is no KL divergence, no reparameterization
trick, and no sampling from N(0,1).  Generation requires a learned prior
(Phase 14) or direct code manipulation (Phase 16).

Architecture overview::

    Input mel  [B, 1, 128, 94]
        |
    VQEncoder  (4-layer Conv2d stride-2, 1x1 projection)
        |
    Embeddings [B, codebook_dim, H, W]   (H=8, W=6 for default mel)
        |
    QuantizerWrapper  (ResidualVQ from vector-quantize-pytorch)
        |  reshape [B,D,H,W] -> [B,H*W,D], quantize, reshape back
        |
    Quantized  [B, codebook_dim, H, W] + indices [B, H*W, num_quantizers]
        |
    VQDecoder  (1x1 projection, 4-layer ConvTranspose2d stride-2)
        |
    Recon mel  [B, 1, 128, 94]   (cropped to original input shape)

Temporal compression: 4 stride-2 layers give 16x compression on each
spatial axis.  For 1-second audio at 48 kHz (128 mels, 94 time frames):
94 -> pad to 96 -> 6 time positions.  Each position covers ~167 ms,
giving medium resolution suitable for region-level code editing.

Design notes:

- Fresh design -- does NOT reuse v1.0 ConvEncoder/ConvDecoder classes.
- No lazy init needed: Conv2d layers are spatially independent.
- float32 throughout -- no float16 on MPS (numerical precision issues).
- Apply gradient clipping externally in the training loop.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class VQEncoder(nn.Module):
    """4-layer convolutional encoder producing spatial embeddings for VQ.

    Input:  ``[B, 1, n_mels, time]``
    Output: ``[B, codebook_dim, H, W]`` where H*W positions are
            independently quantized through RVQ.

    The time dimension is padded to a multiple of 16 before convolutions
    (4 stride-2 layers = 16x downsampling).  The padded shape is stored
    in ``_padded_shape`` for the decoder to crop.

    Parameters
    ----------
    codebook_dim:
        Dimensionality of each codebook embedding vector (default 128).
    dropout:
        Dropout probability for regularisation (default 0.2).
    """

    def __init__(self, codebook_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.codebook_dim = codebook_dim

        # 4 conv blocks: 1 -> 32 -> 64 -> 128 -> 256
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Project to codebook dimension (replaces fc_mu/fc_logvar from v1.0)
        self.proj = nn.Conv2d(256, codebook_dim, 1)

        # Stored for the decoder to know the padded (pre-downsample) shape
        self._padded_shape: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to spatial embeddings.

        Parameters
        ----------
        x:
            Shape ``[B, 1, n_mels, time]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, codebook_dim, H, W]`` where ``H = padded_mels / 16``
            and ``W = padded_time / 16``.
        """
        # Pad both dims to multiples of 16 for 4 stride-2 layers
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        self._padded_shape = (x.shape[2], x.shape[3])

        h_out = self.convs(x)
        return self.proj(h_out)  # [B, codebook_dim, H, W]


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class VQDecoder(nn.Module):
    """4-layer transposed convolutional decoder from spatial embeddings to mel.

    Input:  ``[B, codebook_dim, H, W]`` quantized spatial embeddings
    Output: ``[B, 1, n_mels, time]`` (cropped to original shape if provided)

    Parameters
    ----------
    codebook_dim:
        Dimensionality of each codebook embedding vector (default 128).
    dropout:
        Dropout probability for regularisation (default 0.2).
    """

    def __init__(self, codebook_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()

        # Project from codebook dim back to conv channels
        self.proj = nn.Conv2d(codebook_dim, 256, 1)

        # 4 deconv blocks: 256 -> 128 -> 64 -> 32 -> 1
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),  # output >= 0, unbounded above -- matches log1p mel range
        )

    def forward(
        self,
        x: torch.Tensor,
        target_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode quantized spatial embeddings to mel spectrogram.

        Parameters
        ----------
        x:
            Shape ``[B, codebook_dim, H, W]`` quantized spatial embeddings.
        target_shape:
            ``(n_mels, time)`` of the original (unpadded) mel spectrogram.
            Used to crop the output.  If ``None``, returns full padded output.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, n_mels, time]``.
        """
        h = self.proj(x)       # [B, 256, H, W]
        recon = self.deconvs(h)  # [B, 1, H*16, W*16]

        # Crop to original shape (undo padding)
        if target_shape is not None:
            th, tw = target_shape
            recon = recon[:, :, :th, :tw]

        return recon


# ---------------------------------------------------------------------------
# Quantizer Wrapper
# ---------------------------------------------------------------------------


class QuantizerWrapper(nn.Module):
    """Thin wrapper around :class:`ResidualVQ` with codebook health monitoring.

    Provides a uniform interface for quantization and adds per-level
    utilization, perplexity, and dead-code tracking.

    Parameters
    ----------
    dim:
        Embedding vector dimensionality (must match encoder ``codebook_dim``).
    codebook_size:
        Number of entries per codebook.
    num_quantizers:
        Number of residual quantization levels.
    decay:
        EMA decay for codebook updates (lower = faster adaptation).
    commitment_weight:
        Weight for commitment loss (encoder-to-codebook alignment).
    threshold_ema_dead_code:
        Minimum EMA usage count; codes below this are replaced.
    kmeans_init:
        Whether to initialise codebooks with k-means on the first batch.
    kmeans_iters:
        Number of k-means iterations for initialisation.
    """

    def __init__(
        self,
        dim: int = 128,
        codebook_size: int = 256,
        num_quantizers: int = 3,
        decay: float = 0.95,
        commitment_weight: float = 0.25,
        threshold_ema_dead_code: int = 2,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
    ) -> None:
        super().__init__()
        self.rvq = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
        )
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize embedding sequence through residual VQ.

        Parameters
        ----------
        x:
            Shape ``[B, seq_len, dim]`` embedding sequence.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(quantized, indices, commit_loss)`` where
            ``quantized`` is ``[B, seq_len, dim]``,
            ``indices`` is ``[B, seq_len, num_quantizers]``,
            ``commit_loss`` is a scalar tensor.
        """
        quantized, indices, commit_loss = self.rvq(x)

        # ResidualVQ may return per-quantizer commit_loss -- reduce to scalar
        if commit_loss.dim() > 0:
            commit_loss = commit_loss.sum()

        return quantized, indices, commit_loss

    def get_codebook_utilization(
        self, indices: torch.Tensor,
    ) -> dict[str, dict[str, float | int]]:
        """Compute per-level codebook health metrics.

        Parameters
        ----------
        indices:
            Shape ``[B, seq_len, num_quantizers]`` code indices from forward.

        Returns
        -------
        dict[str, dict[str, float | int]]
            Keyed by ``"level_0"``, ``"level_1"``, etc.  Each contains:

            - ``utilization``: fraction of unique codes used (0.0-1.0)
            - ``perplexity``: exp(entropy) of code distribution
            - ``dead_codes``: number of unused codebook entries
        """
        metrics: dict[str, dict[str, float | int]] = {}
        for q in range(self.num_quantizers):
            level_indices = indices[:, :, q]  # [B, seq_len]
            unique = level_indices.unique()
            utilization = len(unique) / self.codebook_size

            # Perplexity: exp(entropy of code distribution)
            counts = torch.bincount(
                level_indices.flatten(), minlength=self.codebook_size,
            ).float()
            probs = counts / counts.sum()
            probs_nonzero = probs[probs > 0]
            entropy = -(probs_nonzero * probs_nonzero.log()).sum()
            perplexity = entropy.exp().item()

            dead_codes = int((counts == 0).sum().item())

            metrics[f"level_{q}"] = {
                "utilization": utilization,
                "perplexity": perplexity,
                "dead_codes": dead_codes,
            }
        return metrics

    def get_output_from_indices(
        self, indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode code indices back to quantized embeddings.

        Parameters
        ----------
        indices:
            Shape ``[B, seq_len, num_quantizers]`` code indices.

        Returns
        -------
        torch.Tensor
            Shape ``[B, seq_len, dim]`` quantized embeddings (sum of all
            residual levels).
        """
        return self.rvq.get_output_from_indices(indices)


# ---------------------------------------------------------------------------
# ConvVQVAE
# ---------------------------------------------------------------------------


class ConvVQVAE(nn.Module):
    """Convolutional VQ-VAE for audio mel spectrogram encoding/decoding.

    Replaces :class:`ConvVAE` (v1.0).  The bottleneck uses Residual Vector
    Quantization (RVQ) instead of continuous latent space with KL divergence.
    The forward pass returns ``(recon, indices, commit_loss)`` instead of
    ``(recon, mu, logvar)``.

    **Generation** requires a learned prior (Phase 14) that models the
    distribution of code indices.  Unlike the continuous VAE, you cannot
    simply sample from N(0,1).

    Parameters
    ----------
    codebook_dim:
        Dimensionality of each codebook embedding vector.
    codebook_size:
        Number of entries per codebook.
    num_quantizers:
        Number of residual quantization levels (2-4).
    decay:
        EMA decay for codebook updates.
    commitment_weight:
        Weight for commitment loss.
    threshold_ema_dead_code:
        Minimum EMA usage count for dead code replacement.
    dropout:
        Dropout probability in encoder/decoder layers.
    """

    def __init__(
        self,
        codebook_dim: int = 128,
        codebook_size: int = 256,
        num_quantizers: int = 3,
        decay: float = 0.95,
        commitment_weight: float = 0.25,
        threshold_ema_dead_code: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Store config values for persistence / reconstruction
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        # Sub-modules
        self.encoder = VQEncoder(codebook_dim=codebook_dim, dropout=dropout)
        self.quantizer = QuantizerWrapper(
            dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=True,
            kmeans_iters=10,
        )
        self.decoder = VQDecoder(codebook_dim=codebook_dim, dropout=dropout)

        # Set during forward for codes_to_embeddings
        self._spatial_shape: tuple[int, int] | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to spatial embeddings.

        Parameters
        ----------
        x:
            Shape ``[B, 1, n_mels, time]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, codebook_dim, H, W]``.
        """
        return self.encoder(x)

    def quantize(
        self, embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize spatial embeddings through RVQ.

        Reshapes ``[B, D, H, W]`` to ``[B, H*W, D]`` for the quantizer,
        then reshapes back to ``[B, D, H, W]`` afterward.

        Parameters
        ----------
        embeddings:
            Shape ``[B, codebook_dim, H, W]`` from encoder.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(quantized_spatial, indices, commit_loss)`` where
            ``quantized_spatial`` is ``[B, codebook_dim, H, W]``,
            ``indices`` is ``[B, H*W, num_quantizers]``,
            ``commit_loss`` is a scalar.
        """
        B, D, H, W = embeddings.shape

        # Reshape: channel-first spatial -> sequence for RVQ
        flat = embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Quantize through RVQ
        quantized, indices, commit_loss = self.quantizer(flat)

        # Reshape back to spatial: [B, H*W, D] -> [B, D, H, W]
        quantized_spatial = quantized.reshape(B, H, W, D).permute(0, 3, 1, 2)

        return quantized_spatial, indices, commit_loss

    def decode(
        self,
        quantized: torch.Tensor,
        target_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode quantized spatial embeddings to mel spectrogram.

        Parameters
        ----------
        quantized:
            Shape ``[B, codebook_dim, H, W]``.
        target_shape:
            ``(n_mels, time)`` for output cropping.  If ``None``, returns
            full (padded) output.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, n_mels, time]``.
        """
        return self.decoder(quantized, target_shape=target_shape)

    def codes_to_embeddings(
        self,
        indices: torch.Tensor,
        spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Convert code indices to quantized spatial embeddings for decode.

        Used for the decode-from-indices path (Phase 16 encode/decode).

        Parameters
        ----------
        indices:
            Shape ``[B, seq_len, num_quantizers]`` code indices.
        spatial_shape:
            ``(H, W)`` needed to reshape flat sequence back to spatial map.

        Returns
        -------
        torch.Tensor
            Shape ``[B, codebook_dim, H, W]``.
        """
        quantized = self.quantizer.get_output_from_indices(indices)
        B, _S, D = quantized.shape
        H, W = spatial_shape
        return quantized.reshape(B, H, W, D).permute(0, 3, 1, 2)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> quantize -> decode.

        Parameters
        ----------
        x:
            Shape ``[B, 1, n_mels, time]`` mel spectrogram.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(recon, indices, commit_loss)`` where:

            - ``recon`` is ``[B, 1, n_mels, time]`` (shape matches input)
            - ``indices`` is ``[B, H*W, num_quantizers]`` discrete codes
            - ``commit_loss`` is a scalar commitment loss tensor
        """
        original_shape = (x.shape[2], x.shape[3])  # (n_mels, time)

        # Encode to spatial embeddings
        embeddings = self.encode(x)  # [B, codebook_dim, H, W]
        B, D, H, W = embeddings.shape

        # Store spatial shape for codes_to_embeddings
        self._spatial_shape = (H, W)

        # Quantize through RVQ
        quantized, indices, commit_loss = self.quantize(embeddings)

        # Decode with crop to original shape
        recon = self.decode(quantized, target_shape=original_shape)

        return recon, indices, commit_loss
