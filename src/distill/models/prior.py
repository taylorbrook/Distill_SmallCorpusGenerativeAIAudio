"""Autoregressive prior over flattened VQ-VAE code sequences.

Provides a GPT-style decoder-only Transformer (:class:`CodePrior`) that
models ``P(code_t | code_{<t})`` over position-major interleaved code
sequences.  The prior learns to predict the next discrete code token in
a flattened sequence produced by a frozen VQ-VAE.

Companion utilities:

- :func:`flatten_codes` / :func:`unflatten_codes` -- reshape between
  ``[B, seq_len, num_quantizers]`` and ``[B, seq_len * num_quantizers]``
- :func:`extract_code_sequences` -- encode an entire dataset through a
  frozen VQ-VAE and return all code indices as a tensor

Architecture notes:

- Uses ``nn.TransformerEncoder`` with an explicit causal mask (not
  ``is_causal=True``) per research open question 1: the mask tensor
  approach is guaranteed to work on all backends (CPU, CUDA, MPS).
- Three embedding layers summed: token, position, and quantizer-level
  (per research pitfall 3: level embedding disambiguates quantizer
  levels within the flattened sequence).
- ``norm_first=True`` (pre-norm) for stable training.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from distill.audio.spectrogram import AudioSpectrogram

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code flattening utilities
# ---------------------------------------------------------------------------


def flatten_codes(indices: torch.Tensor) -> torch.Tensor:
    """Flatten VQ-VAE indices from spatial to sequence format.

    Interleaves codes in position-major order:
    ``[pos0_q0, pos0_q1, pos0_q2, pos1_q0, ...]``

    Parameters
    ----------
    indices:
        Shape ``[B, seq_len, num_quantizers]`` integer indices from VQ-VAE.

    Returns
    -------
    torch.Tensor
        Shape ``[B, seq_len * num_quantizers]`` flattened code sequence.
    """
    B, S, Q = indices.shape
    return indices.reshape(B, S * Q)


def unflatten_codes(flat: torch.Tensor, num_quantizers: int) -> torch.Tensor:
    """Unflatten a code sequence back to spatial format.

    Inverse of :func:`flatten_codes`.

    Parameters
    ----------
    flat:
        Shape ``[B, seq_len * num_quantizers]`` flattened code sequence.
    num_quantizers:
        Number of quantizer levels per spatial position.

    Returns
    -------
    torch.Tensor
        Shape ``[B, seq_len, num_quantizers]``.
    """
    B, L = flat.shape
    S = L // num_quantizers
    return flat.reshape(B, S, num_quantizers)


# ---------------------------------------------------------------------------
# CodePrior model
# ---------------------------------------------------------------------------


class CodePrior(nn.Module):
    """Autoregressive transformer prior over flattened VQ-VAE code sequences.

    A GPT-style decoder-only transformer that predicts the next discrete
    code token.  Built with ``nn.TransformerEncoder`` + explicit causal mask
    (architecturally identical to GPT but avoids the confusing
    encoder/decoder naming of ``nn.TransformerDecoder``).

    Parameters
    ----------
    codebook_size:
        Number of entries in each codebook (vocabulary size).
    seq_len:
        Maximum flattened sequence length
        (``spatial_positions * num_quantizers``).
    num_quantizers:
        Number of RVQ levels.  Used to compute the level embedding index
        for each token position (``position % num_quantizers``).
    hidden_size:
        Transformer hidden dimension.
    num_layers:
        Number of transformer encoder layers.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        codebook_size: int,
        seq_len: int,
        num_quantizers: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Store for external access
        self.codebook_size = codebook_size
        self.seq_len = seq_len
        self.num_quantizers = num_quantizers

        # Three embedding layers (summed per research pitfall 3)
        self.token_emb = nn.Embedding(codebook_size, hidden_size)
        self.pos_emb = nn.Embedding(seq_len, hidden_size)
        self.level_emb = nn.Embedding(num_quantizers, hidden_size)

        self.drop = nn.Dropout(dropout)

        # Transformer encoder stack with pre-norm (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final layer norm + output projection
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, codebook_size, bias=False)

        # Causal mask: True = blocked (upper triangular)
        # Registered as buffer so it moves with the model to device
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal self-attention.

        Parameters
        ----------
        x:
            Shape ``[B, T]`` long tensor of code indices
            (``0 .. codebook_size - 1``).  ``T <= seq_len``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, T, codebook_size]`` logits for next-token
            prediction.
        """
        B, T = x.shape

        # Position indices [1, T]
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        # Level indices: each token's quantizer level within its position
        levels = (torch.arange(T, device=x.device) % self.num_quantizers).unsqueeze(0)

        # Sum three embeddings
        h = self.token_emb(x) + self.pos_emb(positions) + self.level_emb(levels)
        h = self.drop(h)

        # Slice causal mask to current sequence length
        mask = self.causal_mask[:T, :T]

        # Causal self-attention through transformer encoder
        h = self.transformer(h, mask=mask)

        # Final layer norm + linear head
        h = self.ln_f(h)
        logits = self.head(h)  # [B, T, codebook_size]

        return logits


# ---------------------------------------------------------------------------
# Code extraction pipeline
# ---------------------------------------------------------------------------


def extract_code_sequences(
    model: "ConvVQVAE",
    dataloader: DataLoader,
    spectrogram: AudioSpectrogram,
    device: torch.device | str,
) -> torch.Tensor:
    """Encode an entire dataset through a frozen VQ-VAE to get code sequences.

    Sets the model to eval mode and disables gradients.  Each batch is
    converted to mel spectrograms via *spectrogram*, then encoded through
    the VQ-VAE.  The resulting code indices are collected and concatenated.

    Parameters
    ----------
    model:
        A trained :class:`~distill.models.vqvae.ConvVQVAE` model.
    dataloader:
        DataLoader yielding waveform batches (shape ``[B, samples]``).
    spectrogram:
        :class:`~distill.audio.spectrogram.AudioSpectrogram` instance for
        waveform-to-mel conversion.
    device:
        Device to run inference on.

    Returns
    -------
    torch.Tensor
        Shape ``[N, seq_len, num_quantizers]`` -- all code indices for
        the dataset.
    """
    model.eval()
    all_indices: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mel = spectrogram.waveform_to_mel(batch)
            _recon, indices, _commit_loss = model(mel)
            all_indices.append(indices.cpu())

    return torch.cat(all_indices, dim=0)
