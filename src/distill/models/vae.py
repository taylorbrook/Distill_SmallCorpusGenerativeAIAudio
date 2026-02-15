"""Convolutional Variational Autoencoder for mel spectrograms.

Architecture: 4-layer convolutional encoder/decoder with stride-2 downsampling,
mapping mel spectrograms to a 64-dimensional latent space.  Pad-then-crop
strategy ensures exact shape match between input and output regardless of
time dimension length.

Design notes:
- The encoder pads the time dimension to a multiple of 16 (4 stride-2 layers).
- ``flatten_dim`` is computed lazily on the first forward pass to support
  variable mel shapes without hard-coding spatial dimensions.
- float32 throughout -- no float16 on MPS (numerical precision issues).
- Apply gradient clipping (``max_norm=1.0``) externally in the training loop
  to prevent MPS-specific NaN gradients.

Total parameters: ~3.1M (varies slightly with mel time dimension).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class ConvEncoder(nn.Module):
    """4-layer convolutional encoder mapping mel spectrograms to latent space.

    Input:  ``[B, 1, n_mels, time]``
    Output: ``(mu, logvar)`` each ``[B, latent_dim]``

    The time dimension is padded to a multiple of 16 before convolutions,
    and the padded shape is stored in ``_padded_shape`` for the decoder.
    """

    def __init__(self, latent_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.latent_dim = latent_dim

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

        # Lazy: computed on first forward pass
        self._flatten_dim: int | None = None
        self.fc_mu: nn.Linear | None = None
        self.fc_logvar: nn.Linear | None = None

        # Stored for the decoder to know the target shape
        self._padded_shape: tuple[int, int] | None = None

    def _init_linear(self, flatten_dim: int) -> None:
        """Initialise linear heads once ``flatten_dim`` is known."""
        self._flatten_dim = flatten_dim
        device = next(self.convs.parameters()).device
        self.fc_mu = nn.Linear(flatten_dim, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(flatten_dim, self.latent_dim).to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode mel spectrogram to ``(mu, logvar)``.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, 1, n_mels, time]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mu, logvar)`` each ``[B, latent_dim]``.
        """
        # Pad time dimension to multiple of 16
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        self._padded_shape = (x.shape[2], x.shape[3])

        h = self.convs(x)
        h_flat = h.flatten(1)

        # Lazy init of linear layers on first pass
        if self._flatten_dim is None:
            self._init_linear(h_flat.shape[1])
        assert self.fc_mu is not None  # for type checker

        return self.fc_mu(h_flat), self.fc_logvar(h_flat)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class ConvDecoder(nn.Module):
    """4-layer transposed convolutional decoder mapping latent vectors to mel spectrograms.

    Input:  ``z`` of shape ``[B, latent_dim]`` and ``target_shape`` tuple
    Output: ``[B, 1, n_mels, time]`` (cropped to original shape)
    """

    def __init__(self, latent_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Lazy: computed on first forward pass
        self._spatial_shape: tuple[int, int] | None = None
        self._flatten_dim: int | None = None
        self.fc: nn.Linear | None = None

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
            nn.Sigmoid(),  # output is log1p-normalised mel (always >= 0)
        )

    def _init_linear(self, spatial_shape: tuple[int, int]) -> None:
        """Initialise linear projection once spatial shape is known."""
        self._spatial_shape = spatial_shape
        sh, sw = spatial_shape
        self._flatten_dim = 256 * sh * sw
        device = next(self.deconvs.parameters()).device
        self.fc = nn.Linear(self.latent_dim, self._flatten_dim).to(device)

    def forward(
        self,
        z: torch.Tensor,
        target_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode latent vector to mel spectrogram.

        Parameters
        ----------
        z : torch.Tensor
            Shape ``[B, latent_dim]``.
        target_shape : tuple[int, int] | None
            ``(n_mels, time)`` of the original (unpadded) mel spectrogram.
            Used to crop the output.  If ``None``, returns full padded output.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, n_mels, time]``.
        """
        if self.fc is None:
            raise RuntimeError(
                "Decoder linear layer not initialised. Run encoder forward "
                "pass first, or call model.forward() which does both."
            )
        assert self._spatial_shape is not None

        sh, sw = self._spatial_shape
        h = self.fc(z).view(-1, 256, sh, sw)
        recon = self.deconvs(h)

        # Crop to original shape (undo padding)
        if target_shape is not None:
            th, tw = target_shape
            recon = recon[:, :, :th, :tw]

        return recon


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

# Default mel shape for 1 second at 48 kHz (128 mels, 94 time frames)
_DEFAULT_MEL_SHAPE = (128, 94)


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for audio mel spectrograms.

    Encodes mel spectrograms to a low-dimensional latent space and
    decodes back.  Supports training (forward), encoding, decoding,
    and generation (sample) modes.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space (default 64).
    dropout : float
        Dropout probability for regularisation (default 0.2).
    """

    def __init__(self, latent_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim=latent_dim, dropout=dropout)
        self.decoder = ConvDecoder(latent_dim=latent_dim, dropout=dropout)

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        return self._latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode mel spectrogram to latent parameters.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, 1, n_mels, time]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mu, logvar)`` each ``[B, latent_dim]``.
        """
        return self.encoder(x)

    @staticmethod
    def reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick: ``z = mu + std * eps``.

        Parameters
        ----------
        mu : torch.Tensor
            Latent mean ``[B, latent_dim]``.
        logvar : torch.Tensor
            Latent log-variance ``[B, latent_dim]``.

        Returns
        -------
        torch.Tensor
            Sampled latent vector ``[B, latent_dim]``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(
        self,
        z: torch.Tensor,
        target_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode latent vector to mel spectrogram.

        Parameters
        ----------
        z : torch.Tensor
            Shape ``[B, latent_dim]``.
        target_shape : tuple[int, int] | None
            ``(n_mels, time)`` of the desired output.  When ``None``
            (generation mode), uses default shape for 1-second audio.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, n_mels, time]``.
        """
        return self.decoder(z, target_shape=target_shape)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, reparameterize, decode.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, 1, n_mels, time]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(recon, mu, logvar)`` where ``recon.shape == x.shape``.
        """
        original_shape = (x.shape[2], x.shape[3])  # (n_mels, time)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Initialise decoder linear layer on first pass
        if self.decoder.fc is None:
            # Compute spatial shape after encoder convolutions using
            # the padded shape stored by the encoder during encode()
            assert self.encoder._padded_shape is not None
            padded_h, padded_w = self.encoder._padded_shape
            # 4 stride-2 layers: spatial dims / 16
            spatial = (padded_h // 16, padded_w // 16)
            self.decoder._init_linear(spatial)

        recon = self.decode(z, target_shape=original_shape)
        return recon, mu, logvar

    def sample(
        self,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate mel spectrograms from random latent vectors.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        device : torch.device
            Device to generate on.

        Returns
        -------
        torch.Tensor
            Shape ``[num_samples, 1, n_mels, time]`` using default 1-second shape.
        """
        z = torch.randn(num_samples, self._latent_dim, device=device)

        # Ensure decoder is initialised -- use default mel shape
        if self.decoder.fc is None:
            n_mels, time_frames = _DEFAULT_MEL_SHAPE
            # Pad to multiple of 16
            pad_h = (16 - n_mels % 16) % 16
            pad_w = (16 - time_frames % 16) % 16
            padded_h = n_mels + pad_h
            padded_w = time_frames + pad_w
            # After 4 stride-2 layers: spatial dims / 16
            spatial = (padded_h // 16, padded_w // 16)
            self.decoder._init_linear(spatial)

        return self.decode(z, target_shape=_DEFAULT_MEL_SHAPE)
