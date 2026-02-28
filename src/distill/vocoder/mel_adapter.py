"""Mel adapter: converts VAE mel representation to BigVGAN mel representation.

The VAE produces: log1p(power_mel_htk) at 48kHz
BigVGAN expects: log(clamp(magnitude_mel_slaney, 1e-5)) at 44.1kHz

Conversion strategy (direct mel-domain filterbank transfer):
1. Undo log1p normalization -> linear power mel (HTK, 48kHz)
2. Apply a precomputed transfer matrix to convert HTK mel bands to Slaney mel bands
3. Interpolate time axis for 48kHz -> 44.1kHz sample rate change
4. Apply BigVGAN normalization: log(clamp(mel, 1e-5))

This approach avoids the waveform round-trip through Griffin-Lim entirely:
- **Faster**: single matrix multiply vs iterative Griffin-Lim
- **No quality loss**: no phase reconstruction artifacts
- **Deterministic**: no random phase initialization

The transfer matrix is computed as: T = S @ pinv_reg(H^T)
where H is the HTK mel filterbank [n_freqs, n_mels] and
S is the Slaney mel filterbank [n_mels, n_freqs], with
Tikhonov regularization for numerical stability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# BigVGAN 44kHz 128-band model parameters (from config.json)
_BIGVGAN_N_FFT = 2048
_BIGVGAN_NUM_MELS = 128
_BIGVGAN_SAMPLING_RATE = 44100
_BIGVGAN_HOP_SIZE = 512
_BIGVGAN_WIN_SIZE = 2048
_BIGVGAN_FMIN = 0
_BIGVGAN_FMAX = None  # Nyquist (22050 Hz for 44.1kHz)

# VAE spectrogram parameters (project defaults)
_VAE_SAMPLE_RATE = 48000
_VAE_N_FFT = 2048
_VAE_N_MELS = 128
_VAE_F_MIN = 0.0

# Regularization strength for pseudo-inverse computation.
# Balances reconstruction accuracy vs numerical stability.
_PINV_ALPHA = 1e-4


class MelAdapter:
    """Convert VAE log1p-HTK mels to BigVGAN log-clamp-Slaney mels.

    Uses a direct mel-domain filterbank transfer:
    1. Undo log1p -> linear power mel (HTK, 48kHz)
    2. Transfer matrix: HTK mel bands -> Slaney mel bands
    3. Time interpolation for 48kHz -> 44.1kHz frame alignment
    4. BigVGAN normalization: log(clamp(mel, 1e-5))

    The transfer matrix maps between the two mel filterbank representations
    without reconstructing an intermediate waveform.

    Parameters
    ----------
    spectrogram_config : SpectrogramConfig | None
        Configuration for the VAE-side mel parameters.  ``None`` uses
        project defaults (48kHz, 128 mels, n_fft=2048).

    Notes
    -----
    All intermediate computation runs on CPU for consistency and
    compatibility. The caller (BigVGANVocoder) moves the final output
    to the model's device before inference.
    """

    def __init__(
        self,
        spectrogram_config: "SpectrogramConfig | None" = None,
    ) -> None:
        import torch
        import torchaudio
        import librosa
        import numpy as np

        from distill.audio.spectrogram import SpectrogramConfig

        self._config = spectrogram_config or SpectrogramConfig()
        c = self._config

        # Build HTK mel filterbank (project/VAE format)
        htk_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=c.n_fft // 2 + 1,
            f_min=c.f_min,
            f_max=c.f_max if c.f_max is not None else c.sample_rate / 2,
            n_mels=c.n_mels,
            sample_rate=c.sample_rate,
        )  # [n_freqs, n_mels]

        # Build Slaney mel filterbank (BigVGAN format)
        slaney_np = librosa.filters.mel(
            sr=_BIGVGAN_SAMPLING_RATE,
            n_fft=_BIGVGAN_N_FFT,
            n_mels=_BIGVGAN_NUM_MELS,
            fmin=_BIGVGAN_FMIN,
            fmax=_BIGVGAN_FMAX,
            norm="slaney",
            htk=False,
        )
        slaney_fb = torch.from_numpy(slaney_np).float()  # [n_mels, n_freqs]

        # Compute Tikhonov-regularized transfer matrix
        # Solve: T @ H^T = S  where H is [n_freqs, n_mels], S is [n_mels, n_freqs]
        # T = S @ H^T_T @ (H^T @ H^T_T + alpha*I)^{-1}  (regularized pseudo-inverse)
        h_t = htk_fb.T  # [n_mels, n_freqs]
        gram = h_t @ h_t.T + _PINV_ALPHA * torch.eye(c.n_mels)
        pinv_reg = h_t.T @ torch.linalg.inv(gram)  # [n_freqs, n_mels]
        self._transfer_matrix = slaney_fb @ pinv_reg  # [n_mels, n_mels]

        # Time resampling ratio: mel frames at 48kHz -> frames at 44.1kHz
        # Both use hop_size=512, so ratio is purely from sample rate change.
        # T_44k = T_48k * (48000 / 44100) since more frames cover the same duration
        # at the lower sample rate with same hop size.
        self._time_ratio = c.sample_rate / _BIGVGAN_SAMPLING_RATE

        logger.info(
            "MelAdapter initialized: %dHz -> %dHz (direct filterbank transfer, "
            "time ratio=%.4f)",
            c.sample_rate,
            _BIGVGAN_SAMPLING_RATE,
            self._time_ratio,
        )

    def convert(self, mel_vae: "torch.Tensor") -> "torch.Tensor":
        """Convert VAE-format mel to BigVGAN-format mel.

        Parameters
        ----------
        mel_vae : torch.Tensor
            VAE output mel spectrogram in log1p format.
            Shape: [B, 1, 128, T]

        Returns
        -------
        torch.Tensor
            BigVGAN-format mel spectrogram in log-clamp format.
            Shape: [B, 128, T'] where T' may differ from T due to
            sample rate resampling (44.1kHz vs 48kHz).
        """
        import torch
        import torch.nn.functional as F

        # Squeeze channel dimension: [B, 1, 128, T] -> [B, 128, T]
        mel = mel_vae.squeeze(1)

        # Undo log1p normalization: expm1 -> linear power mel
        mel_linear = torch.expm1(mel.clamp(min=0))  # [B, 128, T]

        # Apply filterbank transfer: HTK mel bands -> Slaney mel bands
        # transfer_matrix is [128, 128], mel_linear is [B, 128, T]
        transfer = self._transfer_matrix.to(mel_linear.device)
        mel_slaney = torch.matmul(transfer, mel_linear)  # [B, 128, T]

        # Interpolate time axis for sample rate change (48kHz -> 44.1kHz)
        # At 48kHz with hop=512, each frame is 512/48000 = 10.667ms
        # At 44.1kHz with hop=512, each frame is 512/44100 = 11.610ms
        # So the 48kHz signal has MORE frames per second -> we need more
        # frames at the target rate: T' = round(T * 48000/44100)
        t_in = mel_slaney.shape[-1]
        t_out = round(t_in * self._time_ratio)
        if t_out != t_in:
            # interpolate expects [B, C, W] for 1D mode='linear'
            mel_slaney = F.interpolate(
                mel_slaney, size=t_out, mode="linear", align_corners=False,
            )

        # Apply BigVGAN normalization: log(clamp(mel, 1e-5))
        mel_bigvgan = torch.log(torch.clamp(mel_slaney, min=1e-5))

        return mel_bigvgan  # [B, 128, T']
