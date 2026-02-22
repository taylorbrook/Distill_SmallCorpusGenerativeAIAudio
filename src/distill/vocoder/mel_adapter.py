"""Mel adapter: converts VAE mel representation to BigVGAN mel representation.

The VAE produces: log1p(power_mel_htk) at 48kHz
BigVGAN expects: log(clamp(magnitude_mel_slaney, 1e-5)) at 44.1kHz

Conversion strategy (waveform round-trip):
1. Undo log1p normalization -> power mel (HTK, 48kHz)
2. Reconstruct approximate waveform via Griffin-Lim (uses existing AudioSpectrogram)
3. Resample waveform 48kHz -> 44.1kHz
4. Compute BigVGAN-format mel using vendored meldataset.mel_spectrogram()

This approach is guaranteed to produce correct BigVGAN mels because
it uses BigVGAN's own mel computation on a real waveform. The Griffin-Lim
intermediate step introduces some quality loss, but BigVGAN's neural
reconstruction is robust to this.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
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


class MelAdapter:
    """Convert VAE log1p-HTK mels to BigVGAN log-clamp-Slaney mels.

    Uses a waveform round-trip approach:
    1. VAE mel (log1p, HTK, 48kHz) -> approximate waveform via Griffin-Lim
    2. Resample waveform 48kHz -> 44.1kHz
    3. Compute BigVGAN mel using vendored meldataset.mel_spectrogram()

    Parameters
    ----------
    spectrogram_config : SpectrogramConfig | None
        Configuration for the AudioSpectrogram instance used for Griffin-Lim
        reconstruction. ``None`` uses project defaults (48kHz, 128 mels).

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
        import torchaudio

        from distill.audio.spectrogram import AudioSpectrogram, SpectrogramConfig

        self._config = spectrogram_config or SpectrogramConfig()
        self._spectrogram = AudioSpectrogram(self._config)
        self._resampler = torchaudio.transforms.Resample(
            orig_freq=self._config.sample_rate,
            new_freq=_BIGVGAN_SAMPLING_RATE,
        )

        # Import vendored meldataset for BigVGAN mel computation
        self._mel_spectrogram = _import_vendored_mel_spectrogram()

        logger.info(
            "MelAdapter initialized: %dHz -> %dHz (waveform round-trip)",
            self._config.sample_rate,
            _BIGVGAN_SAMPLING_RATE,
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
            resampling (44.1kHz vs 48kHz).
        """
        import torch

        # Step 1: Reconstruct approximate waveform from VAE mel via Griffin-Lim
        # AudioSpectrogram.mel_to_waveform handles: expm1 -> InverseMelScale -> Griffin-Lim
        # Input: [B, 1, 128, T] -> Output: [B, 1, samples] at 48kHz
        waveform_48k = self._spectrogram.mel_to_waveform(mel_vae.cpu())

        # Step 2: Resample 48kHz -> 44.1kHz
        # Resampler expects [B, samples] or [B, C, samples]
        waveform_44k = self._resampler(waveform_48k.squeeze(1))  # [B, samples]

        # Normalize to [-1, 1] range as BigVGAN expects (matching training)
        peak = waveform_44k.abs().max()
        if peak > 0:
            waveform_44k = waveform_44k / peak * 0.95

        # Step 3: Compute BigVGAN mel using vendored meldataset.mel_spectrogram()
        # The vendored function expects [B, samples] and returns [B, 128, T']
        mel_bigvgan = self._mel_spectrogram(
            waveform_44k,
            _BIGVGAN_N_FFT,
            _BIGVGAN_NUM_MELS,
            _BIGVGAN_SAMPLING_RATE,
            _BIGVGAN_HOP_SIZE,
            _BIGVGAN_WIN_SIZE,
            _BIGVGAN_FMIN,
            _BIGVGAN_FMAX,
        )

        return mel_bigvgan  # [B, 128, T']


def _import_vendored_mel_spectrogram():
    """Import mel_spectrogram from vendored BigVGAN meldataset.

    Uses sys.path manipulation to resolve BigVGAN's internal imports.
    Returns the mel_spectrogram function.
    """
    vendor_dir = str(Path(__file__).resolve().parents[3] / "vendor" / "bigvgan")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)

    from meldataset import mel_spectrogram

    return mel_spectrogram
