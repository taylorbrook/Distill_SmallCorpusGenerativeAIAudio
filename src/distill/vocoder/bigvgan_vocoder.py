"""BigVGAN-v2 universal vocoder implementation.

Wraps the vendored NVIDIA BigVGAN source code, providing automatic
weight downloading, cross-platform device support, and the standard
VocoderBase interface for mel-to-waveform conversion.

IMPORTANT: This module accepts mel spectrograms that are ALREADY in
BigVGAN's expected format (log-clamp, Slaney, 44.1kHz). The MelAdapter
(Plan 03) handles conversion from VAE format. Direct callers must
provide correctly formatted mels.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from distill.vocoder.base import VocoderBase

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

BIGVGAN_REPO_ID = "nvidia/bigvgan_v2_44khz_128band_512x"


def _import_bigvgan():
    """Import vendored BigVGAN module with sys.path manipulation.

    Adds vendor/bigvgan/ to sys.path so BigVGAN's internal relative
    imports (``from env import AttrDict``, ``from activations import ...``)
    resolve correctly. The path is left in place because BigVGAN's
    internal imports need it to remain available during model construction.
    """
    vendor_dir = str(Path(__file__).resolve().parents[3] / "vendor" / "bigvgan")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    import bigvgan as bigvgan_module

    return bigvgan_module


class BigVGANVocoder(VocoderBase):
    """BigVGAN-v2 universal neural vocoder.

    Wraps the vendored NVIDIA BigVGAN 122M-parameter model with automatic
    weight downloading and cross-platform device support.

    Parameters
    ----------
    device : str
        Device preference: "auto", "cuda", "mps", or "cpu".
        When "auto", uses the project's device auto-detection
        (CUDA > MPS > CPU with smoke testing).

    Notes
    -----
    - Uses ``use_cuda_kernel=False`` for cross-platform compatibility.
    - Weights are downloaded automatically on first use via HuggingFace Hub.
    - After first download, no network access is needed.
    """

    def __init__(self, device: str = "auto") -> None:
        import torch

        from distill.hardware.device import select_device
        from distill.vocoder.weight_manager import ensure_bigvgan_weights

        # Resolve device
        self._device: torch.device = select_device(device)
        logger.info("BigVGANVocoder: using device %s", self._device)

        # Ensure weights are downloaded (logs progress on first use)
        model_dir = ensure_bigvgan_weights()

        # Import vendored BigVGAN and load model from cached directory
        bigvgan_module = _import_bigvgan()
        logger.info("Loading BigVGAN model from %s...", model_dir)

        # Load directly from the cached directory to avoid huggingface_hub
        # mixin API compatibility issues with the vendored code.
        h = bigvgan_module.load_hparams_from_json(str(model_dir / "config.json"))
        self._model = bigvgan_module.BigVGAN(
            h,
            use_cuda_kernel=False,  # MUST be False for MPS/CPU compatibility
        )

        # Load generator weights
        checkpoint_dict = torch.load(
            str(model_dir / "bigvgan_generator.pt"),
            map_location="cpu",
        )
        try:
            self._model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            self._model.remove_weight_norm()
            self._model.load_state_dict(checkpoint_dict["generator"])

        self._model.remove_weight_norm()
        self._model.eval()
        self._model.to(self._device)
        logger.info("BigVGAN model loaded and ready on %s", self._device)

    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to waveform.

        NOTE: At this stage, this method expects mels in BigVGAN's native
        format: log(clamp(slaney_mel, 1e-5)), shape [B, 128, T].
        Plan 03 adds the MelAdapter that converts VAE format automatically.

        After Plan 03: Accepts VAE format [B, 1, 128, T] log1p mels.

        Parameters
        ----------
        mel : torch.Tensor
            Mel spectrogram in BigVGAN format. Shape: [B, 128, T].

        Returns
        -------
        torch.Tensor
            Waveform at 44100 Hz. Shape: [B, 1, samples].
        """
        import torch

        mel = mel.to(self._device)

        with torch.inference_mode():
            wav = self._model(mel)  # [B, 1, T*512]

        return wav

    @property
    def sample_rate(self) -> int:
        """Native output sample rate: 44100 Hz."""
        return 44100

    def to(self, device: torch.device) -> BigVGANVocoder:
        """Move vocoder to device. Returns self for chaining."""
        import torch

        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._model.to(device)
        return self
