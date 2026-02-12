"""Composable audio augmentation pipeline for small dataset expansion.

Applies pitch shift, speed perturbation, noise injection, and volume
variation with independent per-augmentation probabilities.  Each
augmentation is gated by its own probability, so not every transform
fires on every call -- preventing over-augmentation.

Transforms are pre-created where possible to avoid per-call overhead.
PitchShift is created per-call because ``n_steps`` is set at init time.
Vol is created per-call because ``gain`` varies (but Vol is lightweight).

Heavy dependencies (torch, torchaudio) are imported inside method bodies,
matching the project-wide lazy-import pattern.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Configuration for the augmentation pipeline.

    Each probability controls how often that augmentation is applied to a
    given waveform.  Setting a probability to 0.0 disables it entirely.
    """

    pitch_shift_probability: float = 0.5
    pitch_shift_range: tuple[float, float] = (-2.0, 2.0)  # semitones
    pitch_shift_n_fft: int = 2048  # larger for 48kHz to avoid bass artifacts

    speed_probability: float = 0.5
    speed_factors: list[float] = field(
        default_factory=lambda: [0.9, 0.95, 1.0, 1.0, 1.05, 1.1],
    )  # weighted toward 1.0

    noise_probability: float = 0.3  # lower -- noise is more destructive
    noise_snr_range: tuple[float, float] = (15.0, 40.0)  # dB

    volume_probability: float = 0.5
    volume_range: tuple[float, float] = (0.7, 1.3)  # amplitude multiplier

    expansion_ratio: int = 10  # augmented copies per original


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AugmentationPipeline:
    """Apply random augmentations to expand small audio datasets.

    Each augmentation is gated by an independent probability so that
    different combinations fire on each call.  Order:
    pitch shift -> speed perturbation -> noise injection -> volume.

    Parameters
    ----------
    sample_rate:
        Audio sample rate (must match waveforms passed to :meth:`augment`).
    config:
        Augmentation configuration.  ``None`` uses default settings.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        config: AugmentationConfig | None = None,
    ) -> None:
        from torchaudio.transforms import AddNoise, SpeedPerturbation  # noqa: WPS433

        self.sample_rate = sample_rate
        self.config = config or AugmentationConfig()

        # Pre-create reusable transforms ----------------------------------
        # SpeedPerturbation: takes sample_rate and factors at init
        self._speed_perturb = SpeedPerturbation(
            orig_freq=self.sample_rate,
            factors=self.config.speed_factors,
        )
        # AddNoise: stateless, reusable
        self._add_noise = AddNoise()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def augment(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Apply augmentations with independent probabilities.

        Parameters
        ----------
        waveform:
            Float32 tensor with shape ``[channels, samples]``.

        Returns
        -------
        torch.Tensor
            Augmented waveform clamped to ``[-1.0, 1.0]``.
        """
        import torch  # noqa: WPS433
        from torchaudio.transforms import PitchShift, Vol  # noqa: WPS433

        cfg = self.config

        # 1. Pitch shift (per-call: n_steps varies)
        if random.random() < cfg.pitch_shift_probability:
            n_steps = random.uniform(*cfg.pitch_shift_range)
            pitch_shift = PitchShift(
                sample_rate=self.sample_rate,
                n_steps=n_steps,
                n_fft=cfg.pitch_shift_n_fft,
            )
            waveform = pitch_shift(waveform)

        # 2. Speed perturbation
        if random.random() < cfg.speed_probability:
            waveform, _lengths = self._speed_perturb(waveform)

        # 3. Noise injection
        if random.random() < cfg.noise_probability:
            noise = torch.randn_like(waveform)
            snr = torch.tensor([random.uniform(*cfg.noise_snr_range)])
            waveform = self._add_noise(waveform, noise, snr)

        # 4. Volume variation (per-call: gain varies)
        if random.random() < cfg.volume_probability:
            gain = random.uniform(*cfg.volume_range)
            waveform = Vol(gain=gain, gain_type="amplitude")(waveform)

        # Clamp to prevent downstream clipping issues
        waveform = waveform.clamp(-1.0, 1.0)
        return waveform

    def expand_dataset(
        self,
        waveforms: list["torch.Tensor"],
    ) -> list["torch.Tensor"]:
        """Expand a dataset by producing augmented copies of each waveform.

        For each input waveform, the original (unaugmented) copy is kept
        and ``config.expansion_ratio`` augmented variants are added.

        Parameters
        ----------
        waveforms:
            List of float32 tensors, each ``[channels, samples]``.

        Returns
        -------
        list[torch.Tensor]
            Flat list: ``[orig_1, aug_1_1, ..., aug_1_N, orig_2, ...]``.
            Total length = ``len(waveforms) * (1 + expansion_ratio)``.
        """
        expanded: list["torch.Tensor"] = []
        for waveform in waveforms:
            expanded.append(waveform)  # always keep original
            for _ in range(self.config.expansion_ratio):
                expanded.append(self.augment(waveform))
        return expanded
