"""Mel spectrogram representation layer.

Converts between raw waveforms and normalized log-mel spectrograms.
All audio entering or leaving the VAE passes through this module,
ensuring consistent mel parameters (n_fft, n_mels, hop_length, sample_rate).

v2.0 adds :class:`ComplexSpectrogram` which computes 2-channel
magnitude + instantaneous frequency (IF) spectrograms in mel domain.

Design notes:
- Lazy-imports torchaudio.transforms inside ``__init__`` (project pattern).
- ``InverseMelScale`` runs on CPU to avoid ``torch.linalg.lstsq`` issues on MPS.
- ``log1p`` / ``expm1`` normalisation compresses dynamic range and handles zeros.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SpectrogramConfig:
    """Mel spectrogram parameters -- must be consistent across training and inference.

    Default values target professional audio production (48 kHz, 128 mels).
    One second of audio at 48 kHz produces a 128x94 mel spectrogram.
    """

    sample_rate: int = 48_000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None  # defaults to sample_rate / 2
    power: float = 2.0


class AudioSpectrogram:
    """Convert between waveforms and normalized log-mel spectrograms.

    Instantiate once and reuse across batches -- internal torchaudio transforms
    allocate buffers that are expensive to re-create.

    Parameters
    ----------
    config : SpectrogramConfig | None
        Mel parameters.  ``None`` uses project defaults (48 kHz, 128 mels).
    """

    def __init__(self, config: SpectrogramConfig | None = None) -> None:
        import torch  # noqa: WPS433 -- lazy import
        from torchaudio.transforms import GriffinLim, InverseMelScale, MelSpectrogram

        self.config = config or SpectrogramConfig()
        c = self.config

        self.mel_transform = MelSpectrogram(
            sample_rate=c.sample_rate,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            n_mels=c.n_mels,
            f_min=c.f_min,
            f_max=c.f_max,
            power=c.power,
        )
        self.inverse_mel = InverseMelScale(
            n_stft=c.n_fft // 2 + 1,
            n_mels=c.n_mels,
            sample_rate=c.sample_rate,
            f_min=c.f_min,
            f_max=c.f_max,
        )
        self.griffin_lim = GriffinLim(
            n_fft=c.n_fft,
            n_iter=128,
            hop_length=c.hop_length,
            power=c.power,
        )

        # Keep a reference to torch for use in methods without re-importing
        self._torch = torch

    def to(self, device: "torch.device") -> "AudioSpectrogram":
        """Move the forward mel transform to the given device.

        Only ``mel_transform`` is moved -- ``inverse_mel`` and ``griffin_lim``
        remain on CPU (InverseMelScale requires CPU for ``torch.linalg.lstsq``).

        Returns ``self`` for chaining.
        """
        self.mel_transform = self.mel_transform.to(device)
        return self

    # ------------------------------------------------------------------
    # Forward: waveform -> mel
    # ------------------------------------------------------------------

    def waveform_to_mel(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Convert waveform to normalised log-mel spectrogram.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``[B, 1, samples]`` -- batch of mono waveforms.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, n_mels, time]`` -- normalised log-mel spectrogram.
        """
        torch = self._torch
        mel = self.mel_transform(waveform.squeeze(1))  # [B, n_mels, time]
        mel_log = torch.log1p(mel)  # log(1 + x), handles zeros gracefully
        return mel_log.unsqueeze(1)  # [B, 1, n_mels, time]

    # ------------------------------------------------------------------
    # Inverse: mel -> waveform
    # ------------------------------------------------------------------

    def mel_to_waveform(self, mel_log: "torch.Tensor") -> "torch.Tensor":
        """Convert normalised log-mel spectrogram back to waveform.

        ``InverseMelScale`` is forced to CPU to avoid
        ``torch.linalg.lstsq`` issues on MPS.

        Parameters
        ----------
        mel_log : torch.Tensor
            Shape ``[B, 1, n_mels, time]`` -- normalised log-mel spectrogram.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, samples]`` -- reconstructed mono waveforms.
        """
        torch = self._torch
        mel = torch.expm1(mel_log.squeeze(1).clamp(min=0))  # inverse of log1p
        # InverseMelScale must run on CPU (torch.linalg.lstsq MPS issues)
        linear_spec = self.inverse_mel(mel.cpu())  # [B, n_stft, time]
        waveform = self.griffin_lim(linear_spec)  # [B, samples]
        return waveform.unsqueeze(1)  # [B, 1, samples]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_mel_shape(self, num_samples: int) -> tuple[int, int]:
        """Return ``(n_mels, time_frames)`` for a given number of audio samples.

        Useful for sizing the VAE architecture before any data is loaded.
        """
        c = self.config
        # torchaudio pads by n_fft // 2 on each side, then applies hop_length
        time_frames = num_samples // c.hop_length + 1
        return (c.n_mels, time_frames)


# ======================================================================
# v2.0 -- 2-Channel Complex Spectrogram (Magnitude + IF)
# ======================================================================


class ComplexSpectrogram:
    """Convert waveforms to 2-channel magnitude + instantaneous frequency spectrograms.

    Channel 0 is a log-mel magnitude spectrogram (same as v1.0).
    Channel 1 is instantaneous frequency (IF) projected into mel domain
    and normalised to ``[-1, 1]``.  IF values in low-amplitude bins are
    masked to zero because phase is meaningless noise there.

    Parameters
    ----------
    config : ComplexSpectrogramConfig
        Configuration for STFT parameters and IF masking.  Imported from
        :mod:`distill.training.config`.  This is the **single source of
        truth** for STFT parameters in v2.0 code paths.
    """

    def __init__(self, config: "ComplexSpectrogramConfig") -> None:
        import torch  # noqa: WPS433 -- lazy import
        from torchaudio.transforms import InverseMelScale, MelScale, MelSpectrogram

        self.config = config
        self.if_masking_threshold = config.if_masking_threshold

        self.mel_transform = MelSpectrogram(
            sample_rate=48_000,  # project default
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )

        self.mel_scale = MelScale(
            n_mels=config.n_mels,
            sample_rate=48_000,
            n_stft=config.n_fft // 2 + 1,
        )

        # Pre-compute per-mel-bin weight sums for proper weighted averaging.
        # MelScale.fb is [n_stft, n_mels]; sum across frequency axis gives
        # total weight per mel bin.  Used to normalise IF mel projection so
        # that the result is a true energy-weighted average (stays in [-1,1]).
        fb_sum = self.mel_scale.fb.sum(dim=0)  # [n_mels]
        fb_sum = fb_sum.clamp(min=1e-10)  # avoid division by zero
        self._mel_fb_sum = fb_sum  # [n_mels]

        # Pre-create a hann window for STFT to avoid spectral leakage warning
        self._stft_window = torch.hann_window(config.n_fft)

        # Inverse mel scale for reconstruction (must stay on CPU -- project pattern)
        self._inverse_mel = InverseMelScale(
            n_stft=config.n_fft // 2 + 1,
            n_mels=config.n_mels,
            sample_rate=48_000,
        )

        self._torch = torch
        self._pi = math.pi

    # ------------------------------------------------------------------
    # Forward: waveform -> 2-channel mel (magnitude + IF)
    # ------------------------------------------------------------------

    def waveform_to_complex_mel(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Convert mono waveform to 2-channel magnitude + IF spectrogram.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``[B, 1, samples]`` -- batch of mono waveforms.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 2, n_mels, time]`` where channel 0 is log-mel
            magnitude and channel 1 is instantaneous frequency in
            ``[-1, 1]`` range, both in mel domain.
        """
        torch = self._torch
        pi = self._pi

        # Squeeze channel dim: [B, 1, samples] -> [B, samples]
        wav = waveform.squeeze(1)

        # -- Step 1: Mel magnitude (log1p normalised) ----------------------
        mel_power = self.mel_transform(wav)  # [B, n_mels, time]
        mel_mag = torch.log1p(mel_power)     # log(1 + x), non-negative

        # -- Step 2: STFT -> complex -> phase ------------------------------
        window = self._stft_window.to(wav.device)
        stft_complex = torch.stft(
            wav,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=window,
            return_complex=True,
        )  # [B, n_fft//2+1, time_stft]
        phase = torch.angle(stft_complex)  # [B, n_fft//2+1, time_stft]

        # -- Step 3: Instantaneous frequency (phase difference) ------------
        # Prepend zero for first frame so output length matches
        phase_diff = torch.diff(phase, dim=-1)
        zero_pad = torch.zeros_like(phase[:, :, :1])
        phase_diff = torch.cat([zero_pad, phase_diff], dim=-1)

        # Unwrap to [-pi, pi]
        if_linear = torch.remainder(phase_diff + pi, 2.0 * pi) - pi
        # Normalise to [-1, 1]
        if_linear = if_linear / pi

        # -- Step 4: Project IF to mel scale -------------------------------
        # MelScale applies the filterbank matrix (weighted sum).  Divide by
        # the per-mel-bin weight sums to get a proper weighted average so
        # that IF values stay in [-1, 1].
        if_mel = self.mel_scale(if_linear)  # [B, n_mels, time_stft]
        fb_sum = self._mel_fb_sum.to(if_mel.device)
        if_mel = if_mel / fb_sum.unsqueeze(0).unsqueeze(-1)  # [B, n_mels, time]

        # -- Step 5: IF masking (zero IF in low-energy bins) ---------------
        # mel_power is the pre-log1p mel power spectrogram
        mask = mel_power < self.if_masking_threshold
        if_mel = if_mel.masked_fill(mask, 0.0)

        # -- Step 6: Align time dimensions and stack -----------------------
        # MelSpectrogram and torch.stft may produce slightly different time
        # lengths due to padding differences.  Trim to the shorter one.
        time_len = min(mel_mag.shape[-1], if_mel.shape[-1])
        mel_mag = mel_mag[..., :time_len]
        if_mel = if_mel[..., :time_len]

        # Stack: [B, 2, n_mels, time]
        return torch.stack([mel_mag, if_mel], dim=1)

    # ------------------------------------------------------------------
    # Inverse: 2-channel mel -> waveform via ISTFT
    # ------------------------------------------------------------------

    def complex_mel_to_waveform(
        self,
        spectrogram: "torch.Tensor",
        stats: dict[str, float] | None = None,
        sample_rate: int = 48_000,
    ) -> "torch.Tensor":
        """Reconstruct audio waveform from 2-channel magnitude + IF spectrogram.

        Converts the mel-domain representation back to linear frequency,
        reconstructs phase via cumulative sum of IF, and applies ISTFT.

        Parameters
        ----------
        spectrogram : torch.Tensor
            Shape ``[B, 2, n_mels, time]`` -- normalised or raw.
        stats : dict[str, float] | None
            If provided, denormalise before reconstruction.
        sample_rate : int
            Sample rate for InverseMelScale (default 48 kHz).

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1, samples]`` -- reconstructed mono waveforms.
        """
        torch = self._torch
        pi = self._pi
        config = self.config

        # -- Step 1: Denormalize (if stats provided) --------------------------
        if stats is not None:
            spectrogram = self.denormalize(spectrogram, stats)

        # -- Step 2: Split channels -------------------------------------------
        mag_mel = spectrogram[:, 0]   # [B, n_mels, time] -- log1p domain
        if_mel = spectrogram[:, 1]    # [B, n_mels, time] -- normalised [-1, 1]

        # -- Step 3: Undo log1p on magnitude ----------------------------------
        # log1p domain -> linear mel power -> mel amplitude (sqrt for power=2.0)
        mel_power = torch.expm1(mag_mel).clamp(min=0)
        mel_amp = torch.sqrt(mel_power)  # [B, n_mels, time]

        # -- Step 4: Undo IF normalization ------------------------------------
        if_radians = if_mel * pi  # [-1,1] -> [-pi, pi]

        # -- Step 5: Reconstruct phase via cumulative sum ---------------------
        # Starting from zero at time step 0 (per user decision)
        phase_mel = torch.cumsum(if_radians, dim=-1)  # [B, n_mels, time]
        # Leave phase unwrapped (per user decision)

        # -- Step 6: Invert mel scale to linear frequency ---------------------
        # InverseMelScale must run on CPU (torch.linalg.lstsq MPS issues)
        original_device = mel_amp.device
        mel_amp_cpu = mel_amp.cpu()
        phase_mel_cpu = phase_mel.cpu()

        mag_linear = self._inverse_mel(mel_amp_cpu)     # [B, n_stft, time]
        phase_linear = self._inverse_mel(phase_mel_cpu)  # [B, n_stft, time]

        # Move back to original device
        mag_linear = mag_linear.to(original_device)
        phase_linear = phase_linear.to(original_device)

        # -- Step 7: Combine magnitude + phase into complex STFT --------------
        stft_complex = mag_linear * torch.exp(1j * phase_linear)

        # -- Step 8: Apply ISTFT ----------------------------------------------
        window = self._stft_window.to(original_device)
        waveform = torch.istft(
            stft_complex,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            window=window,
            return_complex=False,
        )  # [B, samples]

        # -- Step 9: Return [B, 1, samples] -----------------------------------
        return waveform.unsqueeze(1)

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------

    def compute_dataset_statistics(
        self, spectrograms: list["torch.Tensor"],
    ) -> dict[str, float]:
        """Compute per-channel mean and std across a dataset.

        Parameters
        ----------
        spectrograms : list[torch.Tensor]
            Unbatched 2-channel spectrograms, each ``[2, n_mels, time]``.

        Returns
        -------
        dict
            Keys: ``mag_mean``, ``mag_std``, ``if_mean``, ``if_std``.
        """
        torch = self._torch

        all_mag = torch.cat([s[0].flatten() for s in spectrograms])
        all_if = torch.cat([s[1].flatten() for s in spectrograms])

        mag_mean = all_mag.mean().item()
        mag_std = all_mag.std().item()
        if_mean = all_if.mean().item()
        if_std = all_if.std().item()

        # Guard against near-zero std (e.g. silent audio)
        if mag_std < 1e-8:
            mag_std = 1.0
        if if_std < 1e-8:
            if_std = 1.0

        return {
            "mag_mean": mag_mean,
            "mag_std": mag_std,
            "if_mean": if_mean,
            "if_std": if_std,
        }

    # ------------------------------------------------------------------
    # Normalisation / Denormalisation
    # ------------------------------------------------------------------

    def normalize(
        self, spectrogram: "torch.Tensor", stats: dict[str, float],
    ) -> "torch.Tensor":
        """Normalize a 2-channel spectrogram to zero mean and unit variance.

        Parameters
        ----------
        spectrogram : torch.Tensor
            Shape ``[B, 2, n_mels, time]`` or ``[2, n_mels, time]``.
        stats : dict
            Output of :meth:`compute_dataset_statistics`.

        Returns
        -------
        torch.Tensor
            Normalised spectrogram, same shape as input.
        """
        torch = self._torch
        has_batch = spectrogram.dim() == 4

        if has_batch:
            mag = (spectrogram[:, 0] - stats["mag_mean"]) / stats["mag_std"]
            ifr = (spectrogram[:, 1] - stats["if_mean"]) / stats["if_std"]
            return torch.stack([mag, ifr], dim=1)
        else:
            mag = (spectrogram[0] - stats["mag_mean"]) / stats["mag_std"]
            ifr = (spectrogram[1] - stats["if_mean"]) / stats["if_std"]
            return torch.stack([mag, ifr], dim=0)

    def denormalize(
        self, spectrogram: "torch.Tensor", stats: dict[str, float],
    ) -> "torch.Tensor":
        """Reverse normalisation applied by :meth:`normalize`.

        Parameters
        ----------
        spectrogram : torch.Tensor
            Normalised tensor, ``[B, 2, n_mels, time]`` or ``[2, n_mels, time]``.
        stats : dict
            Same statistics dict used for normalisation.

        Returns
        -------
        torch.Tensor
            Denormalised spectrogram, same shape as input.
        """
        torch = self._torch
        has_batch = spectrogram.dim() == 4

        if has_batch:
            mag = spectrogram[:, 0] * stats["mag_std"] + stats["mag_mean"]
            ifr = spectrogram[:, 1] * stats["if_std"] + stats["if_mean"]
            return torch.stack([mag, ifr], dim=1)
        else:
            mag = spectrogram[0] * stats["mag_std"] + stats["mag_mean"]
            ifr = spectrogram[1] * stats["if_std"] + stats["if_mean"]
            return torch.stack([mag, ifr], dim=0)

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: "torch.device") -> "ComplexSpectrogram":
        """Move transforms to the given device.

        ``_inverse_mel`` is NOT moved -- it must stay on CPU because
        ``InverseMelScale`` uses ``torch.linalg.lstsq`` which has MPS issues.

        Returns ``self`` for chaining.
        """
        self.mel_transform = self.mel_transform.to(device)
        self.mel_scale = self.mel_scale.to(device)
        self._mel_fb_sum = self._mel_fb_sum.to(device)
        self._stft_window = self._stft_window.to(device)
        # Note: self._inverse_mel stays on CPU (project pattern)
        return self
