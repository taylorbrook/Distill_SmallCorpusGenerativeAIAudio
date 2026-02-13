"""Generation pipeline orchestrating chunk synthesis, stereo, and export.

The :class:`GenerationPipeline` is the primary interface for generating audio
from trained models.  It ties together chunk generation, stereo processing,
anti-aliasing, quality metrics, and WAV export into a single coherent API.

Design notes:
- Lazy imports for ``torch``, ``numpy``, ``torchaudio`` (project pattern).
- All internal audio processing at 48 kHz; resample only at the very end.
- Resampler instances cached in a module-level dict (Phase 2 pattern).
- Peak normalisation applied after stereo processing to catch any clipping
  introduced by mid-side widening (research pitfall #3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from small_dataset_audio.inference.export import (
    BIT_DEPTH_MAP,
    SAMPLE_RATE_OPTIONS,
    export_wav,
    write_sidecar_json,
)

if TYPE_CHECKING:
    import numpy as np
    import torch

# Module-level resampler cache (project pattern from Phase 2)
_resampler_cache: dict[tuple[int, int], object] = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Configuration for audio generation.

    All user-facing options for the generation pipeline: duration,
    concatenation mode, stereo mode, sample rate, bit depth, and
    reproducibility seed.
    """

    duration_s: float = 1.0
    """Desired output duration in seconds (freeform, up to max_duration_s)."""

    max_duration_s: float = 60.0
    """Architecture limit: maximum generation duration for v1."""

    seed: int | None = None
    """Random seed for reproducibility.  ``None`` generates a random seed."""

    chunk_duration_s: float = 1.0
    """Duration of each generated chunk in seconds."""

    concat_mode: str = "crossfade"
    """Concatenation mode: ``"crossfade"`` or ``"latent_interpolation"``."""

    stereo_mode: str = "mono"
    """Stereo mode: ``"mono"``, ``"mid_side"``, or ``"dual_seed"``."""

    stereo_width: float = 0.7
    """Width for mid-side stereo mode (0.0-1.5)."""

    sample_rate: int = 48_000
    """Output sample rate in Hz."""

    bit_depth: str = "24-bit"
    """Output bit depth for WAV export."""

    steps_between: int = 10
    """Interpolation steps between anchors in latent_interpolation mode."""

    overlap_samples: int = 2400
    """Crossfade overlap in samples (50 ms at 48 kHz)."""

    latent_vector: "np.ndarray | None" = None
    """User-controlled latent vector from slider mapping.

    When set, all chunks use this vector (with small per-chunk
    perturbations for variation in multi-chunk generation).
    Overrides random sampling.  When ``None``, behaviour is unchanged
    (random latent vectors as before).
    """

    def validate(self) -> None:
        """Validate configuration, raising ValueError on invalid settings."""
        if self.duration_s <= 0 or self.duration_s > self.max_duration_s:
            raise ValueError(
                f"duration_s must be > 0 and <= {self.max_duration_s}, "
                f"got {self.duration_s}"
            )
        if self.concat_mode not in ("crossfade", "latent_interpolation"):
            raise ValueError(
                f"concat_mode must be 'crossfade' or 'latent_interpolation', "
                f"got {self.concat_mode!r}"
            )
        if self.stereo_mode not in ("mono", "mid_side", "dual_seed"):
            raise ValueError(
                f"stereo_mode must be 'mono', 'mid_side', or 'dual_seed', "
                f"got {self.stereo_mode!r}"
            )
        if self.sample_rate not in SAMPLE_RATE_OPTIONS:
            raise ValueError(
                f"sample_rate must be one of {SAMPLE_RATE_OPTIONS}, "
                f"got {self.sample_rate}"
            )
        if self.bit_depth not in BIT_DEPTH_MAP:
            raise ValueError(
                f"bit_depth must be one of {list(BIT_DEPTH_MAP.keys())}, "
                f"got {self.bit_depth!r}"
            )
        if self.latent_vector is not None:
            import numpy as np  # noqa: WPS433 -- lazy import

            if not isinstance(self.latent_vector, np.ndarray):
                raise ValueError(
                    f"latent_vector must be a numpy ndarray, "
                    f"got {type(self.latent_vector).__name__}"
                )
            if self.latent_vector.ndim != 1:
                raise ValueError(
                    f"latent_vector must be 1-D, "
                    f"got {self.latent_vector.ndim}-D"
                )


# ---------------------------------------------------------------------------
# Generation result
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Result of a generation run, containing audio data and metadata."""

    audio: "np.ndarray"
    """Generated audio: 1-D ``[samples]`` for mono, 2-D ``[2, samples]`` for stereo."""

    sample_rate: int
    """Sample rate of the audio data."""

    quality: dict
    """Quality score dictionary from :func:`compute_quality_score`."""

    config: GenerationConfig
    """Configuration used for this generation."""

    seed_used: int
    """Actual seed used (may differ from config.seed if it was None)."""

    duration_s: float
    """Actual duration of generated audio in seconds."""

    channels: int
    """Number of channels: 1 for mono, 2 for stereo."""


# ---------------------------------------------------------------------------
# Resampler helper
# ---------------------------------------------------------------------------


def _get_resampler(orig_freq: int, new_freq: int) -> object:
    """Get or create a cached torchaudio Resample transform.

    Parameters
    ----------
    orig_freq : int
        Source sample rate.
    new_freq : int
        Target sample rate.

    Returns
    -------
    torchaudio.transforms.Resample
        Cached resampler instance.
    """
    key = (orig_freq, new_freq)
    if key not in _resampler_cache:
        import torchaudio  # noqa: WPS433 -- lazy import

        _resampler_cache[key] = torchaudio.transforms.Resample(orig_freq, new_freq)
    return _resampler_cache[key]


# ---------------------------------------------------------------------------
# Slider-controlled chunk generation
# ---------------------------------------------------------------------------


def _generate_chunks_from_vector(
    model: "ConvVAE",
    spectrogram: "AudioSpectrogram",
    latent_vector: "torch.Tensor",
    num_chunks: int,
    device: "torch.device",
    seed: int,
    chunk_samples: int = 48_000,
    overlap_samples: int = 2400,
) -> "np.ndarray":
    """Generate audio from a user-provided latent vector.

    Uses the provided latent vector as a base.  For multi-chunk
    generation, adds small random perturbations (scaled by 0.1) to
    each chunk so longer audio has variation while staying in the
    same latent-space region.

    Parameters
    ----------
    model : ConvVAE
        Trained VAE model.
    spectrogram : AudioSpectrogram
        Spectrogram converter (for mel-to-waveform).
    latent_vector : torch.Tensor
        1-D latent vector on the target device.
    num_chunks : int
        Number of audio chunks to generate.
    device : torch.device
        Device the model is on.
    seed : int
        Random seed for reproducible per-chunk perturbations.
    chunk_samples : int
        Number of audio samples per chunk (default 48000 = 1 s at 48 kHz).
    overlap_samples : int
        Overlap for crossfade in samples (default 2400 = 50 ms at 48 kHz).

    Returns
    -------
    np.ndarray
        Concatenated audio as float32.
    """
    import torch  # noqa: WPS433 -- lazy import
    import numpy as np  # noqa: WPS433 -- lazy import

    from small_dataset_audio.inference.chunking import crossfade_chunks

    mel_shape = spectrogram.get_mel_shape(chunk_samples)
    torch.manual_seed(seed)

    was_training = model.training
    model.eval()

    waveforms: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                # Base vector with small perturbation for variety
                if chunk_idx == 0:
                    z = latent_vector.unsqueeze(0)  # [1, latent_dim]
                else:
                    perturbation = torch.randn_like(latent_vector) * 0.1
                    z = (latent_vector + perturbation).unsqueeze(0)

                # Ensure decoder is initialised
                if model.decoder.fc is None:
                    n_mels, time_frames = mel_shape
                    pad_h = (16 - n_mels % 16) % 16
                    pad_w = (16 - time_frames % 16) % 16
                    spatial = (
                        (n_mels + pad_h) // 16,
                        (time_frames + pad_w) // 16,
                    )
                    model.decoder._init_linear(spatial)

                # Decode latent vector to mel spectrogram
                mel = model.decode(z, target_shape=mel_shape)

                # Convert mel to waveform on CPU
                wav = spectrogram.mel_to_waveform(mel.cpu())
                waveforms.append(wav.squeeze().numpy().astype(np.float32))
    finally:
        if was_training:
            model.train()

    return crossfade_chunks(waveforms, overlap_samples)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class GenerationPipeline:
    """Orchestrator for the full audio generation pipeline.

    Ties together chunk generation, stereo processing, anti-aliasing,
    quality metrics, and WAV export.

    Parameters
    ----------
    model : ConvVAE
        Trained VAE model.
    spectrogram : AudioSpectrogram
        Spectrogram converter for mel-to-waveform conversion.
    device : torch.device
        Device the model is on.
    """

    def __init__(
        self,
        model: "ConvVAE",
        spectrogram: "AudioSpectrogram",
        device: "torch.device",
    ) -> None:
        self.model = model
        self.spectrogram = spectrogram
        self.device = device
        self.model_name: str = "unknown"

    def generate(self, config: GenerationConfig) -> GenerationResult:
        """Run the full generation pipeline.

        Pipeline stages:
        1. Validate config
        2. Generate chunks (crossfade or latent interpolation)
        3. Apply anti-aliasing filter
        4. Apply stereo processing (if not mono)
        5. Peak normalize (after stereo to catch clipping)
        6. Resample to target sample rate (if different from 48 kHz)
        7. Compute quality score
        8. Trim to exact requested duration

        Parameters
        ----------
        config : GenerationConfig
            Generation configuration.

        Returns
        -------
        GenerationResult
            Audio data with quality metrics and metadata.
        """
        import torch  # noqa: WPS433 -- lazy import
        import numpy as np  # noqa: WPS433 -- lazy import

        from small_dataset_audio.inference.chunking import (
            generate_chunks_crossfade,
            generate_chunks_latent_interp,
        )
        from small_dataset_audio.inference.stereo import (
            apply_mid_side_widening,
            create_dual_seed_stereo,
            peak_normalize,
        )
        from small_dataset_audio.audio.filters import apply_anti_alias_filter
        from small_dataset_audio.inference.quality import compute_quality_score

        # 1. Validate config
        config.validate()

        # 2. Compute chunk parameters
        internal_sr = 48_000  # all processing at 48 kHz
        num_chunks = math.ceil(config.duration_s / config.chunk_duration_s)
        chunk_samples = int(config.chunk_duration_s * internal_sr)

        # 3. Set seed
        seed_used: int
        if config.seed is not None:
            seed_used = config.seed
        else:
            seed_used = int(torch.randint(0, 2**31, (1,)).item())

        # 3.5. Prepare user-controlled latent vector (if provided)
        latent_tensor: torch.Tensor | None = None
        if config.latent_vector is not None:
            latent_tensor = (
                torch.from_numpy(config.latent_vector)
                .float()
                .to(self.device)
            )

        # 4. Generate audio based on concat_mode
        if latent_tensor is not None:
            # Slider-controlled generation: use provided latent vector
            audio = _generate_chunks_from_vector(
                model=self.model,
                spectrogram=self.spectrogram,
                latent_vector=latent_tensor,
                num_chunks=num_chunks,
                device=self.device,
                seed=seed_used,
                chunk_samples=chunk_samples,
                overlap_samples=config.overlap_samples,
            )
        elif config.concat_mode == "crossfade":
            audio = generate_chunks_crossfade(
                model=self.model,
                spectrogram=self.spectrogram,
                num_chunks=num_chunks,
                device=self.device,
                seed=seed_used,
                chunk_samples=chunk_samples,
                overlap_samples=config.overlap_samples,
            )
        else:  # latent_interpolation
            audio = generate_chunks_latent_interp(
                model=self.model,
                spectrogram=self.spectrogram,
                num_chunks=num_chunks,
                device=self.device,
                seed=seed_used,
                chunk_samples=chunk_samples,
                steps_between=config.steps_between,
            )

        # 5. Apply anti-aliasing filter (always at internal sample rate)
        audio = apply_anti_alias_filter(audio, internal_sr)

        # 6. Apply stereo processing
        channels = 1
        if config.stereo_mode == "mid_side":
            audio = apply_mid_side_widening(audio, config.stereo_width, internal_sr)
            channels = 2
        elif config.stereo_mode == "dual_seed":
            # Generate second audio stream with seed + 1
            if latent_tensor is not None:
                audio_right = _generate_chunks_from_vector(
                    model=self.model,
                    spectrogram=self.spectrogram,
                    latent_vector=latent_tensor,
                    num_chunks=num_chunks,
                    device=self.device,
                    seed=seed_used + 1,
                    chunk_samples=chunk_samples,
                    overlap_samples=config.overlap_samples,
                )
            elif config.concat_mode == "crossfade":
                audio_right = generate_chunks_crossfade(
                    model=self.model,
                    spectrogram=self.spectrogram,
                    num_chunks=num_chunks,
                    device=self.device,
                    seed=seed_used + 1,
                    chunk_samples=chunk_samples,
                    overlap_samples=config.overlap_samples,
                )
            else:
                audio_right = generate_chunks_latent_interp(
                    model=self.model,
                    spectrogram=self.spectrogram,
                    num_chunks=num_chunks,
                    device=self.device,
                    seed=seed_used + 1,
                    chunk_samples=chunk_samples,
                    steps_between=config.steps_between,
                )
            audio_right = apply_anti_alias_filter(audio_right, internal_sr)
            audio = create_dual_seed_stereo(audio, audio_right)
            channels = 2

        # 7. Peak normalize after stereo (catches clipping from mid-side)
        audio = peak_normalize(audio, target_peak=0.891)

        # 8. Resample if target sample rate differs from internal 48 kHz
        if config.sample_rate != internal_sr:
            resampler = _get_resampler(internal_sr, config.sample_rate)
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # [1, samples]
            audio_tensor = resampler(audio_tensor)
            if channels == 1:
                audio = audio_tensor.squeeze(0).numpy()
            else:
                audio = audio_tensor.numpy()

        # 9. Compute quality score
        quality = compute_quality_score(audio, config.sample_rate)

        # 10. Trim to exact requested duration
        target_samples = int(config.duration_s * config.sample_rate)
        if audio.ndim == 1:
            audio = audio[:target_samples]
        else:
            audio = audio[:, :target_samples]

        actual_duration = audio.shape[-1] / config.sample_rate

        return GenerationResult(
            audio=audio,
            sample_rate=config.sample_rate,
            quality=quality,
            config=config,
            seed_used=seed_used,
            duration_s=actual_duration,
            channels=channels,
        )

    def export(
        self,
        result: GenerationResult,
        output_dir: Path,
        filename: str | None = None,
    ) -> tuple[Path, Path]:
        """Export a generation result as WAV with sidecar JSON.

        Parameters
        ----------
        result : GenerationResult
            The generation result to export.
        output_dir : Path
            Directory for output files.  Created if it doesn't exist.
        filename : str | None
            WAV filename (without extension).  Auto-generated if ``None``.

        Returns
        -------
        tuple[Path, Path]
            ``(wav_path, json_path)`` of the exported files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-generate filename if not provided
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_seed{result.seed_used}.wav"
        elif not filename.endswith(".wav"):
            filename = filename + ".wav"

        wav_path = output_dir / filename

        # Build generation config dict for sidecar
        config_dict = asdict(result.config)
        # Convert numpy latent_vector to JSON-serialisable list
        if config_dict.get("latent_vector") is not None:
            config_dict["latent_vector"] = config_dict["latent_vector"].tolist()

        # Write sidecar JSON first (research pitfall #6)
        json_path = write_sidecar_json(
            wav_path=wav_path,
            model_name=self.model_name,
            generation_config=config_dict,
            seed=result.seed_used,
            quality_metrics=result.quality,
            duration_s=result.duration_s,
            sample_rate=result.sample_rate,
            bit_depth=result.config.bit_depth,
            channels=result.channels,
        )

        # Export WAV
        export_wav(
            audio=result.audio,
            path=wav_path,
            sample_rate=result.sample_rate,
            bit_depth=result.config.bit_depth,
        )

        return (wav_path, json_path)
