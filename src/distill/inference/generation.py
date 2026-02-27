"""Generation pipeline orchestrating chunk synthesis, spatial audio, and export.

The :class:`GenerationPipeline` is the primary interface for generating audio
from trained models.  It ties together chunk generation, spatial processing,
anti-aliasing, quality metrics, and multi-format export into a single
coherent API.

Design notes:
- Lazy imports for ``torch``, ``numpy``, ``torchaudio`` (project pattern).
- All internal audio processing at vocoder native rate (44.1 kHz for BigVGAN).
- Kaiser-windowed sinc resampler instances cached in a module-level dict
  (Phase 2 pattern).
- Peak normalisation applied after spatial processing to catch any clipping
  introduced by mid-side widening (research pitfall #3).
- SpatialConfig replaces old stereo_mode/stereo_width; backward compat
  preserved via migrate_stereo_config (Phase 10).
- export_audio dispatches to WAV/MP3/FLAC/OGG with metadata embedding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from distill.inference.export import (
    BIT_DEPTH_MAP,
    FORMAT_EXTENSIONS,
    SAMPLE_RATE_OPTIONS,
    ExportFormat,
    export_audio,
    export_wav,
    write_sidecar_json,
)

if TYPE_CHECKING:
    import numpy as np
    import torch

    from distill.inference.spatial import SpatialConfig

# Module-level resampler cache (project pattern from Phase 2)
_resampler_cache: dict[tuple[int, int], object] = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Configuration for audio generation.

    All user-facing options for the generation pipeline: duration,
    concatenation mode, spatial mode, sample rate, bit depth, export
    format, and reproducibility seed.
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

    # -- Deprecated: use ``spatial`` instead ---------------------------------
    stereo_mode: str = "mono"
    """DEPRECATED: Use ``spatial`` field.  Kept for backward compatibility."""

    stereo_width: float = 0.7
    """DEPRECATED: Use ``spatial`` field.  Kept for backward compatibility."""
    # ------------------------------------------------------------------------

    spatial: "SpatialConfig | None" = None
    """Spatial audio config (mode/width/depth).

    When set, replaces old ``stereo_mode`` / ``stereo_width`` behaviour.
    When ``None``, the pipeline transparently migrates the legacy fields
    via :func:`migrate_stereo_config`.
    """

    export_format: str = "wav"
    """Export format string: ``"wav"``, ``"mp3"``, ``"flac"``, or ``"ogg"``."""

    sample_rate: int = 44_100
    """Output sample rate in Hz."""

    bit_depth: str = "24-bit"
    """Output bit depth for WAV export."""

    steps_between: int = 10
    """Interpolation steps between anchors in latent_interpolation mode."""

    overlap_samples: int = 2400
    """Crossfade overlap in samples (50 ms at 48 kHz)."""

    evolution_amount: float = 0.5
    """Latent-space drift for multi-chunk slider generation.

    Controls how far the latent vector walks from the slider position
    across chunks via SLERP interpolation.  ``0`` = static (all chunks
    identical), ``1`` = full drift toward a random anchor.  Only affects
    slider-controlled generation (when ``latent_vector`` is set).
    """

    latent_vector: "np.ndarray | None" = None
    """User-controlled latent vector from slider mapping.

    When set, all chunks use this vector (with small per-chunk
    perturbations for variation in multi-chunk generation).
    Overrides random sampling.  When ``None``, behaviour is unchanged
    (random latent vectors as before).
    """

    _VALID_EXPORT_FORMATS = ("wav", "mp3", "flac", "ogg")

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
        # Validate spatial config (or legacy stereo fields)
        if self.spatial is not None:
            self.spatial.validate()
        else:
            # Legacy validation for backward compatibility
            if self.stereo_mode not in ("mono", "mid_side", "dual_seed"):
                raise ValueError(
                    f"stereo_mode must be 'mono', 'mid_side', or 'dual_seed', "
                    f"got {self.stereo_mode!r}"
                )
        # Validate export format
        if self.export_format not in self._VALID_EXPORT_FORMATS:
            raise ValueError(
                f"export_format must be one of {self._VALID_EXPORT_FORMATS}, "
                f"got {self.export_format!r}"
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

    def get_spatial_config(self) -> "SpatialConfig":
        """Return SpatialConfig, migrating from legacy fields if needed.

        Returns
        -------
        SpatialConfig
            Effective spatial configuration for this generation.
        """
        if self.spatial is not None:
            return self.spatial
        from distill.inference.spatial import migrate_stereo_config  # noqa: WPS433
        return migrate_stereo_config(self.stereo_mode, self.stereo_width)


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

        _resampler_cache[key] = torchaudio.transforms.Resample(
            orig_freq,
            new_freq,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
        )
    return _resampler_cache[key]


# ---------------------------------------------------------------------------
# Vocoder OOM fallback helper
# ---------------------------------------------------------------------------


def _vocoder_with_fallback(
    vocoder: "VocoderBase",
    mel: "torch.Tensor",
    original_device: "torch.device",
) -> "torch.Tensor":
    """Run vocoder mel-to-waveform with GPU OOM fallback to CPU.

    Parameters
    ----------
    vocoder : VocoderBase
        Vocoder instance.
    mel : torch.Tensor
        Mel spectrogram tensor.
    original_device : torch.device
        Device the vocoder was originally on (for restoration).

    Returns
    -------
    torch.Tensor
        Waveform tensor.
    """
    import logging  # noqa: WPS433 -- lazy import

    logger = logging.getLogger(__name__)

    try:
        return vocoder.mel_to_waveform(mel)
    except RuntimeError as exc:
        if "out of memory" not in str(exc):
            raise
        logger.warning(
            "GPU OOM during vocoder inference; falling back to CPU"
        )
        import torch  # noqa: WPS433 -- lazy import

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            vocoder.to(torch.device("cpu"))
            result = vocoder.mel_to_waveform(mel.cpu())
            return result
        finally:
            vocoder.to(original_device)


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
    evolution_amount: float = 0.5,
    vocoder: "VocoderBase | None" = None,
) -> "np.ndarray":
    """Generate continuous audio from a user-provided latent vector.

    Builds a multi-anchor SLERP trajectory starting from the user's
    slider position, with each anchor drifted toward a random target
    by ``evolution_amount``.  The trajectory is decoded via overlap-add
    continuous synthesis (50% Hann overlap), producing seamless audio
    with no chunk boundaries.

    Parameters
    ----------
    model : ConvVAE
        Trained VAE model.
    spectrogram : AudioSpectrogram
        Spectrogram converter (for mel shape computation).
    latent_vector : torch.Tensor
        1-D latent vector on the target device.
    num_chunks : int
        Target duration in chunks (each chunk_samples long).
    device : torch.device
        Device the model is on.
    seed : int
        Random seed for reproducible evolution path.
    chunk_samples : int
        Number of audio samples per chunk (default 48000 = 1 s at 48 kHz).
    overlap_samples : int
        Kept for API compatibility.
    evolution_amount : float
        How far each anchor drifts from the starting vector (0-1).
        0 = static, 1 = anchors are fully random targets.
    vocoder : VocoderBase | None
        Vocoder for mel-to-waveform conversion.

    Returns
    -------
    np.ndarray
        Audio as float32.
    """
    import torch  # noqa: WPS433 -- lazy import
    import numpy as np  # noqa: WPS433 -- lazy import

    from distill.inference.chunking import (
        slerp,
        _compute_num_decode_steps,
        _interpolate_trajectory,
        synthesize_continuous_mel,
    )

    torch.manual_seed(seed)

    num_steps, _, _ = _compute_num_decode_steps(
        spectrogram, num_chunks, chunk_samples,
    )

    if num_steps == 1 or evolution_amount == 0:
        # Static: replicate user's vector for all decode steps
        trajectory = [latent_vector.unsqueeze(0)] * max(num_steps, 1)
    else:
        # Multi-anchor trajectory: user's position + evolved anchors
        # One anchor roughly every 1.5 seconds of audio
        num_anchors = max(2, (num_chunks * 2 + 2) // 3)

        anchors_1d = [latent_vector]
        for _ in range(num_anchors - 1):
            random_target = torch.randn_like(latent_vector)
            anchor = slerp(latent_vector, random_target, evolution_amount)
            anchors_1d.append(anchor)

        anchors = [a.unsqueeze(0) for a in anchors_1d]
        trajectory = _interpolate_trajectory(anchors, num_steps)

    combined_mel = synthesize_continuous_mel(
        model, spectrogram, trajectory, chunk_samples,
    )
    wav = _vocoder_with_fallback(vocoder, combined_mel, vocoder._device)
    return wav.squeeze().cpu().numpy().astype(np.float32)


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
        Spectrogram converter (for mel shape computation).
    device : torch.device
        Device the model is on.
    vocoder : VocoderBase | None
        Neural vocoder for mel-to-waveform conversion.  When ``None``,
        a default BigVGAN vocoder is created via :func:`get_vocoder`.
    """

    def __init__(
        self,
        model: "ConvVAE",
        spectrogram: "AudioSpectrogram",
        device: "torch.device",
        vocoder: "VocoderBase | None" = None,
    ) -> None:
        from distill.vocoder import get_vocoder  # noqa: WPS433 -- lazy import

        self.model = model
        self.spectrogram = spectrogram
        self.device = device
        self.vocoder = vocoder or get_vocoder("bigvgan", device=str(device))
        self.model_name: str = "unknown"

    def generate(self, config: GenerationConfig) -> GenerationResult:
        """Run the full generation pipeline.

        Pipeline stages:
        1. Validate config
        2. Resolve spatial config (new or migrated from legacy fields)
        3. Generate chunks (crossfade or latent interpolation)
        4. Apply anti-aliasing filter
        5. Apply spatial processing (mono/stereo/binaural)
        6. Peak normalize (after spatial to catch clipping)
        7. Resample to target sample rate (if different from vocoder native rate)
        8. Compute quality score
        9. Trim to exact requested duration

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

        from distill.inference.chunking import (
            generate_chunks_crossfade,
            generate_chunks_latent_interp,
        )
        from distill.inference.stereo import peak_normalize
        from distill.inference.spatial import (
            SpatialMode,
            apply_spatial,
            apply_spatial_to_dual_seed,
        )
        from distill.audio.filters import apply_anti_alias_filter
        from distill.inference.quality import compute_quality_score

        # 1. Validate config
        config.validate()

        # 2. Resolve spatial config
        spatial_config = config.get_spatial_config()

        # 3. Compute chunk parameters
        internal_sr = self.vocoder.sample_rate  # vocoder native rate (44100 for BigVGAN)
        num_chunks = math.ceil(config.duration_s / config.chunk_duration_s)
        chunk_samples = int(config.chunk_duration_s * self.spectrogram.config.sample_rate)

        # 4. Set seed
        seed_used: int
        if config.seed is not None:
            seed_used = config.seed
        else:
            seed_used = int(torch.randint(0, 2**31, (1,)).item())

        # 4.5. Prepare user-controlled latent vector (if provided)
        latent_tensor: torch.Tensor | None = None
        if config.latent_vector is not None:
            latent_tensor = (
                torch.from_numpy(config.latent_vector)
                .float()
                .to(self.device)
            )

        # 5. Generate audio based on concat_mode
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
                evolution_amount=config.evolution_amount,
                vocoder=self.vocoder,
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
                vocoder=self.vocoder,
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
                vocoder=self.vocoder,
            )

        # 6. Apply anti-aliasing filter (always at internal sample rate)
        audio = apply_anti_alias_filter(audio, internal_sr)

        # 7. Apply spatial processing
        channels = 1
        if spatial_config.mode == SpatialMode.MONO:
            # No spatial processing needed
            pass
        elif spatial_config.mode == SpatialMode.STEREO:
            # Check if we need dual-seed generation (backward compat with
            # old "dual_seed" stereo mode or when explicitly requested)
            is_dual_seed = (
                config.spatial is None and config.stereo_mode == "dual_seed"
            )
            if is_dual_seed:
                # Generate right channel with seed+1
                audio_right = self._generate_right_channel(
                    config, latent_tensor, num_chunks, seed_used,
                    chunk_samples, internal_sr,
                )
                audio_right = apply_anti_alias_filter(audio_right, internal_sr)
                audio = apply_spatial_to_dual_seed(
                    audio, audio_right, spatial_config, internal_sr,
                )
            else:
                audio = apply_spatial(audio, spatial_config, internal_sr)
            channels = 2
        elif spatial_config.mode == SpatialMode.BINAURAL:
            # For binaural: if dual-seed origin, combine to mono first
            is_dual_seed = (
                config.spatial is None and config.stereo_mode == "dual_seed"
            )
            if is_dual_seed:
                audio_right = self._generate_right_channel(
                    config, latent_tensor, num_chunks, seed_used,
                    chunk_samples, internal_sr,
                )
                audio_right = apply_anti_alias_filter(audio_right, internal_sr)
                audio = apply_spatial_to_dual_seed(
                    audio, audio_right, spatial_config, internal_sr,
                )
            else:
                audio = apply_spatial(audio, spatial_config, internal_sr)
            channels = 2

        # 8. Peak normalize after spatial (catches clipping from widening)
        audio = peak_normalize(audio, target_peak=0.891)

        # 9. Resample if target sample rate differs from vocoder native rate
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

        # 10. Compute quality score
        quality = compute_quality_score(audio, config.sample_rate)

        # 11. Trim to exact requested duration
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

    def _generate_right_channel(
        self,
        config: GenerationConfig,
        latent_tensor: "torch.Tensor | None",
        num_chunks: int,
        seed_used: int,
        chunk_samples: int,
        internal_sr: int,
    ) -> "np.ndarray":
        """Generate the right channel for dual-seed stereo.

        Parameters
        ----------
        config : GenerationConfig
            Generation configuration.
        latent_tensor : torch.Tensor | None
            User-provided latent vector (or None).
        num_chunks : int
            Number of chunks to generate.
        seed_used : int
            Base seed (right channel uses seed + 1).
        chunk_samples : int
            Samples per chunk.
        internal_sr : int
            Internal sample rate (48 kHz).

        Returns
        -------
        np.ndarray
            Right channel audio.
        """
        from distill.inference.chunking import (
            generate_chunks_crossfade,
            generate_chunks_latent_interp,
        )

        if latent_tensor is not None:
            return _generate_chunks_from_vector(
                model=self.model,
                spectrogram=self.spectrogram,
                latent_vector=latent_tensor,
                num_chunks=num_chunks,
                device=self.device,
                seed=seed_used + 1,
                chunk_samples=chunk_samples,
                overlap_samples=config.overlap_samples,
                evolution_amount=config.evolution_amount,
                vocoder=self.vocoder,
            )
        elif config.concat_mode == "crossfade":
            return generate_chunks_crossfade(
                model=self.model,
                spectrogram=self.spectrogram,
                num_chunks=num_chunks,
                device=self.device,
                seed=seed_used + 1,
                chunk_samples=chunk_samples,
                overlap_samples=config.overlap_samples,
                vocoder=self.vocoder,
            )
        else:
            return generate_chunks_latent_interp(
                model=self.model,
                spectrogram=self.spectrogram,
                num_chunks=num_chunks,
                device=self.device,
                seed=seed_used + 1,
                chunk_samples=chunk_samples,
                steps_between=config.steps_between,
                vocoder=self.vocoder,
            )

    def export(
        self,
        result: GenerationResult,
        output_dir: Path,
        filename: str | None = None,
        export_format: "ExportFormat | None" = None,
        metadata: dict | None = None,
    ) -> tuple[Path, Path]:
        """Export a generation result in the specified format with sidecar JSON.

        Parameters
        ----------
        result : GenerationResult
            The generation result to export.
        output_dir : Path
            Directory for output files.  Created if it doesn't exist.
        filename : str | None
            Base filename (without extension).  Auto-generated if ``None``.
        export_format : ExportFormat | None
            Target format.  If ``None``, uses the format from
            ``result.config.export_format``.
        metadata : dict | None
            Metadata dict for tag embedding.  If ``None``, no tags are
            embedded (WAV format never embeds tags regardless).

        Returns
        -------
        tuple[Path, Path]
            ``(audio_path, json_path)`` of the exported files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve export format
        if export_format is None:
            export_format = ExportFormat(result.config.export_format)

        # Get correct file extension
        extension = FORMAT_EXTENSIONS[export_format]

        # Auto-generate filename if not provided
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_seed{result.seed_used}{extension}"
        else:
            # Strip any existing extension and apply the correct one
            base = filename
            for ext in (".wav", ".mp3", ".flac", ".ogg"):
                if base.lower().endswith(ext):
                    base = base[:-len(ext)]
                    break
            filename = base + extension

        audio_path = output_dir / filename

        # Build generation config dict for sidecar
        config_dict = asdict(result.config)
        # Convert numpy latent_vector to JSON-serialisable list
        if config_dict.get("latent_vector") is not None:
            config_dict["latent_vector"] = config_dict["latent_vector"].tolist()
        # Convert SpatialConfig to dict if present
        if config_dict.get("spatial") is not None:
            spatial_dict = config_dict["spatial"]
            if hasattr(spatial_dict, "__dict__"):
                config_dict["spatial"] = vars(spatial_dict)

        # Write sidecar JSON first (research pitfall #6)
        json_path = write_sidecar_json(
            wav_path=audio_path,
            model_name=self.model_name,
            generation_config=config_dict,
            seed=result.seed_used,
            quality_metrics=result.quality,
            duration_s=result.duration_s,
            sample_rate=result.sample_rate,
            bit_depth=result.config.bit_depth,
            channels=result.channels,
        )

        # Export audio in selected format
        export_audio(
            audio=result.audio,
            path=audio_path,
            sample_rate=result.sample_rate,
            format=export_format,
            bit_depth=result.config.bit_depth,
            metadata=metadata,
        )

        return (audio_path, json_path)
