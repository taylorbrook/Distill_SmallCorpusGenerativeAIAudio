"""Multi-model blending engine for loading up to 4 models simultaneously and blending their outputs via latent-space or audio-domain averaging.

Supports two blending strategies:
- **Latent-space blending**: weighted average of latent vectors before decoding
  (requires matching latent_dim across all models).
- **Audio-domain blending**: weighted average of generated waveforms (works with
  any model architecture combination).

Design notes:
- Lazy imports for numpy, torch (project pattern).
- TYPE_CHECKING imports for type hints only.
- Union slider resolution merges PCA components across models by index order.
- Non-shared parameters zero-filled (neutral/mean position) per discretion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

    from distill.controls.analyzer import AnalysisResult
    from distill.audio.spectrogram import AudioSpectrogram
    from distill.models.persistence import ModelMetadata
    from distill.models.vae import ConvVAE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BLEND_MODELS = 4
"""Maximum number of models that can be loaded simultaneously for blending."""

# ---------------------------------------------------------------------------
# Blend mode
# ---------------------------------------------------------------------------


class BlendMode(str, Enum):
    """Blending strategy for multi-model generation."""

    LATENT = "latent"
    """Weighted average of latent vectors before decoding."""

    AUDIO = "audio"
    """Weighted average of audio waveforms after generation."""


# ---------------------------------------------------------------------------
# Model slot
# ---------------------------------------------------------------------------


@dataclass
class ModelSlot:
    """A loaded model participating in multi-model blending.

    Each slot holds a fully loaded model with its spectrogram converter,
    optional PCA analysis, metadata, and a user-assigned blend weight.
    """

    model: "ConvVAE"
    """Trained VAE model."""

    spectrogram: "AudioSpectrogram"
    """Spectrogram converter for mel-to-waveform."""

    analysis: "AnalysisResult | None"
    """PCA analysis result (``None`` if model has no analysis)."""

    metadata: "ModelMetadata"
    """User-facing model metadata."""

    device: "torch.device"
    """Device the model resides on."""

    weight: float = 25.0
    """Raw blend weight 0-100 (not normalised). Default 25% (equal share of 4)."""

    active: bool = True
    """Whether this slot participates in blending."""


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------


def normalize_weights(raw_weights: list[float]) -> list[float]:
    """Normalise raw blend weights (0-100 scale) to sum to 1.0.

    If all weights are zero, distributes equally (1/n each).

    Parameters
    ----------
    raw_weights : list[float]
        Raw weight values (any non-negative scale).

    Returns
    -------
    list[float]
        Normalised weights summing to 1.0.
    """
    total = sum(raw_weights)
    if total == 0:
        n = len(raw_weights)
        return [1.0 / n] * n if n > 0 else []
    return [w / total for w in raw_weights]


# ---------------------------------------------------------------------------
# Union slider resolution
# ---------------------------------------------------------------------------


@dataclass
class UnionSliderInfo:
    """Metadata for a single slider in the union slider set.

    When multiple models are loaded, the union slider set is the
    superset of all active PCA components across all models.
    """

    index: int
    """Global slider index in the union set."""

    label: str
    """Suggested label (from the first model that has this component)."""

    variance_pct: float
    """Maximum variance explained percentage from any model for this component."""

    model_indices: list[int]
    """Which model slot indices have this parameter."""

    component_indices: list[int]
    """Corresponding PCA component index in each model (parallel to model_indices)."""


def resolve_union_sliders(slots: list["ModelSlot"]) -> list[UnionSliderInfo]:
    """Build a unified slider set across all active model slots.

    Merges PCA components by index order (component 0 from model A aligns
    with component 0 from model B).  Models without a given component
    index are excluded from that slider's model_indices (and will receive
    zero-fill during generation).

    Parameters
    ----------
    slots : list[ModelSlot]
        Active model slots to merge.

    Returns
    -------
    list[UnionSliderInfo]
        Union slider metadata, length = max(n_active_components) across
        all models.
    """
    from distill.controls.mapping import get_slider_info

    # Collect per-model slider info
    per_model_infos: list[list[dict]] = []
    for slot in slots:
        if slot.analysis is not None:
            per_model_infos.append(get_slider_info(slot.analysis))
        else:
            per_model_infos.append([])

    # Determine max component count across all models
    max_components = max(
        (len(infos) for infos in per_model_infos),
        default=0,
    )

    union_sliders: list[UnionSliderInfo] = []
    for comp_idx in range(max_components):
        label = f"Axis {comp_idx + 1}"
        max_variance = 0.0
        model_indices: list[int] = []
        component_indices: list[int] = []

        for model_idx, infos in enumerate(per_model_infos):
            if comp_idx < len(infos):
                info = infos[comp_idx]
                # Use first model's label as the union label
                if not model_indices:
                    label = info.get("suggested_label") or info.get("label", label)
                variance = info.get("variance_explained_pct", 0.0)
                if variance > max_variance:
                    max_variance = variance
                model_indices.append(model_idx)
                component_indices.append(comp_idx)

        union_sliders.append(UnionSliderInfo(
            index=comp_idx,
            label=label,
            variance_pct=max_variance,
            model_indices=model_indices,
            component_indices=component_indices,
        ))

    return union_sliders


# ---------------------------------------------------------------------------
# Latent-space blending
# ---------------------------------------------------------------------------


def blend_latent_space(
    slots: list["ModelSlot"],
    latent_vectors: list["np.ndarray"],
    weights: list[float],
) -> "np.ndarray":
    """Blend multiple latent vectors via weighted average.

    All models must share the same ``latent_dim``.

    Parameters
    ----------
    slots : list[ModelSlot]
        Model slots (used to validate latent_dim consistency).
    latent_vectors : list[np.ndarray]
        Per-model latent vectors, each shape ``[latent_dim]``.
    weights : list[float]
        Raw blend weights (will be normalised internally).

    Returns
    -------
    np.ndarray
        Blended latent vector of shape ``[latent_dim]`` (float32).

    Raises
    ------
    ValueError
        If models have mismatched latent_dim.
    """
    import numpy as np  # noqa: WPS433 -- lazy import

    # Validate all models share latent_dim
    dims = [slot.model.latent_dim for slot in slots]
    if len(set(dims)) > 1:
        raise ValueError(
            f"Latent-space blending requires matching latent_dim across "
            f"all models, but got dimensions: {dims}. "
            f"Use audio-domain blending (BlendMode.AUDIO) instead."
        )

    norm_weights = normalize_weights(weights)
    blended = np.zeros_like(latent_vectors[0], dtype=np.float32)
    for w, z in zip(norm_weights, latent_vectors):
        blended += w * z.astype(np.float32)
    return blended


# ---------------------------------------------------------------------------
# Audio-domain blending
# ---------------------------------------------------------------------------


def blend_audio_domain(
    audio_outputs: list["np.ndarray"],
    weights: list[float],
) -> "np.ndarray":
    """Blend multiple audio waveforms via weighted average.

    Works with any model architecture combination.  Shorter outputs
    are zero-padded to match the longest.

    Parameters
    ----------
    audio_outputs : list[np.ndarray]
        Per-model audio arrays.  Each is either 1-D ``[samples]`` (mono)
        or 2-D ``[2, samples]`` (stereo).
    weights : list[float]
        Raw blend weights (will be normalised internally).

    Returns
    -------
    np.ndarray
        Blended audio as float32, same shape convention as inputs.
    """
    import numpy as np  # noqa: WPS433 -- lazy import

    if not audio_outputs:
        return np.array([], dtype=np.float32)

    norm_weights = normalize_weights(weights)

    # Determine max length and dimensionality
    ndim = audio_outputs[0].ndim
    if ndim == 1:
        max_len = max(a.shape[0] for a in audio_outputs)
        blended = np.zeros(max_len, dtype=np.float32)
        for w, audio in zip(norm_weights, audio_outputs):
            padded = np.zeros(max_len, dtype=np.float32)
            padded[:audio.shape[0]] = audio.astype(np.float32)
            blended += w * padded
    else:
        # Stereo: [2, samples]
        channels = audio_outputs[0].shape[0]
        max_len = max(a.shape[1] for a in audio_outputs)
        blended = np.zeros((channels, max_len), dtype=np.float32)
        for w, audio in zip(norm_weights, audio_outputs):
            padded = np.zeros((channels, max_len), dtype=np.float32)
            padded[:, :audio.shape[1]] = audio.astype(np.float32)
            blended += w * padded

    return blended


# ---------------------------------------------------------------------------
# Blend engine
# ---------------------------------------------------------------------------


class BlendEngine:
    """Multi-model blending engine.

    Manages up to :data:`MAX_BLEND_MODELS` model slots with
    add/remove/set_weight operations and generates blended audio via
    either latent-space or audio-domain averaging.
    """

    def __init__(self) -> None:
        self.slots: list[ModelSlot] = []
        self.blend_mode: BlendMode = BlendMode.LATENT

    def add_model(
        self,
        model: "ConvVAE",
        spectrogram: "AudioSpectrogram",
        analysis: "AnalysisResult | None",
        metadata: "ModelMetadata",
        device: "torch.device",
        weight: float = 25.0,
    ) -> int:
        """Load a model into the next available slot.

        Parameters
        ----------
        model : ConvVAE
            Trained VAE model.
        spectrogram : AudioSpectrogram
            Spectrogram converter.
        analysis : AnalysisResult | None
            PCA analysis result (or ``None``).
        metadata : ModelMetadata
            Model metadata.
        device : torch.device
            Device the model resides on.
        weight : float
            Initial blend weight (0-100).

        Returns
        -------
        int
            Index of the new slot.

        Raises
        ------
        ValueError
            If already at maximum model capacity.
        """
        if len(self.slots) >= MAX_BLEND_MODELS:
            raise ValueError(
                f"Cannot add model: already at maximum capacity "
                f"({MAX_BLEND_MODELS} models)."
            )
        slot = ModelSlot(
            model=model,
            spectrogram=spectrogram,
            analysis=analysis,
            metadata=metadata,
            device=device,
            weight=weight,
            active=True,
        )
        self.slots.append(slot)
        logger.info(
            "Added model '%s' to blend slot %d (weight=%.1f)",
            metadata.name,
            len(self.slots) - 1,
            weight,
        )
        return len(self.slots) - 1

    def remove_model(self, index: int) -> None:
        """Remove a model from a slot and free GPU memory.

        The model is moved to CPU before removal to release GPU memory
        (research pitfall #6: memory pressure with multiple models).

        Parameters
        ----------
        index : int
            Slot index to remove.

        Raises
        ------
        IndexError
            If index is out of range.
        """
        import torch  # noqa: WPS433 -- lazy import

        if index < 0 or index >= len(self.slots):
            raise IndexError(f"Slot index {index} out of range (0-{len(self.slots) - 1}).")
        slot = self.slots[index]
        # Move model to CPU to free GPU memory
        slot.model.to(torch.device("cpu"))
        logger.info(
            "Removed model '%s' from blend slot %d (moved to CPU)",
            slot.metadata.name,
            index,
        )
        self.slots.pop(index)

    def set_weight(self, index: int, weight: float) -> None:
        """Set raw blend weight (0-100) for a slot.

        Parameters
        ----------
        index : int
            Slot index.
        weight : float
            New weight value (0-100).

        Raises
        ------
        IndexError
            If index is out of range.
        """
        if index < 0 or index >= len(self.slots):
            raise IndexError(f"Slot index {index} out of range (0-{len(self.slots) - 1}).")
        self.slots[index].weight = weight

    def set_blend_mode(self, mode: BlendMode) -> None:
        """Set the blending strategy.

        When switching to latent-space mode, validates that all active
        models share the same ``latent_dim``.

        Parameters
        ----------
        mode : BlendMode
            Desired blending mode.

        Raises
        ------
        ValueError
            If ``LATENT`` mode is requested but models have mismatched
            latent dimensions.
        """
        if mode is BlendMode.LATENT:
            active = self.get_active_slots()
            if len(active) > 1:
                dims = [s.model.latent_dim for s in active]
                if len(set(dims)) > 1:
                    raise ValueError(
                        f"Cannot use latent-space blending: active models "
                        f"have mismatched latent_dim {dims}. "
                        f"Use BlendMode.AUDIO (audio-domain blending) instead."
                    )
        self.blend_mode = mode

    def get_active_slots(self) -> list[ModelSlot]:
        """Return slots that are active and have non-zero weight.

        Returns
        -------
        list[ModelSlot]
            Active slots participating in blending.
        """
        return [s for s in self.slots if s.active and s.weight > 0]

    def get_union_sliders(self) -> list[UnionSliderInfo]:
        """Get the unified slider set across all active models.

        Returns
        -------
        list[UnionSliderInfo]
            Union slider metadata.
        """
        return resolve_union_sliders(self.get_active_slots())

    def blend_generate(
        self,
        slider_positions: list[int],
        config: "GenerationConfig",
    ) -> "GenerationResult":
        """Generate blended audio from multiple models.

        Uses the current :attr:`blend_mode` to combine outputs from all
        active model slots.  Union slider positions are mapped to each
        model's PCA components (zero-filled for missing components).

        Parameters
        ----------
        slider_positions : list[int]
            Integer positions for each union slider.
        config : GenerationConfig
            Generation configuration (duration, stereo mode, etc.).

        Returns
        -------
        GenerationResult
            Blended generation result.

        Raises
        ------
        ValueError
            If no active slots are available.
        """
        import numpy as np  # noqa: WPS433 -- lazy import

        from distill.controls.mapping import SliderState, sliders_to_latent
        from distill.inference.generation import (
            GenerationConfig,
            GenerationPipeline,
            GenerationResult,
        )

        active_slots = self.get_active_slots()
        if not active_slots:
            raise ValueError("No active model slots for blending.")

        # Single model: no blending needed, just generate normally
        if len(active_slots) == 1:
            slot = active_slots[0]
            latent_vector = self._build_latent_for_slot(
                slot, slider_positions, 0, self.get_union_sliders(),
            )
            gen_config = GenerationConfig(
                duration_s=config.duration_s,
                max_duration_s=config.max_duration_s,
                seed=config.seed,
                chunk_duration_s=config.chunk_duration_s,
                concat_mode=config.concat_mode,
                stereo_mode=config.stereo_mode,
                stereo_width=config.stereo_width,
                sample_rate=config.sample_rate,
                bit_depth=config.bit_depth,
                steps_between=config.steps_between,
                overlap_samples=config.overlap_samples,
                latent_vector=latent_vector,
            )
            from distill.vocoder import get_vocoder

            vocoder = get_vocoder("bigvgan", device=str(slot.device))
            pipeline = GenerationPipeline(
                model=slot.model,
                spectrogram=slot.spectrogram,
                device=slot.device,
                vocoder=vocoder,
            )
            pipeline.model_name = slot.metadata.name
            return pipeline.generate(gen_config)

        # Multiple models: blend according to mode
        raw_weights = [s.weight for s in active_slots]
        union_sliders = self.get_union_sliders()

        if self.blend_mode is BlendMode.LATENT:
            return self._blend_latent(
                active_slots, slider_positions, union_sliders,
                raw_weights, config,
            )
        else:
            return self._blend_audio(
                active_slots, slider_positions, union_sliders,
                raw_weights, config,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_latent_for_slot(
        self,
        slot: ModelSlot,
        slider_positions: list[int],
        slot_index_in_active: int,
        union_sliders: list[UnionSliderInfo],
    ) -> "np.ndarray":
        """Build a latent vector for a single slot from union slider positions.

        Maps union slider positions to this model's PCA components.
        Components that this model does not have are zero-filled
        (neutral/mean position).

        Parameters
        ----------
        slot : ModelSlot
            The model slot.
        slider_positions : list[int]
            Union slider positions (one per union slider).
        slot_index_in_active : int
            Index of this slot in the active slots list.
        union_sliders : list[UnionSliderInfo]
            Union slider metadata.

        Returns
        -------
        np.ndarray
            Latent vector of shape ``[latent_dim]``.
        """
        import numpy as np  # noqa: WPS433 -- lazy import
        from distill.controls.mapping import SliderState, sliders_to_latent

        if slot.analysis is None:
            # No analysis: return mean latent vector (zeros)
            return np.zeros(slot.model.latent_dim, dtype=np.float32)

        # Build per-model slider positions from union positions
        n_components = slot.analysis.n_active_components
        model_positions = [0] * n_components

        for u_slider in union_sliders:
            if slot_index_in_active in u_slider.model_indices:
                # This model has this component
                active_idx = u_slider.model_indices.index(slot_index_in_active)
                comp_idx = u_slider.component_indices[active_idx]
                if comp_idx < n_components and u_slider.index < len(slider_positions):
                    model_positions[comp_idx] = slider_positions[u_slider.index]
            # else: zero-fill (already 0)

        state = SliderState(
            positions=model_positions,
            n_components=n_components,
        )
        return sliders_to_latent(state, slot.analysis)

    def _blend_latent(
        self,
        active_slots: list[ModelSlot],
        slider_positions: list[int],
        union_sliders: list[UnionSliderInfo],
        raw_weights: list[float],
        config: "GenerationConfig",
    ) -> "GenerationResult":
        """Latent-space blending: average latent vectors, decode once."""
        from distill.inference.generation import (
            GenerationConfig,
            GenerationPipeline,
        )

        # Build per-model latent vectors
        latent_vectors = []
        for i, slot in enumerate(active_slots):
            z = self._build_latent_for_slot(slot, slider_positions, i, union_sliders)
            latent_vectors.append(z)

        # Blend latent vectors
        blended_z = blend_latent_space(active_slots, latent_vectors, raw_weights)

        # Use first model's pipeline to decode
        first = active_slots[0]
        gen_config = GenerationConfig(
            duration_s=config.duration_s,
            max_duration_s=config.max_duration_s,
            seed=config.seed,
            chunk_duration_s=config.chunk_duration_s,
            concat_mode=config.concat_mode,
            stereo_mode=config.stereo_mode,
            stereo_width=config.stereo_width,
            sample_rate=config.sample_rate,
            bit_depth=config.bit_depth,
            steps_between=config.steps_between,
            overlap_samples=config.overlap_samples,
            latent_vector=blended_z,
        )
        from distill.vocoder import get_vocoder

        vocoder = get_vocoder("bigvgan", device=str(first.device))
        pipeline = GenerationPipeline(
            model=first.model,
            spectrogram=first.spectrogram,
            device=first.device,
            vocoder=vocoder,
        )
        model_names = [s.metadata.name for s in active_slots]
        pipeline.model_name = " + ".join(model_names)
        return pipeline.generate(gen_config)

    def _blend_audio(
        self,
        active_slots: list[ModelSlot],
        slider_positions: list[int],
        union_sliders: list[UnionSliderInfo],
        raw_weights: list[float],
        config: "GenerationConfig",
    ) -> "GenerationResult":
        """Audio-domain blending: generate per-model, average waveforms."""
        import numpy as np  # noqa: WPS433 -- lazy import

        from distill.inference.generation import (
            GenerationConfig,
            GenerationPipeline,
            GenerationResult,
        )
        from distill.inference.quality import compute_quality_score

        from distill.vocoder import get_vocoder

        audio_outputs = []
        seed_used = None

        # Create vocoder once for all slots (typically same device)
        vocoder = get_vocoder("bigvgan", device=str(active_slots[0].device))

        for i, slot in enumerate(active_slots):
            z = self._build_latent_for_slot(slot, slider_positions, i, union_sliders)
            gen_config = GenerationConfig(
                duration_s=config.duration_s,
                max_duration_s=config.max_duration_s,
                seed=config.seed,
                chunk_duration_s=config.chunk_duration_s,
                concat_mode=config.concat_mode,
                stereo_mode=config.stereo_mode,
                stereo_width=config.stereo_width,
                sample_rate=config.sample_rate,
                bit_depth=config.bit_depth,
                steps_between=config.steps_between,
                overlap_samples=config.overlap_samples,
                latent_vector=z,
            )
            pipeline = GenerationPipeline(
                model=slot.model,
                spectrogram=slot.spectrogram,
                device=slot.device,
                vocoder=vocoder,
            )
            pipeline.model_name = slot.metadata.name
            result = pipeline.generate(gen_config)
            audio_outputs.append(result.audio)
            if seed_used is None:
                seed_used = result.seed_used

        # Blend audio waveforms
        blended_audio = blend_audio_domain(audio_outputs, raw_weights)

        # Compute quality on blended result
        quality = compute_quality_score(blended_audio, config.sample_rate)

        model_names = [s.metadata.name for s in active_slots]
        channels = 1 if blended_audio.ndim == 1 else blended_audio.shape[0]
        actual_duration = blended_audio.shape[-1] / config.sample_rate

        return GenerationResult(
            audio=blended_audio,
            sample_rate=config.sample_rate,
            quality=quality,
            config=config,
            seed_used=seed_used or 0,
            duration_s=actual_duration,
            channels=channels,
        )
