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

    from small_dataset_audio.controls.analyzer import AnalysisResult
    from small_dataset_audio.audio.spectrogram import AudioSpectrogram
    from small_dataset_audio.models.persistence import ModelMetadata
    from small_dataset_audio.models.vae import ConvVAE

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
    from small_dataset_audio.controls.mapping import get_slider_info

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
