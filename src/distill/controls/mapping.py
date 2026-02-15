"""Slider-to-latent vector mapping for musically meaningful controls.

Converts discrete integer slider positions to continuous latent vectors
via PCA component reconstruction.  Provides randomize-all, reset-to-center,
slider metadata, and warning-zone detection.

Design notes:
- Integer step indices are the ground truth (locked decision).
- Fully independent sliders -- each maps to exactly one PCA component.
- Lazy imports for numpy (project pattern).
- ``logging.getLogger(__name__)`` for module-level logger.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slider state
# ---------------------------------------------------------------------------


@dataclass
class SliderState:
    """Current state of all latent-space sliders.

    ``positions`` contains one integer per active PCA component.
    Valid range is ``[-(n_steps // 2), n_steps // 2]`` (e.g. -10 to +10
    for ``n_steps=21``).
    """

    positions: list[int]
    """Current integer step index for each slider."""

    n_components: int
    """Number of active PCA components (matches AnalysisResult.n_active_components)."""


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def sliders_to_latent(
    slider_state: SliderState,
    analysis: "AnalysisResult",
) -> "np.ndarray":
    """Convert integer slider positions to a latent vector.

    Reconstruction formula::

        value_i = position_i * step_size[i]
        z = pca_mean + sum(value_i * pca_components[i])

    Parameters
    ----------
    slider_state : SliderState
        Current slider positions (integers).
    analysis : AnalysisResult
        PCA analysis result providing components, mean, and step sizes.

    Returns
    -------
    np.ndarray
        Latent vector of shape ``[latent_dim]`` (float32).
    """
    import numpy as np  # noqa: WPS433 -- lazy import

    z = analysis.pca_mean.copy().astype(np.float32)
    for i in range(slider_state.n_components):
        value = slider_state.positions[i] * float(analysis.step_size[i])
        z += value * analysis.pca_components[i]
    return z


# ---------------------------------------------------------------------------
# Preset operations
# ---------------------------------------------------------------------------


def randomize_sliders(
    analysis: "AnalysisResult",
    seed: int | None = None,
) -> SliderState:
    """Set all sliders to random integer positions within safe bounds.

    For each component, computes the valid step range from
    ``safe_min`` / ``safe_max`` and ``step_size``, clamps to the
    slider's hard limits, then picks a uniform random integer.

    Parameters
    ----------
    analysis : AnalysisResult
        PCA analysis result.
    seed : int | None
        Optional numpy random seed for reproducibility.

    Returns
    -------
    SliderState
        Slider state with random positions within safe bounds.
    """
    import numpy as np  # noqa: WPS433 -- lazy import

    rng = np.random.default_rng(seed)
    half = analysis.n_steps // 2  # e.g. 10 for n_steps=21

    positions: list[int] = []
    for i in range(analysis.n_active_components):
        step = float(analysis.step_size[i])
        if step > 0:
            min_step = math.ceil(float(analysis.safe_min[i]) / step)
            max_step = math.floor(float(analysis.safe_max[i]) / step)
        else:
            # Degenerate case: step_size is zero
            min_step = 0
            max_step = 0
        # Clamp to slider hard limits
        min_step = max(min_step, -half)
        max_step = min(max_step, half)
        # Ensure min <= max
        if min_step > max_step:
            min_step = max_step = 0
        positions.append(int(rng.integers(min_step, max_step + 1)))

    return SliderState(
        positions=positions,
        n_components=analysis.n_active_components,
    )


def center_sliders(analysis: "AnalysisResult") -> SliderState:
    """Reset all sliders to center (position 0 = latent space mean).

    Parameters
    ----------
    analysis : AnalysisResult
        PCA analysis result (used for component count).

    Returns
    -------
    SliderState
        Slider state with all positions set to 0.
    """
    return SliderState(
        positions=[0] * analysis.n_active_components,
        n_components=analysis.n_active_components,
    )


# ---------------------------------------------------------------------------
# Slider metadata
# ---------------------------------------------------------------------------


def get_slider_info(analysis: "AnalysisResult") -> list[dict]:
    """Return UI-ready metadata for each active slider.

    Each dict contains the component index, labels, step ranges,
    safe/warning boundaries, and variance-explained percentage.

    Parameters
    ----------
    analysis : AnalysisResult
        PCA analysis result.

    Returns
    -------
    list[dict]
        One dict per active component with keys: ``index``, ``label``,
        ``suggested_label``, ``min_step``, ``max_step``,
        ``safe_min_step``, ``safe_max_step``, ``warning_min_step``,
        ``warning_max_step``, ``variance_explained_pct``.
    """
    half = analysis.n_steps // 2
    infos: list[dict] = []

    for i in range(analysis.n_active_components):
        step = float(analysis.step_size[i])

        if step > 0:
            safe_min_step = math.ceil(float(analysis.safe_min[i]) / step)
            safe_max_step = math.floor(float(analysis.safe_max[i]) / step)
            warning_min_step = math.ceil(float(analysis.warning_min[i]) / step)
            warning_max_step = math.floor(float(analysis.warning_max[i]) / step)
        else:
            safe_min_step = 0
            safe_max_step = 0
            warning_min_step = 0
            warning_max_step = 0

        # Clamp all to hard limits
        safe_min_step = max(safe_min_step, -half)
        safe_max_step = min(safe_max_step, half)
        warning_min_step = max(warning_min_step, -half)
        warning_max_step = min(warning_max_step, half)

        infos.append({
            "index": i,
            "label": analysis.component_labels[i],
            "suggested_label": analysis.suggested_labels[i],
            "min_step": -half,
            "max_step": half,
            "safe_min_step": safe_min_step,
            "safe_max_step": safe_max_step,
            "warning_min_step": warning_min_step,
            "warning_max_step": warning_max_step,
            "variance_explained_pct": float(analysis.explained_variance_ratio[i]) * 100,
        })

    return infos


# ---------------------------------------------------------------------------
# Warning zone check
# ---------------------------------------------------------------------------


def is_in_warning_zone(position: int, slider_info: dict) -> bool:
    """Check whether a slider position falls within the soft warning zone.

    The warning zone is between the warning boundary and the safe
    boundary on either side of the slider range.

    Parameters
    ----------
    position : int
        Current integer step position.
    slider_info : dict
        Slider metadata from :func:`get_slider_info`.

    Returns
    -------
    bool
        ``True`` if position is in the warning zone (outside safe but
        inside warning), ``False`` otherwise.
    """
    # Below safe range but within warning range
    if slider_info["warning_min_step"] <= position < slider_info["safe_min_step"]:
        return True
    # Above safe range but within warning range
    if slider_info["safe_max_step"] < position <= slider_info["warning_max_step"]:
        return True
    return False
