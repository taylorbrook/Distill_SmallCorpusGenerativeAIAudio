"""Serialization of AnalysisResult for checkpoint persistence.

Converts :class:`~small_dataset_audio.controls.analyzer.AnalysisResult`
to/from plain dicts suitable for ``torch.save`` checkpoints.

Design notes:
- Numpy arrays are kept as-is (``torch.save`` handles them natively).
- A ``version`` field enables future schema migration.
- Missing keys in older checkpoints are handled gracefully for forward
  compatibility.
- Lazy imports for numpy (project pattern).
- ``logging.getLogger(__name__)`` for module-level logger.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Current serialization format version
_CURRENT_VERSION = 1


def analysis_to_dict(analysis: "AnalysisResult") -> dict:
    """Serialize an AnalysisResult to a checkpoint-compatible dict.

    Numpy arrays are stored directly (``torch.save`` handles them).
    A ``version`` field is included for future migration support.

    Parameters
    ----------
    analysis : AnalysisResult
        The analysis result to serialize.

    Returns
    -------
    dict
        Plain dict with all AnalysisResult fields plus ``version``.
    """
    return {
        "version": _CURRENT_VERSION,
        "pca_components": analysis.pca_components,
        "pca_mean": analysis.pca_mean,
        "explained_variance_ratio": analysis.explained_variance_ratio,
        "n_active_components": analysis.n_active_components,
        "component_labels": list(analysis.component_labels),
        "suggested_labels": list(analysis.suggested_labels),
        "safe_min": analysis.safe_min,
        "safe_max": analysis.safe_max,
        "warning_min": analysis.warning_min,
        "warning_max": analysis.warning_max,
        "step_size": analysis.step_size,
        "n_steps": analysis.n_steps,
        "feature_correlations": dict(analysis.feature_correlations),
        "latent_dim": analysis.latent_dim,
    }


def analysis_from_dict(d: dict) -> "AnalysisResult":
    """Reconstruct an AnalysisResult from a checkpoint dict.

    Handles missing keys gracefully (for forward compatibility if new
    fields are added in future versions).  Raises ``ValueError`` if the
    serialization version is unsupported.

    Parameters
    ----------
    d : dict
        Dict previously produced by :func:`analysis_to_dict`.

    Returns
    -------
    AnalysisResult
        Reconstructed analysis result.

    Raises
    ------
    ValueError
        If the ``version`` field indicates an unsupported format.
    """
    import numpy as np  # noqa: WPS433 -- lazy import
    from small_dataset_audio.controls.analyzer import AnalysisResult

    version = d.get("version", 1)
    if version > _CURRENT_VERSION:
        raise ValueError(
            f"Unsupported analysis serialization version {version} "
            f"(max supported: {_CURRENT_VERSION}). "
            f"Please update small_dataset_audio."
        )

    def _to_array(value: object) -> "np.ndarray":
        """Convert lists back to numpy arrays if needed."""
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value, dtype=np.float32)

    return AnalysisResult(
        pca_components=_to_array(d["pca_components"]),
        pca_mean=_to_array(d["pca_mean"]),
        explained_variance_ratio=_to_array(d["explained_variance_ratio"]),
        n_active_components=int(d["n_active_components"]),
        component_labels=list(d.get("component_labels", [])),
        suggested_labels=list(d.get("suggested_labels", [])),
        safe_min=_to_array(d["safe_min"]),
        safe_max=_to_array(d["safe_max"]),
        warning_min=_to_array(d["warning_min"]),
        warning_max=_to_array(d["warning_max"]),
        step_size=_to_array(d["step_size"]),
        n_steps=int(d.get("n_steps", 21)),
        feature_correlations=dict(d.get("feature_correlations", {})),
        latent_dim=int(d.get("latent_dim", 64)),
    )
