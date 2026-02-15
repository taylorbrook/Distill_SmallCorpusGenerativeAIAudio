"""Musically meaningful latent space controls.

Maps opaque latent dimensions to perceptually meaningful musical parameters
via PCA discovery and audio feature correlation, enabling users to control
generation with intuitive sliders instead of raw latent vectors.
"""

from distill.controls.analyzer import AnalysisResult, LatentSpaceAnalyzer
from distill.controls.features import (
    FEATURE_NAMES,
    compute_audio_features,
    compute_features_batch,
)
from distill.controls.mapping import (
    SliderState,
    center_sliders,
    get_slider_info,
    is_in_warning_zone,
    randomize_sliders,
    sliders_to_latent,
)
from distill.controls.serialization import (
    analysis_from_dict,
    analysis_to_dict,
)

__all__ = [
    # analyzer.py
    "AnalysisResult",
    "LatentSpaceAnalyzer",
    # features.py
    "compute_audio_features",
    "compute_features_batch",
    "FEATURE_NAMES",
    # mapping.py
    "SliderState",
    "sliders_to_latent",
    "randomize_sliders",
    "center_sliders",
    "get_slider_info",
    "is_in_warning_zone",
    # serialization.py
    "analysis_to_dict",
    "analysis_from_dict",
]
