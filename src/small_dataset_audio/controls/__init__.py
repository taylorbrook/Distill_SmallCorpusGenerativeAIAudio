"""Musically meaningful latent space controls.

Maps opaque latent dimensions to perceptually meaningful musical parameters
via PCA discovery and audio feature correlation, enabling users to control
generation with intuitive sliders instead of raw latent vectors.
"""

from small_dataset_audio.controls.analyzer import AnalysisResult, LatentSpaceAnalyzer
from small_dataset_audio.controls.features import FEATURE_NAMES, compute_audio_features

__all__ = [
    "AnalysisResult",
    "LatentSpaceAnalyzer",
    "compute_audio_features",
    "FEATURE_NAMES",
]
