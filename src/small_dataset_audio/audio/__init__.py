"""Audio I/O, preprocessing, augmentation, thumbnails, spectrogram, and filters.

Public API re-exports from :mod:`audio.io`, :mod:`audio.validation`,
:mod:`audio.augmentation`, :mod:`audio.preprocessing`, :mod:`audio.thumbnails`,
:mod:`audio.spectrogram`, and :mod:`audio.filters`.
"""

from small_dataset_audio.audio.io import (
    SUPPORTED_FORMATS,
    DEFAULT_SAMPLE_RATE,
    AudioFile,
    AudioMetadata,
    check_file_integrity,
    get_metadata,
    is_supported_format,
    load_audio,
)
from small_dataset_audio.audio.validation import (
    Severity,
    ValidationIssue,
    collect_audio_files,
    format_validation_report,
    validate_dataset,
)
from small_dataset_audio.audio.augmentation import (
    AugmentationConfig,
    AugmentationPipeline,
)
from small_dataset_audio.audio.preprocessing import (
    PreprocessingConfig,
    preprocess_for_training,
    preprocess_dataset,
    load_cached_dataset,
    clear_cache,
)
from small_dataset_audio.audio.thumbnails import (
    generate_waveform_thumbnail,
    generate_dataset_thumbnails,
)
from small_dataset_audio.audio.spectrogram import (
    SpectrogramConfig,
    AudioSpectrogram,
)
from small_dataset_audio.audio.filters import (
    apply_anti_alias_filter,
)
from small_dataset_audio.audio.hrtf import (
    HRTFData,
    load_hrtf,
    apply_binaural,
    clear_hrtf_cache,
)
from small_dataset_audio.audio.metadata import (
    embed_metadata,
    build_export_metadata,
)

__all__ = [
    # io.py
    "SUPPORTED_FORMATS",
    "DEFAULT_SAMPLE_RATE",
    "AudioFile",
    "AudioMetadata",
    "check_file_integrity",
    "get_metadata",
    "is_supported_format",
    "load_audio",
    # validation.py
    "Severity",
    "ValidationIssue",
    "collect_audio_files",
    "format_validation_report",
    "validate_dataset",
    # augmentation.py
    "AugmentationConfig",
    "AugmentationPipeline",
    # preprocessing.py
    "PreprocessingConfig",
    "preprocess_for_training",
    "preprocess_dataset",
    "load_cached_dataset",
    "clear_cache",
    # thumbnails.py
    "generate_waveform_thumbnail",
    "generate_dataset_thumbnails",
    # spectrogram.py
    "SpectrogramConfig",
    "AudioSpectrogram",
    # filters.py
    "apply_anti_alias_filter",
    # hrtf.py
    "HRTFData",
    "load_hrtf",
    "apply_binaural",
    "clear_hrtf_cache",
    # metadata.py
    "embed_metadata",
    "build_export_metadata",
]
