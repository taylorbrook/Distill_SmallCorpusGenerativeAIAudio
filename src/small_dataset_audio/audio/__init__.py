"""Audio I/O, preprocessing, and augmentation.

Public API re-exports from :mod:`audio.io` and :mod:`audio.validation`.
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
]
