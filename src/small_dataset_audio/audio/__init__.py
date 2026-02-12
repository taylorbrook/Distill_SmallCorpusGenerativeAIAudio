"""Audio I/O, preprocessing, and augmentation.

Public API re-exports from :mod:`audio.io`, :mod:`audio.validation`,
and :mod:`audio.thumbnails`.
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
from small_dataset_audio.audio.thumbnails import (
    generate_waveform_thumbnail,
    generate_dataset_thumbnails,
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
    # thumbnails.py
    "generate_waveform_thumbnail",
    "generate_dataset_thumbnails",
]
