"""Dataset management, summary computation, and reporting.

Public API re-exports from :mod:`data.dataset` and :mod:`data.summary`.
"""

from small_dataset_audio.data.dataset import Dataset
from small_dataset_audio.data.summary import (
    DatasetSummary,
    compute_summary,
    format_summary_report,
)

__all__ = [
    "Dataset",
    "DatasetSummary",
    "compute_summary",
    "format_summary_report",
]
