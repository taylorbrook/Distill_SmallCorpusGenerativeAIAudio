"""Dataset summary computation and reporting.

Computes aggregate statistics from a :class:`Dataset` instance:
file counts, durations, sample rate distribution, format breakdown,
channel distribution, and optional thumbnail generation.

Design notes:
- Uses plain text formatting (no rich dependency -- that's a UI concern
  for Phase 8).
- Thumbnail generation is optional (default on) and delegates to
  audio.thumbnails.
- Duration formatting uses HH:MM:SS for human readability.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from distill.audio.validation import ValidationIssue


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetSummary:
    """Aggregate statistics for a dataset."""

    name: str
    file_count: int
    valid_file_count: int
    total_duration_seconds: float
    sample_rates: dict[int, int]              # rate -> count of files
    dominant_sample_rate: int
    sample_rate_consistent: bool
    formats: dict[str, int]                   # format -> count
    channel_counts: dict[int, int]            # channels -> count
    min_duration_seconds: float
    max_duration_seconds: float
    avg_duration_seconds: float
    thumbnail_paths: dict[Path, Path] = field(default_factory=dict)
    validation_issues: list[ValidationIssue] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_summary(
    dataset: "Dataset",  # noqa: F821 -- avoid circular import
    generate_thumbnails: bool = True,
) -> DatasetSummary:
    """Compute summary statistics from a Dataset instance.

    Args:
        dataset: The :class:`~distill.data.dataset.Dataset`
            to summarize.
        generate_thumbnails: If ``True``, generate waveform thumbnails
            for all valid files via :func:`generate_dataset_thumbnails`.

    Returns:
        A :class:`DatasetSummary` with all computed statistics.
    """
    valid_files = dataset.valid_files

    # -- Sample rate distribution --
    rate_counter: Counter[int] = Counter()
    for f in valid_files:
        meta = dataset.metadata.get(f)
        if meta is not None:
            rate_counter[meta.sample_rate] += 1

    if rate_counter:
        dominant_rate, _ = rate_counter.most_common(1)[0]
        sample_rate_consistent = len(rate_counter) == 1
    else:
        dominant_rate = 0
        sample_rate_consistent = True

    # -- Format distribution --
    format_counter: Counter[str] = Counter()
    for f in valid_files:
        meta = dataset.metadata.get(f)
        if meta is not None:
            format_counter[meta.format] += 1

    # -- Channel distribution --
    channel_counter: Counter[int] = Counter()
    for f in valid_files:
        meta = dataset.metadata.get(f)
        if meta is not None:
            channel_counter[meta.num_channels] += 1

    # -- Duration stats --
    durations = []
    for f in valid_files:
        meta = dataset.metadata.get(f)
        if meta is not None:
            durations.append(meta.duration_seconds)

    total_duration = sum(durations) if durations else 0.0
    min_duration = min(durations) if durations else 0.0
    max_duration = max(durations) if durations else 0.0
    avg_duration = (total_duration / len(durations)) if durations else 0.0

    # -- Thumbnails --
    thumbnail_paths: dict[Path, Path] = {}
    if generate_thumbnails and valid_files:
        from distill.audio.thumbnails import (  # noqa: WPS433
            generate_dataset_thumbnails,
        )

        pairs = [
            (f, dataset.metadata[f])
            for f in valid_files
            if f in dataset.metadata
        ]
        if pairs:
            thumbnail_paths = generate_dataset_thumbnails(
                pairs,
                thumbnail_dir=dataset.thumbnail_dir,
            )

    return DatasetSummary(
        name=dataset.name,
        file_count=len(dataset.files),
        valid_file_count=len(valid_files),
        total_duration_seconds=total_duration,
        sample_rates=dict(rate_counter),
        dominant_sample_rate=dominant_rate,
        sample_rate_consistent=sample_rate_consistent,
        formats=dict(format_counter),
        channel_counts=dict(channel_counter),
        min_duration_seconds=min_duration,
        max_duration_seconds=max_duration,
        avg_duration_seconds=avg_duration,
        thumbnail_paths=thumbnail_paths,
        validation_issues=list(dataset.validation_issues),
    )


def format_summary_report(summary: DatasetSummary) -> str:
    """Format a DatasetSummary as a human-readable text report.

    Args:
        summary: The summary to format.

    Returns:
        Plain text report string.
    """
    lines: list[str] = []

    # Header
    lines.append(f"Dataset: {summary.name}")
    lines.append("=" * (len(lines[0])))
    lines.append("")

    # File counts
    lines.append(f"Files: {summary.valid_file_count} valid / {summary.file_count} total")
    lines.append("")

    # Duration
    total_fmt = _format_duration(summary.total_duration_seconds)
    lines.append(f"Total duration: {total_fmt}")
    if summary.valid_file_count > 0:
        lines.append(
            f"  Min: {summary.min_duration_seconds:.2f}s  "
            f"Max: {summary.max_duration_seconds:.2f}s  "
            f"Avg: {summary.avg_duration_seconds:.2f}s"
        )
    lines.append("")

    # Sample rates
    if summary.sample_rate_consistent:
        lines.append(
            f"Sample rate: {summary.dominant_sample_rate} Hz (consistent)"
        )
    else:
        lines.append("Sample rate: INCONSISTENT")
        for rate, count in sorted(summary.sample_rates.items()):
            marker = " *" if rate == summary.dominant_sample_rate else ""
            lines.append(f"  {rate} Hz: {count} file(s){marker}")
    lines.append("")

    # Formats
    if summary.formats:
        lines.append("Formats:")
        for fmt, count in sorted(summary.formats.items()):
            lines.append(f"  {fmt}: {count} file(s)")
        lines.append("")

    # Channels
    if summary.channel_counts:
        lines.append("Channels:")
        for ch, count in sorted(summary.channel_counts.items()):
            label = "mono" if ch == 1 else "stereo" if ch == 2 else f"{ch}ch"
            lines.append(f"  {label}: {count} file(s)")
        lines.append("")

    # Validation issues
    errors = [
        i for i in summary.validation_issues
        if i.severity.value == "error"
    ]
    warnings = [
        i for i in summary.validation_issues
        if i.severity.value == "warning"
    ]
    if errors or warnings:
        lines.append(f"Validation: {len(errors)} error(s), {len(warnings)} warning(s)")
    else:
        lines.append("Validation: passed")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    if seconds <= 0:
        return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
