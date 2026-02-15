"""Dataset management class wrapping a collection of audio files.

Provides import workflows for audio files -- from a directory (DATA-02)
or from an explicit file list (DATA-01) -- with validation on import so
the user sees issues immediately.

Design notes:
- Does NOT load waveforms into memory.  Only metadata is stored.
  Waveform loading is deferred to preprocessing/training.
- Validation runs on every import (from_directory, from_files, add_files)
  using the error-collection pattern from audio.validation.
- Imports from audio.io and audio.validation -- never duplicates I/O
  or validation logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from distill.audio.io import AudioMetadata, get_metadata
from distill.audio.validation import (
    Severity,
    ValidationIssue,
    collect_audio_files,
    validate_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Dataset:
    """A collection of audio files with metadata and validation status.

    This is a data-pipeline wrapper, NOT a PyTorch Dataset.  Phase 3
    will create a PyTorch Dataset that wraps this class for training.
    """

    def __init__(self, name: str, base_dir: Path) -> None:
        """Initialize an empty dataset.

        Args:
            name: Human-readable dataset name.
            base_dir: Root directory for this dataset.
        """
        self.name = name
        self.base_dir = Path(base_dir)
        self.files: list[Path] = []
        self.metadata: dict[Path, AudioMetadata] = {}
        self.validation_issues: list[ValidationIssue] = []
        self.thumbnail_dir: Path = self.base_dir / ".thumbnails"

    # ------------------------------------------------------------------
    # Class constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        name: str | None = None,
    ) -> Dataset:
        """Import all supported audio files from a directory (DATA-02).

        Recursively scans *directory* for audio files using
        :func:`~distill.audio.validation.collect_audio_files`,
        then validates the collection.

        Args:
            directory: Root directory to scan.
            name: Dataset name (defaults to directory name).

        Returns:
            A new :class:`Dataset` populated with found files.
        """
        directory = Path(directory)
        ds_name = name if name is not None else directory.name
        ds = cls(name=ds_name, base_dir=directory)

        found_files = collect_audio_files(directory)
        if found_files:
            ds._import_files(found_files)

        return ds

    @classmethod
    def from_files(
        cls,
        files: list[Path],
        name: str,
        base_dir: Path | None = None,
    ) -> Dataset:
        """Import specific audio files (DATA-01).

        Args:
            files: List of audio file paths.
            name: Dataset name.
            base_dir: Root directory (defaults to common parent of all files).

        Returns:
            A new :class:`Dataset` populated with the given files.
        """
        resolved = [Path(f).resolve() for f in files]

        if base_dir is None:
            if resolved:
                # Common parent of all files
                parents = [f.parent for f in resolved]
                # Find the longest common prefix
                base_dir = parents[0]
                for p in parents[1:]:
                    # Walk up until we find a common ancestor
                    while base_dir not in (p, *p.parents):
                        base_dir = base_dir.parent
            else:
                base_dir = Path.cwd()

        ds = cls(name=name, base_dir=base_dir)
        if resolved:
            ds._import_files(resolved)

        return ds

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def add_files(self, files: list[Path]) -> list[ValidationIssue]:
        """Add more files to this dataset.

        Validates the new files, loads their metadata, and appends to
        the dataset.  Returns validation issues for the new files only.

        Args:
            files: Paths to audio files to add.

        Returns:
            Validation issues for the newly added files.
        """
        resolved = [Path(f).resolve() for f in files]
        new_issues = validate_dataset(resolved, min_file_count=0)
        self.validation_issues.extend(new_issues)

        # Determine which files have errors
        error_files = {
            issue.file_path
            for issue in new_issues
            if issue.severity == Severity.ERROR and issue.file_path is not None
        }

        for f in resolved:
            if f not in error_files:
                self.files.append(f)
                try:
                    self.metadata[f] = get_metadata(f)
                except Exception as exc:
                    logger.warning("Could not read metadata for %s: %s", f.name, exc)

        return new_issues

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def valid_files(self) -> list[Path]:
        """Files that passed validation (no ERROR-severity issues)."""
        error_files = {
            issue.file_path
            for issue in self.validation_issues
            if issue.severity == Severity.ERROR and issue.file_path is not None
        }
        return [f for f in self.files if f not in error_files]

    @property
    def file_count(self) -> int:
        """Number of valid files in the dataset."""
        return len(self.valid_files)

    @property
    def total_duration(self) -> float:
        """Sum of duration_seconds for all valid files."""
        total = 0.0
        for f in self.valid_files:
            meta = self.metadata.get(f)
            if meta is not None:
                total += meta.duration_seconds
        return total

    @property
    def sample_rates(self) -> set[int]:
        """Set of unique sample rates across valid files."""
        rates: set[int] = set()
        for f in self.valid_files:
            meta = self.metadata.get(f)
            if meta is not None:
                rates.add(meta.sample_rate)
        return rates

    def has_errors(self) -> bool:
        """True if any validation issue has ERROR severity."""
        return any(
            issue.severity == Severity.ERROR
            for issue in self.validation_issues
        )

    def has_warnings(self) -> bool:
        """True if any validation issue has WARNING severity."""
        return any(
            issue.severity == Severity.WARNING
            for issue in self.validation_issues
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _import_files(self, files: list[Path]) -> None:
        """Validate and import files, loading metadata for valid ones."""
        issues = validate_dataset(files)
        self.validation_issues.extend(issues)

        # Determine which files had per-file errors
        error_files = {
            issue.file_path
            for issue in issues
            if issue.severity == Severity.ERROR and issue.file_path is not None
        }

        for f in files:
            if f not in error_files:
                self.files.append(f)
                try:
                    self.metadata[f] = get_metadata(f)
                except Exception as exc:
                    logger.warning(
                        "Could not read metadata for %s: %s", f.name, exc
                    )

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, "
            f"files={self.file_count}, "
            f"duration={self.total_duration:.1f}s)"
        )
