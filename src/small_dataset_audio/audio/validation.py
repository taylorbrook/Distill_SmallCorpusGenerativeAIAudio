"""Dataset integrity validation with error-collection pattern.

Validates a collection of audio files for training readiness: format
support, file integrity, sample-rate consistency, minimum file count,
and minimum duration.

Design notes:
- Never raises exceptions during validation -- collects all issues and
  returns them (matching Phase 1 pattern from validation/environment.py).
- Each per-file check is wrapped in try/except so one bad file cannot
  stop validation of the rest.
- Imports from audio.io for get_metadata, check_file_integrity,
  is_supported_format, SUPPORTED_FORMATS.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from small_dataset_audio.audio.io import (
    SUPPORTED_FORMATS,
    check_file_integrity,
    get_metadata,
    is_supported_format,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class Severity(Enum):
    """Issue severity level, matching Phase 1's error/warning distinction."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """A single validation finding for a dataset or file."""

    severity: Severity
    file_path: Path | None
    message: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_dataset(
    files: list[Path],
    min_file_count: int = 5,
) -> list[ValidationIssue]:
    """Validate a collection of audio files for training readiness.

    Checks (in order):
    1. Minimum file count.
    2. Format support (extension in SUPPORTED_FORMATS).
    3. File integrity (readable header + data).
    4. Sample-rate consistency across valid files.
    5. Very short files (< 0.1 s).

    Args:
        files: Paths to audio files to validate.
        min_file_count: Minimum number of files required (default 5).

    Returns:
        A list of :class:`ValidationIssue` objects.  An empty list means
        the dataset passed all checks.
    """
    issues: list[ValidationIssue] = []

    # ------------------------------------------------------------------
    # 1. Minimum file count
    # ------------------------------------------------------------------
    if len(files) < min_file_count:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                file_path=None,
                message=(
                    f"Dataset has {len(files)} file(s), "
                    f"minimum required is {min_file_count}"
                ),
            )
        )
    if len(files) == 0:
        return issues  # nothing more to check

    # ------------------------------------------------------------------
    # 2. Format support
    # ------------------------------------------------------------------
    supported_files: list[Path] = []
    for f in files:
        try:
            if not is_supported_format(f):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        file_path=f,
                        message=(
                            f"Unsupported format '{f.suffix}'. "
                            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
                        ),
                    )
                )
            else:
                supported_files.append(f)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    file_path=f,
                    message=f"Error checking format: {exc}",
                )
            )

    # ------------------------------------------------------------------
    # 3. File integrity
    # ------------------------------------------------------------------
    valid_files: list[Path] = []
    for f in supported_files:
        try:
            ok, msg = check_file_integrity(f)
            if not ok:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        file_path=f,
                        message=msg,
                    )
                )
            else:
                valid_files.append(f)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    file_path=f,
                    message=f"Integrity check failed: {exc}",
                )
            )

    if not valid_files:
        return issues  # no valid files to inspect further

    # ------------------------------------------------------------------
    # 4. Sample-rate consistency
    # ------------------------------------------------------------------
    sample_rates: dict[Path, int] = {}
    for f in valid_files:
        try:
            meta = get_metadata(f)
            sample_rates[f] = meta.sample_rate
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    file_path=f,
                    message=f"Could not read metadata: {exc}",
                )
            )

    if sample_rates:
        rate_counts = Counter(sample_rates.values())
        majority_rate, _ = rate_counts.most_common(1)[0]
        for f, rate in sample_rates.items():
            if rate != majority_rate:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        file_path=f,
                        message=(
                            f"Sample rate {rate} Hz differs from majority "
                            f"rate {majority_rate} Hz"
                        ),
                    )
                )

    # ------------------------------------------------------------------
    # 5. Very short files (< 0.1 s)
    # ------------------------------------------------------------------
    for f in valid_files:
        try:
            meta = get_metadata(f)
            if meta.duration_seconds < 0.1:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        file_path=f,
                        message=(
                            f"Very short audio ({meta.duration_seconds:.3f}s), "
                            "may not be useful for training"
                        ),
                    )
                )
        except Exception:
            pass  # already reported in step 4

    return issues


def collect_audio_files(directory: Path) -> list[Path]:
    """Recursively find audio files with supported extensions.

    Skips hidden files and directories (names starting with ``'.'``).
    Returns a sorted list of absolute paths.

    Args:
        directory: Root directory to scan.

    Returns:
        Sorted list of :class:`~pathlib.Path` objects.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    found: list[Path] = []
    for p in directory.rglob("*"):
        # Skip hidden files/directories
        if any(part.startswith(".") for part in p.parts if part != "."):
            continue
        if p.is_file() and is_supported_format(p):
            found.append(p.resolve())

    return sorted(found)


def format_validation_report(issues: list[ValidationIssue]) -> str:
    """Format validation issues as a human-readable report.

    Groups errors first, then warnings, with a header line summarising
    the counts.

    Args:
        issues: List of :class:`ValidationIssue` objects.

    Returns:
        Formatted report string.
    """
    if not issues:
        return "Validation passed: no issues found."

    errors = [i for i in issues if i.severity == Severity.ERROR]
    warnings = [i for i in issues if i.severity == Severity.WARNING]

    lines: list[str] = [
        f"Validation complete: {len(errors)} error(s), {len(warnings)} warning(s)",
        "",
    ]

    if errors:
        lines.append("ERRORS:")
        for issue in errors:
            prefix = f"  [{issue.file_path.name}] " if issue.file_path else "  "
            lines.append(f"{prefix}{issue.message}")
        lines.append("")

    if warnings:
        lines.append("WARNINGS:")
        for issue in warnings:
            prefix = f"  [{issue.file_path.name}] " if issue.file_path else "  "
            lines.append(f"{prefix}{issue.message}")
        lines.append("")

    return "\n".join(lines)
