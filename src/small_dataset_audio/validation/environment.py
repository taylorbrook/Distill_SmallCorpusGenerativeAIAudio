"""Individual environment validation checks.

Each check function returns a list of error strings (empty list = OK).
These are composed by :func:`validate_environment` which separates
errors (fatal) from warnings (informational).

Heavy dependencies (torch, torchaudio) are imported inside function
bodies so that validation can report "not installed" instead of crashing
with an ImportError at module level.
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_python_version() -> list[str]:
    """Verify Python >= 3.11.

    Returns:
        List of error strings (empty if version is acceptable).
    """
    if sys.version_info < (3, 11):
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return [
            f"Python 3.11+ required (found {version}). "
            "Install with: uv python install 3.11"
        ]
    return []


def check_pytorch() -> list[str]:
    """Verify PyTorch is installed and version >= 2.10.0.

    Returns:
        List of error strings (empty if PyTorch is acceptable).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return ["PyTorch not installed. Run: uv sync"]

    from packaging.version import Version, InvalidVersion

    try:
        # torch.__version__ can contain suffixes like "2.10.0+cu128"
        # Extract just the base version
        version_str = torch.__version__.split("+")[0]
        version = Version(version_str)
    except (InvalidVersion, AttributeError):
        return [
            f"Could not parse PyTorch version '{torch.__version__}'. "
            "Run: uv sync"
        ]

    if version < Version("2.10.0"):
        return [
            f"PyTorch 2.10.0+ required (found {torch.__version__}). "
            "Run: uv sync"
        ]
    return []


def check_torchaudio() -> list[str]:
    """Verify TorchAudio is installed.

    Returns:
        List of error strings (empty if TorchAudio is available).
    """
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        return ["TorchAudio not installed. Run: uv sync"]
    return []


def check_paths(config: dict) -> list[str]:
    """Check that configured data directories exist or can be created.

    Uses :func:`~small_dataset_audio.config.settings.resolve_path` to
    turn relative paths into absolute ones.

    Returns:
        List of warning strings for missing directories.  These are
        warnings (not errors) because the application will auto-create
        them on first run.  Only returns an error if a parent directory
        does not exist *and* cannot be created.
    """
    from small_dataset_audio.config.settings import resolve_path

    warnings: list[str] = []
    paths_config = config.get("paths", {})

    for key, path_str in paths_config.items():
        resolved = resolve_path(path_str)
        if not resolved.exists():
            # Check if parent exists or can be created
            parent = resolved.parent
            if not parent.exists():
                try:
                    # Don't actually create it -- just verify we can
                    parent.mkdir(parents=True, exist_ok=True)
                    parent.rmdir()  # clean up the test
                except OSError:
                    warnings.append(
                        f"Cannot create directory for '{key}': {resolved} "
                        f"(parent {parent} is not writable)"
                    )
                    continue
            warnings.append(
                f"Directory for '{key}' does not exist yet: {resolved} "
                "(will be created on first run)"
            )
    return warnings


def validate_environment(config: dict) -> tuple[list[str], list[str]]:
    """Run all environment validation checks.

    Args:
        config: Application configuration dictionary (needs ``paths``
            section for directory checks).

    Returns:
        A ``(errors, warnings)`` tuple.  Errors are fatal (the
        application should exit).  Warnings are informational (the
        application can continue).

        * Python version and PyTorch/TorchAudio checks produce errors.
        * Missing data directories produce warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Fatal checks
    errors.extend(check_python_version())
    errors.extend(check_pytorch())
    errors.extend(check_torchaudio())

    # Non-fatal checks
    warnings.extend(check_paths(config))

    return errors, warnings
