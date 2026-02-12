"""Configuration loading, saving, and path resolution.

This module handles TOML-based configuration using stdlib tomllib (read)
and tomli-w (write). It intentionally does NOT import torch or any heavy
dependencies so that configuration errors can be reported even when
PyTorch is broken or missing.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from small_dataset_audio.config.defaults import DEFAULT_CONFIG


def get_config_path() -> Path:
    """Return the path to config.toml in the project root.

    The project root is found by walking up from the package location
    to find the nearest directory containing pyproject.toml. Falls back
    to the current working directory if no pyproject.toml is found.

    Returns:
        Path to config.toml (may not exist yet).
    """
    # Start from this file's location and walk up
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Safety limit to avoid infinite loop
        if (current / "pyproject.toml").exists():
            return current / "config.toml"
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fallback: current working directory
    return Path.cwd() / "config.toml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict.

    Values in override take precedence. Missing keys in override are
    filled from base. Nested dicts are merged recursively; non-dict
    values are replaced entirely.

    Args:
        base: The base dictionary (typically DEFAULT_CONFIG).
        override: The override dictionary (typically loaded from file).

    Returns:
        A new merged dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from a TOML file.

    If the file does not exist, returns a deep copy of DEFAULT_CONFIG.
    If the file exists, reads it and merges with defaults so that any
    new config keys added in future versions are present.

    Args:
        config_path: Path to the config file. Defaults to get_config_path().

    Returns:
        Configuration dictionary with all keys populated.
    """
    if config_path is None:
        config_path = get_config_path()

    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    with open(config_path, "rb") as f:
        user_config = tomllib.load(f)

    return _deep_merge(DEFAULT_CONFIG, user_config)


def save_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    """Save configuration dictionary to a TOML file.

    Creates parent directories if they do not exist.

    Args:
        config: Configuration dictionary to save.
        config_path: Path to the config file. Defaults to get_config_path().
    """
    if config_path is None:
        config_path = get_config_path()

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)


def resolve_path(path_str: str, base_dir: Path | None = None) -> Path:
    """Resolve a configuration path string to an absolute Path.

    Handles three cases:
    - Paths starting with ~ are expanded to the user's home directory.
    - Absolute paths are returned as-is.
    - Relative paths are resolved against base_dir (defaults to project root).

    Args:
        path_str: A path string from configuration.
        base_dir: Base directory for relative paths. Defaults to project root
                  (the directory containing pyproject.toml).

    Returns:
        An absolute, resolved Path.
    """
    path = Path(path_str).expanduser()

    if path.is_absolute():
        return path.resolve()

    if base_dir is None:
        # Derive project root from config path location
        config_path = get_config_path()
        base_dir = config_path.parent

    return (base_dir / path).resolve()
