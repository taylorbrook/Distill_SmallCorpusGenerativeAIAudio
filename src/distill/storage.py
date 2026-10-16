"""Shared helpers for the JSON indexes and saved-dict formats.

Used by the model library, preset manager and generation history (JSON
indexes) and by model/checkpoint loading (dataclasses rebuilt from saved
dicts).

Design notes:
- Writes go to a temp file that is fsynced and then atomically renamed,
  so a crash leaves either the old or the new index, never a truncated one.
- The ``.json.bak`` copy is refreshed only from an index that parses, so a
  corrupt index can never overwrite a good backup.
- Reads fall back to the backup when the index is unreadable, and move the
  corrupt file aside (``*.corrupt-<timestamp>``) instead of overwriting it.
- Dataclasses are rebuilt ignoring unknown keys, so files written by a
  newer version (with extra fields) still load.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _backup_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".bak")


def _read_json_dict(path: Path) -> dict:
    """Parse *path* as a JSON object; raise ``ValueError`` if it is not one."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def read_json_index(index_path: Path) -> dict:
    """Read a JSON index, recovering from its ``.bak`` copy if needed.

    Returns ``{}`` when the index does not exist.  When it exists but cannot
    be parsed, the backup is used instead (if it parses) and the corrupt
    file is moved aside so the next write does not destroy it.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        return {}

    try:
        return _read_json_dict(index_path)
    except (OSError, ValueError) as exc:
        logger.warning("Index %s is unreadable: %s", index_path, exc)

    backup_path = _backup_path(index_path)
    data: dict = {}
    try:
        data = _read_json_dict(backup_path)
        logger.warning("Recovered %s from backup %s", index_path.name, backup_path.name)
    except (OSError, ValueError) as exc:
        logger.warning("Backup %s is unusable (%s); starting empty", backup_path, exc)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    corrupt_path = index_path.with_name(f"{index_path.name}.corrupt-{stamp}")
    try:
        os.replace(index_path, corrupt_path)
        logger.warning("Moved unreadable index to %s", corrupt_path)
    except OSError:
        logger.warning(
            "Could not move unreadable index %s aside", index_path, exc_info=True
        )
    return data


def write_json_atomic(index_path: Path, data: dict) -> None:
    """Write *data* as JSON to *index_path* atomically and durably.

    1. Copy the current index to ``.json.bak`` -- only if it parses.
    2. Write to a temp file in the same directory and ``fsync`` it.
    3. ``os.replace`` the temp file over the index (atomic on POSIX/NTFS).
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        try:
            _read_json_dict(index_path)
        except (OSError, ValueError):
            logger.warning("Not backing up unreadable index %s", index_path)
        else:
            try:
                shutil.copy2(index_path, _backup_path(index_path))
            except OSError:
                logger.warning("Failed to back up %s", index_path, exc_info=True)

    fd, tmp_path = tempfile.mkstemp(dir=index_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, index_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


_safe_globals_registered = False


def _register_safe_globals() -> None:
    """Allowlist the few non-tensor types our saved files contain.

    ``.distill`` models and training checkpoints hold numpy arrays (latent
    analysis) and the ``OverfittingPreset`` enum (training config) besides
    tensors and plain containers.  Registering exactly these lets
    ``torch.load(weights_only=True)`` read every existing file while still
    refusing anything else a pickle could smuggle in.
    """
    global _safe_globals_registered
    if _safe_globals_registered:
        return

    import numpy as np  # noqa: WPS433 -- lazy import
    import torch  # noqa: WPS433 -- lazy import

    try:
        from numpy._core import multiarray  # numpy >= 2
    except ImportError:  # pragma: no cover -- numpy < 2
        from numpy.core import multiarray

    from distill.training.config import OverfittingPreset

    allowed: list = [np.ndarray, np.dtype, OverfittingPreset]
    # numpy 2.x pickles concrete dtype classes (e.g. ``Float32DType``);
    # allow the numeric and bool ones only (never object dtype).
    allowed.extend({type(np.dtype(code)) for code in "?bhilqBHILQefdgFDG"})
    # Files record numpy's module path at save time: numpy 1.x wrote
    # ``numpy.core.multiarray``, numpy 2.x writes ``numpy._core.multiarray``.
    for func in (multiarray._reconstruct, multiarray.scalar):
        for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed.append((func, f"{module}.{func.__name__}"))
    torch.serialization.add_safe_globals(allowed)
    _safe_globals_registered = True


def safe_torch_load(path: Path, map_location: Any = "cpu") -> Any:
    """``torch.load`` with ``weights_only=True`` for Distill's own file formats.

    Raises
    ------
    ValueError
        If the file contains objects outside the allowlist -- e.g. a
        tampered ``.distill`` file carrying code to execute on load.
    """
    import pickle  # noqa: WPS433 -- lazy import

    import torch  # noqa: WPS433 -- lazy import

    _register_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            f"Refusing to load {path}: it contains data other than model "
            f"weights and settings, which could run code on this machine. "
            f"Only load files from sources you trust. Details: {exc}"
        ) from exc


def dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build dataclass *cls* from *data*, ignoring keys it does not define."""
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        logger.debug("Ignoring unknown %s fields: %s", cls.__name__, unknown)
    return cls(**{k: v for k, v in data.items() if k in names})
