"""Model library catalog with JSON index management.

Provides :class:`ModelEntry` (dataclass for per-model metadata) and
:class:`ModelLibrary` (JSON-indexed catalog with search, filter, and
atomic writes).

The JSON index (``model_library.json``) stores lightweight metadata
for every saved ``.sda`` model so that catalog browsing, searching,
and filtering never require loading heavy model files via ``torch.load``.

Design notes:
- ``from __future__ import annotations`` for modern type hints.
- ``logging.getLogger(__name__)`` for module-level logger.
- Lazy ``torch`` import only inside :meth:`ModelLibrary.repair_index`.
- ``dataclasses.asdict`` for ModelEntry -> dict serialization in JSON.
- Atomic writes via temp file + ``os.replace`` (POSIX-atomic).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Index file name within models directory
_INDEX_FILENAME = "model_library.json"

# Current index format version
_INDEX_VERSION = 1


# ---------------------------------------------------------------------------
# ModelEntry
# ---------------------------------------------------------------------------


@dataclass
class ModelEntry:
    """Lightweight metadata for a single saved model in the library.

    Fields mirror the JSON index structure for direct serialization
    via ``dataclasses.asdict``.
    """

    model_id: str
    name: str
    description: str
    file_path: str  # relative filename within models_dir, NOT absolute
    file_size_bytes: int
    dataset_name: str
    dataset_file_count: int
    dataset_total_duration_s: float
    training_date: str  # ISO 8601
    save_date: str  # ISO 8601
    training_epochs: int
    final_train_loss: float
    final_val_loss: float
    has_analysis: bool
    n_active_components: int
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Atomic index write
# ---------------------------------------------------------------------------


def _write_index_atomic(index_path: Path, data: dict) -> None:
    """Write JSON index atomically to prevent corruption.

    Strategy:
    1. Backup existing index to ``.json.bak`` (best effort).
    2. Write to a temp file in the same directory (same filesystem).
    3. ``os.replace(tmp_path, index_path)`` for atomic swap (POSIX).
    4. Clean up temp file on failure.

    Parameters
    ----------
    index_path : Path
        Target path for the JSON index file.
    data : dict
        Full index dict to serialize.
    """
    # Backup existing index (best effort)
    if index_path.exists():
        backup_path = index_path.with_suffix(".json.bak")
        try:
            import shutil

            shutil.copy2(index_path, backup_path)
        except OSError:
            pass

    # Ensure parent directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=index_path.parent,
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, index_path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# ModelLibrary
# ---------------------------------------------------------------------------


class ModelLibrary:
    """Manage the model library catalog backed by a JSON index file.

    On construction, loads (or creates) the index from
    ``models_dir / model_library.json``.  All mutations
    (``add_entry``, ``remove``) persist immediately via atomic writes.

    Parameters
    ----------
    models_dir : Path
        Directory containing ``.sda`` model files and the JSON index.
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)
        self._index_path = self.models_dir / _INDEX_FILENAME
        self._entries: dict[str, ModelEntry] = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Read JSON index, deserialize to dict[str, ModelEntry].

        Handles missing or corrupt file gracefully (logs warning,
        starts fresh).
        """
        if not self._index_path.exists():
            self._entries = {}
            return

        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            models_raw = raw.get("models", {})
            entries: dict[str, ModelEntry] = {}
            for model_id, entry_dict in models_raw.items():
                try:
                    entries[model_id] = ModelEntry(**entry_dict)
                except (TypeError, KeyError) as exc:
                    logger.warning(
                        "Skipping corrupt entry %s in index: %s",
                        model_id,
                        exc,
                    )
            self._entries = entries
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning(
                "Failed to load model library index (%s), starting fresh: %s",
                self._index_path,
                exc,
            )
            self._entries = {}

    def _save_index(self) -> None:
        """Persist current entries to the JSON index file atomically."""
        data = {
            "version": _INDEX_VERSION,
            "models": {
                model_id: asdict(entry)
                for model_id, entry in self._entries.items()
            },
        }
        _write_index_atomic(self._index_path, data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, entry: ModelEntry) -> None:
        """Add a model entry to the catalog and persist.

        Parameters
        ----------
        entry : ModelEntry
            The entry to add.  ``entry.model_id`` is the key.
        """
        self._entries[entry.model_id] = entry
        self._save_index()

    def remove(self, model_id: str) -> bool:
        """Remove a model entry from the catalog and persist.

        Parameters
        ----------
        model_id : str
            ID of the model to remove.

        Returns
        -------
        bool
            ``True`` if the entry existed and was removed,
            ``False`` if not found.
        """
        if model_id not in self._entries:
            return False
        del self._entries[model_id]
        self._save_index()
        return True

    def get(self, model_id: str) -> ModelEntry | None:
        """Look up a model entry by ID.

        Parameters
        ----------
        model_id : str
            The model ID to look up.

        Returns
        -------
        ModelEntry | None
            The entry, or ``None`` if not found.
        """
        return self._entries.get(model_id)

    def list_all(
        self,
        sort_by: str = "save_date",
        reverse: bool = True,
    ) -> list[ModelEntry]:
        """Return all entries, sorted.

        Parameters
        ----------
        sort_by : str
            Field name to sort by (default ``"save_date"``).
        reverse : bool
            Descending order if ``True`` (default).

        Returns
        -------
        list[ModelEntry]
            All catalog entries, sorted.
        """
        results = list(self._entries.values())
        results.sort(
            key=lambda e: getattr(e, sort_by, ""),
            reverse=reverse,
        )
        return results

    def search(
        self,
        query: str = "",
        tags: list[str] | None = None,
        sort_by: str = "save_date",
        reverse: bool = True,
    ) -> list[ModelEntry]:
        """Search and filter model entries.

        Case-insensitive substring match on name and description.
        Tag filter uses any-match (entry matches if it has any
        of the requested tags).

        Parameters
        ----------
        query : str
            Search string matched against name and description.
        tags : list[str] | None
            Tags to filter by (any-match).
        sort_by : str
            Field name to sort by (default ``"save_date"``).
        reverse : bool
            Descending order if ``True`` (default).

        Returns
        -------
        list[ModelEntry]
            Matching entries, sorted.
        """
        results = list(self._entries.values())

        if query:
            q = query.lower()
            results = [
                e
                for e in results
                if q in e.name.lower() or q in e.description.lower()
            ]

        if tags:
            tag_set = {t.lower() for t in tags}
            results = [
                e
                for e in results
                if tag_set & {t.lower() for t in e.tags}
            ]

        results.sort(
            key=lambda e: getattr(e, sort_by, ""),
            reverse=reverse,
        )
        return results

    def count(self) -> int:
        """Return the number of entries in the catalog.

        Returns
        -------
        int
            Number of model entries.
        """
        return len(self._entries)

    def repair_index(self) -> tuple[int, int]:
        """Check consistency between index and files on disk.

        1. Remove entries whose ``.sda`` file is missing.
        2. Scan for orphan ``.sda`` files not in the index and add
           them by reading metadata via ``torch.load``.

        Returns
        -------
        tuple[int, int]
            ``(removed_count, added_count)`` -- number of stale
            entries removed and orphan files added.
        """
        import torch  # noqa: WPS433 -- lazy import

        removed_count = 0
        added_count = 0

        # 1. Remove entries whose .sda file is missing
        stale_ids: list[str] = []
        for model_id, entry in self._entries.items():
            model_path = self.models_dir / entry.file_path
            if not model_path.exists():
                logger.warning(
                    "Index entry %s (%s) has no file on disk -- removing",
                    model_id,
                    entry.file_path,
                )
                stale_ids.append(model_id)

        for model_id in stale_ids:
            del self._entries[model_id]
            removed_count += 1

        # 2. Scan for orphan .sda files not in the index
        indexed_files = {e.file_path for e in self._entries.values()}
        for sda_path in self.models_dir.glob("*.sda"):
            rel_name = sda_path.name
            if rel_name in indexed_files:
                continue

            logger.warning(
                "Orphan .sda file not in index: %s -- adding",
                rel_name,
            )
            try:
                saved = torch.load(sda_path, map_location="cpu", weights_only=False)
                meta = saved.get("metadata", {})
                entry = ModelEntry(
                    model_id=meta.get("model_id", str(sda_path.stem)),
                    name=meta.get("name", sda_path.stem),
                    description=meta.get("description", ""),
                    file_path=rel_name,
                    file_size_bytes=sda_path.stat().st_size,
                    dataset_name=meta.get("dataset_name", ""),
                    dataset_file_count=meta.get("dataset_file_count", 0),
                    dataset_total_duration_s=meta.get("dataset_total_duration_s", 0.0),
                    training_date=meta.get("training_date", ""),
                    save_date=meta.get("save_date", ""),
                    training_epochs=meta.get("training_epochs", 0),
                    final_train_loss=meta.get("final_train_loss", 0.0),
                    final_val_loss=meta.get("final_val_loss", 0.0),
                    has_analysis=meta.get("has_analysis", False),
                    n_active_components=meta.get("n_active_components", 0),
                    tags=meta.get("tags", []),
                )
                self._entries[entry.model_id] = entry
                added_count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to read orphan .sda file %s: %s",
                    rel_name,
                    exc,
                )

        # Persist if anything changed
        if removed_count > 0 or added_count > 0:
            self._save_index()

        return (removed_count, added_count)
