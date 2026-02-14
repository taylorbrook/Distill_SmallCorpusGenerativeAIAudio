"""Preset management with model-scoped CRUD and virtual folder organization.

Provides :class:`PresetEntry` (dataclass for a single preset) and
:class:`PresetManager` (JSON-indexed manager with preset CRUD and
virtual folder management).

Each model gets its own ``presets.json`` index file inside a
``{model_id}/`` directory under the presets root.  Presets store
slider positions + optional seed (no duration -- locked decision).

Design notes:
- ``from __future__ import annotations`` for modern type hints.
- ``logging.getLogger(__name__)`` for module-level logger.
- ``dataclasses.asdict`` for PresetEntry -> dict serialization in JSON.
- Atomic writes via temp file + ``os.replace`` (POSIX-atomic).
- Seed ``None`` maps to JSON ``null``; integer maps to integer.
- Virtual folders: string field in JSON, not filesystem directories.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Current index format version
_INDEX_VERSION = 1


# ---------------------------------------------------------------------------
# PresetEntry
# ---------------------------------------------------------------------------


@dataclass
class PresetEntry:
    """A named slider configuration preset for a specific model.

    Stores integer slider positions (matching ``SliderState.positions``)
    plus an optional seed.  Duration is NOT stored (locked decision:
    duration is set per-generation).

    Fields:
    - ``preset_id``: UUID string identifying this preset.
    - ``name``: User-visible display name.
    - ``folder``: Virtual folder name (empty string = root / no folder).
    - ``slider_positions``: Integer step indices for each PCA component.
    - ``n_components``: Number of active PCA components.
    - ``seed``: Optional seed (``None`` = random seed each time).
    - ``created``: ISO 8601 creation timestamp.
    - ``modified``: ISO 8601 last-modified timestamp.
    - ``description``: Optional user description.
    """

    preset_id: str
    name: str
    folder: str = ""
    slider_positions: list[int] = field(default_factory=list)
    n_components: int = 0
    seed: int | None = None
    created: str = ""
    modified: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Atomic index write (copied pattern from library/catalog.py)
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
# PresetManager
# ---------------------------------------------------------------------------


class PresetManager:
    """Manage presets for a specific model.

    Each model has its own presets directory and JSON index file at
    ``presets_dir / model_id / presets.json``.

    Preset CRUD: save, load, get, list, rename, update, delete, count.
    Folder management: create, rename, delete, list (virtual folders).

    Parameters
    ----------
    presets_dir : Path
        Root directory for all preset storage.
    model_id : str
        Model ID this manager is scoped to.
    """

    def __init__(self, presets_dir: Path, model_id: str) -> None:
        self.presets_dir = Path(presets_dir)
        self.model_id = model_id
        self._model_dir = self.presets_dir / model_id
        self._index_path = self._model_dir / "presets.json"
        self._entries: dict[str, PresetEntry] = {}
        self._folders: list[dict] = []  # [{"name": ..., "created": ...}]
        self._load_index()

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Read JSON index, deserialize to entries and folders.

        Handles missing or corrupt file gracefully (logs warning,
        starts fresh).
        """
        if not self._index_path.exists():
            self._entries = {}
            self._folders = []
            return

        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            # Deserialize folders
            self._folders = raw.get("folders", [])
            # Deserialize preset entries
            presets_raw = raw.get("presets", {})
            entries: dict[str, PresetEntry] = {}
            for preset_id, entry_dict in presets_raw.items():
                try:
                    # Explicit seed handling: JSON null -> Python None
                    seed_val = entry_dict.get("seed")
                    if seed_val is not None:
                        seed_val = int(seed_val)
                    entry_dict["seed"] = seed_val
                    entries[preset_id] = PresetEntry(**entry_dict)
                except (TypeError, KeyError) as exc:
                    logger.warning(
                        "Skipping corrupt preset entry %s in index: %s",
                        preset_id,
                        exc,
                    )
            self._entries = entries
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning(
                "Failed to load preset index (%s), starting fresh: %s",
                self._index_path,
                exc,
            )
            self._entries = {}
            self._folders = []

    def _save_index(self) -> None:
        """Persist current entries and folders to the JSON index file atomically."""
        data = {
            "version": _INDEX_VERSION,
            "model_id": self.model_id,
            "folders": self._folders,
            "presets": {
                preset_id: asdict(entry)
                for preset_id, entry in self._entries.items()
            },
        }
        _write_index_atomic(self._index_path, data)

    # ------------------------------------------------------------------
    # Preset CRUD
    # ------------------------------------------------------------------

    def save_preset(
        self,
        name: str,
        slider_positions: list[int],
        n_components: int,
        seed: int | None = None,
        folder: str = "",
        description: str = "",
    ) -> PresetEntry:
        """Save a new preset.

        Parameters
        ----------
        name : str
            Display name for the preset.
        slider_positions : list[int]
            Integer step positions from ``SliderState.positions``.
        n_components : int
            Number of active PCA components.
        seed : int | None
            Optional seed (``None`` = random each time).
        folder : str
            Folder name (empty string = root).
        description : str
            Optional description.

        Returns
        -------
        PresetEntry
            The created preset entry.
        """
        now = datetime.now(timezone.utc).isoformat()
        entry = PresetEntry(
            preset_id=str(uuid.uuid4()),
            name=name,
            folder=folder,
            slider_positions=list(slider_positions),
            n_components=n_components,
            seed=seed,
            created=now,
            modified=now,
            description=description,
        )
        self._entries[entry.preset_id] = entry
        self._save_index()
        return entry

    def load_preset(
        self,
        preset_id: str,
    ) -> tuple:
        """Load a preset and return ``(SliderState, seed)``.

        Parameters
        ----------
        preset_id : str
            The preset ID to load.

        Returns
        -------
        tuple[SliderState, int | None]
            The slider state and optional seed.

        Raises
        ------
        KeyError
            If ``preset_id`` not found.
        """
        from small_dataset_audio.controls.mapping import SliderState

        entry = self._entries.get(preset_id)
        if entry is None:
            raise KeyError(f"Preset not found: {preset_id}")

        slider_state = SliderState(
            positions=list(entry.slider_positions),
            n_components=entry.n_components,
        )
        return slider_state, entry.seed

    def get_preset(self, preset_id: str) -> PresetEntry | None:
        """Look up a preset entry by ID.

        Parameters
        ----------
        preset_id : str
            The preset ID to look up.

        Returns
        -------
        PresetEntry | None
            The entry, or ``None`` if not found.
        """
        return self._entries.get(preset_id)

    def list_presets(self, folder: str | None = None) -> list[PresetEntry]:
        """Return all presets, optionally filtered by folder.

        Sorted by modified timestamp descending (most recent first).

        Parameters
        ----------
        folder : str | None
            If provided, only return presets in this folder.
            If ``None``, return all presets.

        Returns
        -------
        list[PresetEntry]
            Matching presets sorted by modified descending.
        """
        results = list(self._entries.values())
        if folder is not None:
            results = [e for e in results if e.folder == folder]
        results.sort(key=lambda e: e.modified, reverse=True)
        return results

    def rename_preset(self, preset_id: str, new_name: str) -> PresetEntry:
        """Rename a preset.

        Parameters
        ----------
        preset_id : str
            The preset to rename.
        new_name : str
            New display name.

        Returns
        -------
        PresetEntry
            The updated preset entry.

        Raises
        ------
        KeyError
            If ``preset_id`` not found.
        """
        entry = self._entries.get(preset_id)
        if entry is None:
            raise KeyError(f"Preset not found: {preset_id}")

        entry.name = new_name
        entry.modified = datetime.now(timezone.utc).isoformat()
        self._save_index()
        return entry

    def update_preset(
        self,
        preset_id: str,
        slider_positions: list[int],
        n_components: int,
        seed: int | None = None,
    ) -> PresetEntry:
        """Update a preset's slider data (overwrite with new values).

        Parameters
        ----------
        preset_id : str
            The preset to update.
        slider_positions : list[int]
            New slider positions.
        n_components : int
            New component count.
        seed : int | None
            New seed value.

        Returns
        -------
        PresetEntry
            The updated preset entry.

        Raises
        ------
        KeyError
            If ``preset_id`` not found.
        """
        entry = self._entries.get(preset_id)
        if entry is None:
            raise KeyError(f"Preset not found: {preset_id}")

        entry.slider_positions = list(slider_positions)
        entry.n_components = n_components
        entry.seed = seed
        entry.modified = datetime.now(timezone.utc).isoformat()
        self._save_index()
        return entry

    def delete_preset(self, preset_id: str) -> bool:
        """Delete a preset.

        Parameters
        ----------
        preset_id : str
            The preset to delete.

        Returns
        -------
        bool
            ``True`` if deleted, ``False`` if not found.
        """
        if preset_id not in self._entries:
            return False
        del self._entries[preset_id]
        self._save_index()
        return True

    def count(self) -> int:
        """Return the number of presets.

        Returns
        -------
        int
            Number of preset entries.
        """
        return len(self._entries)

    # ------------------------------------------------------------------
    # Folder management (virtual folders -- string field, not filesystem)
    # ------------------------------------------------------------------

    def create_folder(self, name: str) -> None:
        """Create a new virtual folder.

        Parameters
        ----------
        name : str
            Folder name.

        Raises
        ------
        ValueError
            If name is empty or already exists (case-insensitive).
        """
        name = name.strip()
        if not name:
            raise ValueError("Folder name cannot be empty")

        # Case-insensitive duplicate check
        existing_lower = {f["name"].lower() for f in self._folders}
        if name.lower() in existing_lower:
            raise ValueError(f"Folder already exists: {name}")

        now = datetime.now(timezone.utc).isoformat()
        self._folders.append({"name": name, "created": now})
        self._save_index()

    def rename_folder(self, old_name: str, new_name: str) -> int:
        """Rename a virtual folder.

        Bulk-updates all presets with ``folder == old_name`` to ``new_name``.
        Updates the folders list entry.

        Parameters
        ----------
        old_name : str
            Current folder name.
        new_name : str
            New folder name.

        Returns
        -------
        int
            Number of presets updated.

        Raises
        ------
        ValueError
            If ``old_name`` not found or ``new_name`` already exists.
        """
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Folder name cannot be empty")

        # Find old folder
        old_idx = None
        for i, f in enumerate(self._folders):
            if f["name"] == old_name:
                old_idx = i
                break
        if old_idx is None:
            raise ValueError(f"Folder not found: {old_name}")

        # Check new name does not exist (case-insensitive)
        existing_lower = {
            f["name"].lower() for i, f in enumerate(self._folders) if i != old_idx
        }
        if new_name.lower() in existing_lower:
            raise ValueError(f"Folder already exists: {new_name}")

        # Update folder name in list
        self._folders[old_idx]["name"] = new_name

        # Bulk-update presets in old folder
        updated = 0
        for entry in self._entries.values():
            if entry.folder == old_name:
                entry.folder = new_name
                updated += 1

        self._save_index()
        return updated

    def delete_folder(self, name: str, move_presets_to: str = "") -> int:
        """Delete a virtual folder.

        Moves its presets to ``move_presets_to`` (default root ``""``).
        Removes the folder from the folders list.

        Parameters
        ----------
        name : str
            Folder to delete.
        move_presets_to : str
            Target folder for displaced presets (default: root).

        Returns
        -------
        int
            Number of presets moved.

        Raises
        ------
        ValueError
            If ``name`` not found.
        """
        # Find folder
        folder_idx = None
        for i, f in enumerate(self._folders):
            if f["name"] == name:
                folder_idx = i
                break
        if folder_idx is None:
            raise ValueError(f"Folder not found: {name}")

        # Move presets to target folder
        moved = 0
        for entry in self._entries.values():
            if entry.folder == name:
                entry.folder = move_presets_to
                moved += 1

        # Remove folder from list
        del self._folders[folder_idx]

        self._save_index()
        return moved

    def list_folders(self) -> list[str]:
        """Return sorted list of folder names.

        Returns
        -------
        list[str]
            Folder names, sorted alphabetically.
        """
        return sorted(f["name"] for f in self._folders)
