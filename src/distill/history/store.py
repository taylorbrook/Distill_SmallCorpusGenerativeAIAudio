"""Generation history storage with WAV files, waveform thumbnails, and parameter snapshots.

Provides :class:`HistoryEntry` (dataclass for a single generation record) and
:class:`GenerationHistory` (JSON-indexed CRUD with WAV + thumbnail storage).

Each history entry captures:
- The generated audio as a WAV file on disk
- A waveform thumbnail PNG for quick visual browsing
- A full parameter snapshot (slider positions, seed, model, duration, etc.)

The JSON index (``history.json``) stores lightweight metadata for every entry
so that history browsing never requires loading heavy WAV files.

Design notes:
- ``from __future__ import annotations`` for modern type hints.
- ``logging.getLogger(__name__)`` for module-level logger.
- Lazy imports for ``export_wav`` and ``generate_waveform_thumbnail`` inside
  method bodies (heavy dependencies -- project pattern).
- ``dataclasses.asdict`` for HistoryEntry -> dict serialization in JSON.
- Atomic writes via temp file + ``os.replace`` (POSIX-atomic), same
  pattern as ``library/catalog.py``.
- Slider positions ``None`` handling: JSON ``null`` round-trips correctly.
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
# HistoryEntry
# ---------------------------------------------------------------------------


@dataclass
class HistoryEntry:
    """A single generation in the history log.

    Fields capture the full parameter snapshot plus references to the
    WAV audio file and waveform thumbnail PNG on disk.
    """

    entry_id: str
    timestamp: str  # ISO 8601

    # Model reference
    model_id: str
    model_name: str

    # Generation parameters (full snapshot)
    slider_positions: list[int] | None  # None if generated without sliders
    n_components: int
    seed: int
    duration_s: float
    sample_rate: int
    stereo_mode: str
    preset_name: str  # "custom" if no preset was used

    # File references (relative to history_dir)
    audio_file: str  # e.g., "audio/{entry_id}.wav"
    thumbnail_file: str  # e.g., "thumbnails/{entry_id}.png"

    # Quality metadata
    quality_score: dict = field(default_factory=dict)

    # Latent vector stored as list for JSON serialization
    latent_vector: list[float] | None = None


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
# GenerationHistory
# ---------------------------------------------------------------------------


class GenerationHistory:
    """Manage generation history backed by a JSON index file.

    On construction, loads (or creates) the index from
    ``history_dir / history.json``.  All mutations persist immediately
    via atomic writes.

    Parameters
    ----------
    history_dir : Path
        Directory for history storage (index, audio files, thumbnails).
    """

    def __init__(self, history_dir: Path) -> None:
        self.history_dir = Path(history_dir)
        self._index_path = self.history_dir / "history.json"
        self._entries: dict[str, HistoryEntry] = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Read JSON index, deserialize to dict[str, HistoryEntry].

        Handles missing or corrupt file gracefully (logs warning,
        starts fresh).
        """
        if not self._index_path.exists():
            self._entries = {}
            return

        try:
            raw = json.loads(self._index_path.read_text(encoding="utf-8"))
            entries_raw = raw.get("entries", {})
            entries: dict[str, HistoryEntry] = {}
            for entry_id, entry_dict in entries_raw.items():
                try:
                    entries[entry_id] = HistoryEntry(**entry_dict)
                except (TypeError, KeyError) as exc:
                    logger.warning(
                        "Skipping corrupt history entry %s: %s",
                        entry_id,
                        exc,
                    )
            self._entries = entries
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning(
                "Failed to load history index (%s), starting fresh: %s",
                self._index_path,
                exc,
            )
            self._entries = {}

    def _save_index(self) -> None:
        """Persist current entries to the JSON index file atomically."""
        data = {
            "version": _INDEX_VERSION,
            "entries": {
                entry_id: asdict(entry)
                for entry_id, entry in self._entries.items()
            },
        }
        _write_index_atomic(self._index_path, data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_to_history(
        self,
        result: "GenerationResult",  # noqa: F821
        model_id: str,
        model_name: str,
        slider_positions: list[int] | None,
        n_components: int,
        preset_name: str,
    ) -> HistoryEntry:
        """Record a generation in the history.

        Order of operations is CRITICAL for consistency (Research Pitfall #7):
        1. Save WAV file to ``history/audio/{entry_id}.wav``
        2. Generate waveform thumbnail to ``history/thumbnails/{entry_id}.png``
        3. Create HistoryEntry with all fields
        4. Add to index (atomic JSON write)

        If WAV write fails, no index entry is created.

        Parameters
        ----------
        result : GenerationResult
            The generation result containing audio data and metadata.
        model_id : str
            ID of the model that generated the audio.
        model_name : str
            Display name of the model.
        slider_positions : list[int] | None
            Integer slider positions, or ``None`` if generated without sliders.
        n_components : int
            Number of active PCA components.
        preset_name : str
            Name of the preset used (``"custom"`` if none).

        Returns
        -------
        HistoryEntry
            The newly created history entry.
        """
        # Lazy imports (heavy dependencies -- project pattern)
        from distill.audio.thumbnails import generate_waveform_thumbnail
        from distill.inference.export import export_wav

        entry_id = str(uuid.uuid4())

        # 1. Save WAV file (files first, then index)
        audio_dir = self.history_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_filename = f"{entry_id}.wav"
        audio_path = audio_dir / audio_filename
        export_wav(
            audio=result.audio,
            path=audio_path,
            sample_rate=result.sample_rate,
            bit_depth=result.config.bit_depth,
        )

        # 2. Generate waveform thumbnail (smaller than dataset thumbnails)
        thumb_dir = self.history_dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_filename = f"{entry_id}.png"
        thumb_path = thumb_dir / thumb_filename
        generate_waveform_thumbnail(
            waveform=result.audio,
            output_path=thumb_path,
            width=400,
            height=60,
        )

        # 3. Create HistoryEntry with full parameter snapshot
        now = datetime.now(timezone.utc).isoformat()
        latent_list = None
        if result.config.latent_vector is not None:
            latent_list = result.config.latent_vector.tolist()

        entry = HistoryEntry(
            entry_id=entry_id,
            timestamp=now,
            model_id=model_id,
            model_name=model_name,
            slider_positions=list(slider_positions) if slider_positions else None,
            n_components=n_components,
            seed=result.seed_used,
            duration_s=result.duration_s,
            sample_rate=result.sample_rate,
            stereo_mode=result.config.stereo_mode,
            preset_name=preset_name,
            audio_file=f"audio/{audio_filename}",
            thumbnail_file=f"thumbnails/{thumb_filename}",
            quality_score=result.quality,
            latent_vector=latent_list,
        )

        # 4. Add to index and persist atomically
        self._entries[entry_id] = entry
        self._save_index()

        logger.info(
            "Added history entry %s for model %s (%.1fs, %s)",
            entry_id[:8],
            model_name,
            result.duration_s,
            preset_name,
        )

        return entry

    def list_entries(
        self,
        model_id: str | None = None,
        limit: int | None = None,
    ) -> list[HistoryEntry]:
        """Return entries sorted by timestamp descending (most recent first).

        Parameters
        ----------
        model_id : str | None
            If provided, filter to entries for this model only.
        limit : int | None
            If provided, return at most this many entries.

        Returns
        -------
        list[HistoryEntry]
            History entries in reverse-chronological order.
        """
        results = list(self._entries.values())

        if model_id is not None:
            results = [e for e in results if e.model_id == model_id]

        # Reverse-chronological (locked decision)
        results.sort(key=lambda e: e.timestamp, reverse=True)

        if limit is not None:
            results = results[:limit]

        return results

    def get(self, entry_id: str) -> HistoryEntry | None:
        """Look up a history entry by ID.

        Parameters
        ----------
        entry_id : str
            The entry ID to look up.

        Returns
        -------
        HistoryEntry | None
            The entry, or ``None`` if not found.
        """
        return self._entries.get(entry_id)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a history entry and its associated files.

        Deletes: WAV file, thumbnail PNG, then index entry.
        Order: files first, then index (same principle as WAV-before-index).

        Parameters
        ----------
        entry_id : str
            The entry ID to delete.

        Returns
        -------
        bool
            ``True`` if deleted, ``False`` if not found.
        """
        entry = self._entries.get(entry_id)
        if entry is None:
            return False

        # Delete audio file (file may already be missing)
        audio_path = self.history_dir / entry.audio_file
        try:
            audio_path.unlink()
        except OSError:
            pass

        # Delete thumbnail (file may already be missing)
        thumb_path = self.history_dir / entry.thumbnail_file
        try:
            thumb_path.unlink()
        except OSError:
            pass

        # Remove from index and persist
        del self._entries[entry_id]
        self._save_index()

        logger.info("Deleted history entry %s", entry_id[:8])

        return True

    def get_total_size(self) -> int:
        """Compute cumulative disk usage of all audio and thumbnail files.

        Skips missing files gracefully.

        Returns
        -------
        int
            Total bytes used by history files on disk.
        """
        total = 0
        for entry in self._entries.values():
            audio_path = self.history_dir / entry.audio_file
            try:
                total += audio_path.stat().st_size
            except OSError:
                pass

            thumb_path = self.history_dir / entry.thumbnail_file
            try:
                total += thumb_path.stat().st_size
            except OSError:
                pass

        return total

    def count(self) -> int:
        """Return the number of entries in the history.

        Returns
        -------
        int
            Number of history entries.
        """
        return len(self._entries)

    def repair_history(self) -> tuple[int, int]:
        """Check consistency between index and files on disk.

        1. Remove entries whose audio file is missing (stale entries).
        2. Count orphan audio files in ``audio/`` not referenced by
           any entry (reported but NOT deleted -- user may want them).

        Returns
        -------
        tuple[int, int]
            ``(removed_count, orphan_count)`` -- stale entries removed
            and orphan audio files found.
        """
        removed_count = 0
        orphan_count = 0

        # 1. Remove entries whose audio file is missing
        stale_ids: list[str] = []
        for entry_id, entry in self._entries.items():
            audio_path = self.history_dir / entry.audio_file
            if not audio_path.exists():
                logger.warning(
                    "History entry %s has no audio file on disk -- removing",
                    entry_id[:8],
                )
                stale_ids.append(entry_id)

        for entry_id in stale_ids:
            del self._entries[entry_id]
            removed_count += 1

        # 2. Count orphan audio files not referenced by any entry
        audio_dir = self.history_dir / "audio"
        if audio_dir.is_dir():
            referenced_files = {entry.audio_file for entry in self._entries.values()}
            for wav_path in audio_dir.glob("*.wav"):
                rel_path = f"audio/{wav_path.name}"
                if rel_path not in referenced_files:
                    logger.warning(
                        "Orphan audio file not in history index: %s",
                        wav_path.name,
                    )
                    orphan_count += 1

        # Persist if stale entries were removed
        if removed_count > 0:
            self._save_index()

        return (removed_count, orphan_count)
