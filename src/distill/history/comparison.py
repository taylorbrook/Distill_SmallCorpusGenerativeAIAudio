"""A/B comparison runtime state for comparing two generations.

Provides :class:`ABComparison`, an ephemeral (not persisted) dataclass that
holds references to two history entries (or one entry + the current live
generation) and provides audio path resolution and a "keep this one" action.

Locked decisions honored:
- Toggle A/B button: single play control, toggle switches between A and B
- Audio-only comparison: no visual parameter diff
- Supports: current/latest vs history entry, OR any two history entries
- "Keep this one": saves winner's parameters as a preset

Design notes:
- ``from __future__ import annotations`` for modern type hints.
- ``TYPE_CHECKING`` guard for GenerationHistory, PresetManager, PresetEntry,
  HistoryEntry to avoid circular imports.
- ``logging.getLogger(__name__)`` for module-level logger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distill.history.store import GenerationHistory, HistoryEntry
    from distill.presets.manager import PresetEntry, PresetManager

logger = logging.getLogger(__name__)


@dataclass
class ABComparison:
    """Runtime state for A/B comparison between two generations.

    This is ephemeral UI state -- not persisted to disk. It holds
    references to two history entries (or one entry + the current
    live generation) and provides audio path resolution and a
    "keep this one" action.

    Locked decisions honored:
    - Toggle A/B button: single play control, toggle switches between A and B
    - Audio-only comparison: no visual parameter diff
    - Supports: current/latest vs history entry, OR any two history entries
    - "Keep this one": saves winner's parameters as a preset
    """

    entry_a_id: str | None  # History entry ID. None = current/live generation
    entry_b_id: str | None  # History entry ID. None = current/live generation
    active_side: str = "a"  # "a" or "b" -- which side is currently playing

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def toggle(self) -> str:
        """Switch active_side between "a" and "b".

        Returns
        -------
        str
            The new active side ("a" or "b").
        """
        self.active_side = "b" if self.active_side == "a" else "a"
        return self.active_side

    def get_active_entry_id(self) -> str | None:
        """Return the entry_id for the currently active side.

        Returns
        -------
        str | None
            Entry ID, or ``None`` if active side is a live generation.
        """
        if self.active_side == "a":
            return self.entry_a_id
        return self.entry_b_id

    def get_audio_paths(
        self, history: GenerationHistory
    ) -> tuple[Path | None, Path | None]:
        """Return absolute audio file paths for both sides.

        Parameters
        ----------
        history : GenerationHistory
            The history store to look up entries from.

        Returns
        -------
        tuple[Path | None, Path | None]
            ``(path_a, path_b)`` -- absolute paths to WAV files.
            ``None`` means live generation (UI handles playback from buffer).

        Raises
        ------
        ValueError
            If an entry_id is not ``None`` but not found in history.
        """
        path_a = self._resolve_audio_path(self.entry_a_id, "a", history)
        path_b = self._resolve_audio_path(self.entry_b_id, "b", history)
        return (path_a, path_b)

    def get_entry(
        self, side: str, history: GenerationHistory
    ) -> HistoryEntry | None:
        """Return the HistoryEntry for the given side.

        Parameters
        ----------
        side : str
            ``"a"`` or ``"b"``.
        history : GenerationHistory
            The history store to look up entries from.

        Returns
        -------
        HistoryEntry | None
            The entry, or ``None`` if that side is a live generation.

        Raises
        ------
        ValueError
            If ``side`` is not ``"a"`` or ``"b"``.
        """
        if side not in ("a", "b"):
            raise ValueError(f"Side must be 'a' or 'b', got: {side!r}")

        entry_id = self.entry_a_id if side == "a" else self.entry_b_id
        if entry_id is None:
            return None
        return history.get(entry_id)

    def keep_winner(
        self,
        winner: str,
        preset_name: str,
        history: GenerationHistory,
        preset_manager: PresetManager,
    ) -> PresetEntry:
        """Save the winning side's parameters as a preset.

        Parameters
        ----------
        winner : str
            ``"a"`` or ``"b"`` -- the winning side.
        preset_name : str
            Name for the new preset.
        history : GenerationHistory
            The history store to look up the winning entry.
        preset_manager : PresetManager
            The preset manager to save the preset through.

        Returns
        -------
        PresetEntry
            The created preset entry.

        Raises
        ------
        ValueError
            If ``winner`` is not ``"a"`` or ``"b"``, or if the winning
            side is a live generation (entry_id is None), or if the
            entry has no slider_positions.
        KeyError
            If the winning entry is not found in history.
        """
        if winner not in ("a", "b"):
            raise ValueError(f"Winner must be 'a' or 'b', got: {winner!r}")

        entry_id = self.entry_a_id if winner == "a" else self.entry_b_id

        if entry_id is None:
            raise ValueError(
                "Cannot save live generation as preset -- "
                "it must be in history first"
            )

        entry = history.get(entry_id)
        if entry is None:
            raise KeyError(f"History entry not found: {entry_id}")

        if entry.slider_positions is None:
            raise ValueError(
                f"Entry {entry_id[:8]} has no slider_positions -- "
                "cannot create preset"
            )

        logger.info(
            "Keeping winner '%s' (entry %s) as preset '%s'",
            winner,
            entry_id[:8],
            preset_name,
        )

        return preset_manager.save_preset(
            name=preset_name,
            slider_positions=entry.slider_positions,
            n_components=entry.n_components,
            seed=entry.seed,
        )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_current_and_history(cls, history_entry_id: str) -> ABComparison:
        """Compare current/latest generation (A) against a history entry (B).

        Default mode (locked decision): A = live generation, B = history entry.

        Parameters
        ----------
        history_entry_id : str
            ID of the history entry to compare against.

        Returns
        -------
        ABComparison
            New comparison with ``entry_a_id=None`` (live) and
            ``entry_b_id=history_entry_id``.
        """
        return cls(entry_a_id=None, entry_b_id=history_entry_id)

    @classmethod
    def from_two_entries(
        cls, entry_a_id: str, entry_b_id: str
    ) -> ABComparison:
        """Compare any two history entries.

        Parameters
        ----------
        entry_a_id : str
            ID of the first history entry (side A).
        entry_b_id : str
            ID of the second history entry (side B).

        Returns
        -------
        ABComparison
            New comparison with both entry IDs set.
        """
        return cls(entry_a_id=entry_a_id, entry_b_id=entry_b_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_audio_path(
        self,
        entry_id: str | None,
        side: str,
        history: GenerationHistory,
    ) -> Path | None:
        """Resolve an entry_id to an absolute audio file path.

        Parameters
        ----------
        entry_id : str | None
            History entry ID, or ``None`` for live generation.
        side : str
            ``"a"`` or ``"b"`` (for error messages only).
        history : GenerationHistory
            The history store.

        Returns
        -------
        Path | None
            Absolute path to WAV file, or ``None`` for live generation.

        Raises
        ------
        ValueError
            If entry_id is not found in history.
        """
        if entry_id is None:
            return None

        entry = history.get(entry_id)
        if entry is None:
            raise ValueError(
                f"Side {side}: history entry {entry_id} not found"
            )

        return history.history_dir / entry.audio_file
