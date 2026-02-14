"""Preset management for slider configurations.

Public API:
- PresetEntry: dataclass for a single preset
- PresetManager: CRUD and folder management for model-scoped presets
"""

from small_dataset_audio.presets.manager import PresetEntry, PresetManager

__all__ = ["PresetEntry", "PresetManager"]
