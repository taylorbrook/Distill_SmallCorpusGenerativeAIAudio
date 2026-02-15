"""Spatial audio processing replacing Phase 4 stereo system.

Provides stereo (mid-side widening with depth), binaural (HRTF-based),
and mono output modes.  Replaces the simple ``stereo_width`` parameter
with a two-dimensional spatial system (width + depth) and adds binaural
output mode for immersive headphone listening.

Design notes:
- Lazy imports for numpy (project pattern).
- SpatialMode enum selects between stereo, binaural, and mono.
- SpatialConfig dataclass holds mode + width + depth.
- apply_spatial dispatches to the correct processing for each mode.
- Stereo mode reuses mid-side widening from stereo.py plus depth effect.
- Binaural mode delegates to hrtf.py for HRTF convolution.
- migrate_stereo_config provides backward compatibility for old presets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SpatialMode(str, Enum):
    """Output spatial mode selector.

    Values
    ------
    STEREO
        Mid-side stereo with width + depth controls.
    BINAURAL
        HRTF-based binaural rendering for headphone listening.
    MONO
        Mono output (no spatial processing).
    """

    STEREO = "stereo"
    BINAURAL = "binaural"
    MONO = "mono"


@dataclass
class SpatialConfig:
    """Configuration for spatial audio processing.

    Attributes
    ----------
    mode : SpatialMode
        Output mode.  Default ``MONO``.
    width : float
        Spatial width.  ``0.0`` = mono center, ``1.0`` = natural,
        up to ``1.5`` = exaggerated.  Default ``0.7``.
    depth : float
        Front-back depth.  ``0.0`` = close/intimate,
        ``1.0`` = distant/diffuse.  Default ``0.5``.
    """

    mode: SpatialMode = SpatialMode.MONO
    width: float = 0.7
    depth: float = 0.5

    def validate(self) -> None:
        """Validate that all fields are within acceptable ranges.

        Raises
        ------
        ValueError
            If width or depth are out of range, or mode is invalid.
        """
        if not isinstance(self.mode, SpatialMode):
            raise ValueError(
                f"Invalid spatial mode: {self.mode!r}. "
                f"Must be one of {[m.value for m in SpatialMode]}"
            )
        if not (0.0 <= self.width <= 1.5):
            raise ValueError(
                f"Spatial width must be in [0.0, 1.5], got {self.width}"
            )
        if not (0.0 <= self.depth <= 1.0):
            raise ValueError(
                f"Spatial depth must be in [0.0, 1.0], got {self.depth}"
            )


def apply_spatial(
    mono: "np.ndarray",
    config: SpatialConfig,
    sample_rate: int = 48_000,
) -> "np.ndarray":
    """Apply spatial processing to mono audio based on config mode.

    Parameters
    ----------
    mono : np.ndarray
        1-D mono audio array ``[samples]``.
    config : SpatialConfig
        Spatial configuration with mode, width, and depth.
    sample_rate : int
        Audio sample rate in Hz.  Default ``48_000``.

    Returns
    -------
    np.ndarray
        Processed audio:
        - MONO: 1-D ``[samples]``
        - STEREO: 2-D ``[2, samples]``
        - BINAURAL: 2-D ``[2, samples]``
    """
    import numpy as np  # noqa: WPS433

    mono = np.asarray(mono, dtype=np.float32).ravel()

    if config.mode == SpatialMode.MONO:
        return mono

    if config.mode == SpatialMode.STEREO:
        return _apply_stereo_with_depth(mono, config, sample_rate)

    if config.mode == SpatialMode.BINAURAL:
        return _apply_binaural(mono, config, sample_rate)

    # Unreachable if validate() was called, but defensive
    raise ValueError(f"Unknown spatial mode: {config.mode!r}")


def apply_spatial_to_dual_seed(
    left_mono: "np.ndarray",
    right_mono: "np.ndarray",
    config: SpatialConfig,
    sample_rate: int = 48_000,
) -> "np.ndarray":
    """Apply spatial processing to pre-generated dual-seed L/R channels.

    For dual-seed stereo generation: applies spatial processing to two
    independently generated mono signals.

    Parameters
    ----------
    left_mono : np.ndarray
        1-D mono array for the left channel.
    right_mono : np.ndarray
        1-D mono array for the right channel.
    config : SpatialConfig
        Spatial configuration with mode, width, and depth.
    sample_rate : int
        Audio sample rate in Hz.  Default ``48_000``.

    Returns
    -------
    np.ndarray
        Processed audio:
        - MONO: 1-D ``[samples]`` (average of both channels)
        - STEREO: 2-D ``[2, samples]``
        - BINAURAL: 2-D ``[2, samples]``
    """
    import numpy as np  # noqa: WPS433

    left_mono = np.asarray(left_mono, dtype=np.float32).ravel()
    right_mono = np.asarray(right_mono, dtype=np.float32).ravel()

    if config.mode == SpatialMode.MONO:
        # Average both channels to mono
        max_len = max(len(left_mono), len(right_mono))
        if len(left_mono) < max_len:
            left_mono = np.pad(left_mono, (0, max_len - len(left_mono)))
        if len(right_mono) < max_len:
            right_mono = np.pad(right_mono, (0, max_len - len(right_mono)))
        return ((left_mono + right_mono) * 0.5).astype(np.float32)

    if config.mode == SpatialMode.STEREO:
        from small_dataset_audio.inference.stereo import create_dual_seed_stereo  # noqa: WPS433

        stereo = create_dual_seed_stereo(left_mono, right_mono)
        # Apply depth effect to the combined stereo result
        return _apply_depth_to_stereo(stereo, config.depth, sample_rate)

    if config.mode == SpatialMode.BINAURAL:
        # Combine to mono first, then apply binaural
        max_len = max(len(left_mono), len(right_mono))
        if len(left_mono) < max_len:
            left_mono = np.pad(left_mono, (0, max_len - len(left_mono)))
        if len(right_mono) < max_len:
            right_mono = np.pad(right_mono, (0, max_len - len(right_mono)))
        combined_mono = ((left_mono + right_mono) * 0.5).astype(np.float32)
        return _apply_binaural(combined_mono, config, sample_rate)

    raise ValueError(f"Unknown spatial mode: {config.mode!r}")


def migrate_stereo_config(
    stereo_mode: str,
    stereo_width: float,
) -> SpatialConfig:
    """Map old Phase 4 stereo_mode/stereo_width to new SpatialConfig.

    Provides backward compatibility when loading old presets or generation
    history that used the Phase 4 stereo parameters.

    Parameters
    ----------
    stereo_mode : str
        Old stereo mode string: ``"mono"``, ``"mid_side"``, or ``"dual_seed"``.
    stereo_width : float
        Old stereo width parameter (0.0-1.5).

    Returns
    -------
    SpatialConfig
        Equivalent spatial configuration.
    """
    mode_lower = stereo_mode.lower().strip()

    if mode_lower == "mono":
        return SpatialConfig(mode=SpatialMode.MONO)

    if mode_lower == "mid_side":
        return SpatialConfig(
            mode=SpatialMode.STEREO,
            width=stereo_width,
            depth=0.3,
        )

    if mode_lower == "dual_seed":
        return SpatialConfig(
            mode=SpatialMode.STEREO,
            width=0.7,
            depth=0.3,
        )

    logger.warning("Unknown legacy stereo mode %r; defaulting to MONO", stereo_mode)
    return SpatialConfig(mode=SpatialMode.MONO)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_stereo_with_depth(
    mono: "np.ndarray",
    config: SpatialConfig,
    sample_rate: int,
) -> "np.ndarray":
    """Apply mid-side stereo widening with depth control.

    Uses the existing ``apply_mid_side_widening`` from stereo.py for the
    width dimension, then adds a depth effect using early reflections.
    """
    from small_dataset_audio.inference.stereo import apply_mid_side_widening  # noqa: WPS433

    # Apply mid-side widening for the width dimension
    stereo = apply_mid_side_widening(mono, width=config.width, sample_rate=sample_rate)

    # Apply depth effect
    return _apply_depth_to_stereo(stereo, config.depth, sample_rate)


def _apply_depth_to_stereo(
    stereo: "np.ndarray",
    depth: float,
    sample_rate: int,
) -> "np.ndarray":
    """Apply depth control to a stereo signal using early reflection pattern.

    Close (depth=0.0): no change (direct sound emphasis).
    Far (depth=1.0): mix in a delayed copy at ~40ms for diffuse character.
    Intermediate depths interpolate delay and mix level.
    """
    import numpy as np  # noqa: WPS433

    if depth < 0.01:
        return stereo

    # Depth-dependent delay: 0ms at depth=0 to 40ms at depth=1.0
    delay_ms = depth * 40.0
    delay_samples = int(delay_ms * sample_rate / 1000.0)

    if delay_samples < 1 or delay_samples >= stereo.shape[1]:
        return stereo

    # Mix level scaled by depth (max 0.15 for subtle effect)
    mix_level = depth * 0.15

    # Create delayed copy mixed into the output
    result = stereo.copy()
    result[:, delay_samples:] += mix_level * stereo[:, :-delay_samples]

    return result.astype(np.float32)


def _apply_binaural(
    mono: "np.ndarray",
    config: SpatialConfig,
    sample_rate: int,
) -> "np.ndarray":
    """Apply HRTF-based binaural rendering."""
    from small_dataset_audio.audio.hrtf import load_hrtf, apply_binaural  # noqa: WPS433

    hrtf = load_hrtf(target_sample_rate=sample_rate)
    return apply_binaural(mono, hrtf, width=config.width, depth=config.depth)
