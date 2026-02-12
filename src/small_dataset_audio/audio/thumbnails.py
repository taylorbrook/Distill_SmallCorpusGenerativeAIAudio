"""Waveform thumbnail generation for dataset visualization.

Generates compact PNG images of audio waveforms for quick visual
inspection.  Thumbnails are cached in a `.thumbnails/` directory with
mtime-based invalidation so they're only regenerated when source files
change.

Design notes:
- matplotlib.use('Agg') is called BEFORE importing pyplot to avoid
  TclError on headless systems (research pitfall #7).
- Heavy imports (matplotlib, numpy, torch) are lazy -- inside function
  bodies, matching the Phase 1 lazy-import pattern.
- plt.close(fig) is always called to prevent memory leaks when
  generating many thumbnails in a batch.
- Per-file try/except in batch generation so one failure doesn't stop
  the rest.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_waveform_thumbnail(
    waveform: "np.ndarray | torch.Tensor",  # noqa: F821
    output_path: Path,
    width: int = 800,
    height: int = 120,
    color: str = "#4A90D9",
) -> None:
    """Generate a compact waveform thumbnail PNG.

    Args:
        waveform: Audio waveform as numpy array or torch Tensor.
            Shape ``[channels, samples]`` or ``[samples]``.
        output_path: Where to write the PNG file.
        width: Image width in pixels.
        height: Image height in pixels.
        color: Fill color for the waveform visualization.
    """
    import numpy as np  # noqa: WPS433

    # Ensure Agg backend BEFORE importing pyplot (headless pitfall #7)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402, WPS433

    # Convert torch.Tensor to numpy if needed
    try:
        import torch  # noqa: WPS433
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
    except ImportError:
        pass  # torch not available -- waveform must already be numpy

    waveform = np.asarray(waveform, dtype=np.float32)

    # Mix to mono if multi-channel: shape [channels, samples] -> [samples]
    if waveform.ndim == 2:
        mono = waveform.mean(axis=0)
    elif waveform.ndim == 1:
        mono = waveform
    else:
        # Unexpected shape -- flatten and hope for the best
        mono = waveform.reshape(-1)

    # Downsample for display efficiency
    target_points = width * 2
    if len(mono) > target_points:
        indices = np.linspace(0, len(mono) - 1, target_points, dtype=int)
        mono = mono[indices]

    # Normalize time axis to [0, 1]
    time = np.linspace(0, 1, len(mono))

    # Create figure
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Symmetric fill for waveform visualization
    ax.fill_between(time, mono, -mono, alpha=0.7, color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        str(output_path),
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)


def generate_dataset_thumbnails(
    file_metadata_pairs: "list[tuple[Path, AudioMetadata]]",  # noqa: F821
    thumbnail_dir: Path,
    width: int = 800,
    height: int = 120,
    force: bool = False,
) -> dict[Path, Path]:
    """Generate waveform thumbnails for a list of audio files with caching.

    Thumbnails are cached based on file modification time: a thumbnail
    is only regenerated when the source audio file is newer than the
    existing thumbnail (or when *force* is ``True``).

    Args:
        file_metadata_pairs: List of ``(audio_path, metadata)`` tuples.
        thumbnail_dir: Directory in which to store thumbnail PNGs.
        width: Image width in pixels.
        height: Image height in pixels.
        force: If ``True``, regenerate all thumbnails regardless of cache.

    Returns:
        Dict mapping audio file paths to their thumbnail paths for all
        successfully generated thumbnails.
    """
    thumbnail_dir = Path(thumbnail_dir)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    results: dict[Path, Path] = {}

    for audio_path, _metadata in file_metadata_pairs:
        thumb_path = thumbnail_dir / f"{audio_path.stem}.png"

        # Skip if cached and up-to-date (mtime comparison)
        if not force and thumb_path.exists():
            if thumb_path.stat().st_mtime >= audio_path.stat().st_mtime:
                results[audio_path] = thumb_path
                continue

        try:
            # Lazy import to avoid loading audio.io at module level
            from small_dataset_audio.audio.io import load_audio  # noqa: WPS433

            audio_file = load_audio(audio_path)
            waveform = audio_file.waveform.detach().cpu().numpy()

            generate_waveform_thumbnail(
                waveform,
                thumb_path,
                width=width,
                height=height,
            )
            results[audio_path] = thumb_path

        except Exception as exc:
            logger.warning(
                "Failed to generate thumbnail for %s: %s",
                audio_path.name,
                exc,
            )
            continue

    return results
