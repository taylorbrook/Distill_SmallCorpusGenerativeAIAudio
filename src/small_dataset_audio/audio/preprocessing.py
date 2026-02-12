"""Audio preprocessing pipeline for training with optional caching.

Handles resampling to the project baseline (48kHz), peak normalisation,
data augmentation via :class:`AugmentationPipeline`, and tensor caching
as ``.pt`` files for fast training restarts.

Heavy dependencies (torch, torchaudio) are imported inside function
bodies, matching the project-wide lazy-import pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from small_dataset_audio.audio.augmentation import AugmentationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline.

    Attributes
    ----------
    target_sample_rate:
        Sample rate all files are resampled to.  Default 48kHz (project
        baseline).
    normalize:
        If ``True``, peak-normalise waveforms to ``[-1, 1]``.
    cache_dir:
        If set, preprocessed tensors are saved as ``.pt`` files in this
        directory.  ``None`` disables caching.
    augment:
        If ``True``, apply data augmentation during preprocessing.
    augmentation_config:
        Passed to :class:`AugmentationPipeline`.  ``None`` uses defaults.
    """

    target_sample_rate: int = 48_000
    normalize: bool = True
    cache_dir: str | None = None
    augment: bool = True
    augmentation_config: AugmentationConfig | None = None


# ---------------------------------------------------------------------------
# Single-file preprocessing
# ---------------------------------------------------------------------------


def preprocess_for_training(
    waveform: "torch.Tensor",
    sample_rate: int,
    config: PreprocessingConfig,
) -> "torch.Tensor":
    """Preprocess a single waveform for training.

    Resamples to ``config.target_sample_rate`` if the source differs and
    peak-normalises when ``config.normalize`` is ``True``.

    Parameters
    ----------
    waveform:
        Float32 tensor ``[channels, samples]``.
    sample_rate:
        Source sample rate of *waveform*.
    config:
        Preprocessing settings.

    Returns
    -------
    torch.Tensor
        Preprocessed waveform ``[channels, samples]``.
    """
    import torch  # noqa: WPS433

    # Resample if needed
    if sample_rate != config.target_sample_rate:
        from torchaudio.transforms import Resample  # noqa: WPS433

        resampler = Resample(
            orig_freq=sample_rate,
            new_freq=config.target_sample_rate,
        )
        waveform = resampler(waveform)

    # Peak normalisation
    if config.normalize:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    return waveform


# ---------------------------------------------------------------------------
# Full dataset preprocessing
# ---------------------------------------------------------------------------


def preprocess_dataset(
    files: list[Path],
    config: PreprocessingConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list["torch.Tensor"]:
    """Preprocess a list of audio files for training.

    1. Load each file via :func:`audio.io.load_audio` (handles resampling).
    2. Peak-normalise if ``config.normalize``.
    3. Optionally augment with :class:`AugmentationPipeline`.
    4. Optionally cache tensors as ``.pt`` files.

    Corrupt or unreadable files are skipped with a logged warning.

    Parameters
    ----------
    files:
        Paths to audio files.
    config:
        Preprocessing settings.
    progress_callback:
        ``callback(current_index, total_count, filename)`` called after
        each file.  ``None`` for silent operation.

    Returns
    -------
    list[torch.Tensor]
        All preprocessed (and optionally augmented) tensors.
    """
    import torch  # noqa: WPS433

    from small_dataset_audio.audio.augmentation import AugmentationPipeline
    from small_dataset_audio.audio.io import load_audio

    total = len(files)
    waveforms: list[torch.Tensor] = []

    # Phase 1: load and normalise each file
    for idx, filepath in enumerate(files):
        try:
            audio_file = load_audio(filepath, target_sample_rate=config.target_sample_rate)
            waveform = audio_file.waveform

            # Peak normalisation
            if config.normalize:
                peak = waveform.abs().max()
                if peak > 0:
                    waveform = waveform / peak

            waveforms.append(waveform)
        except Exception as exc:
            logger.warning(
                "Skipping corrupt file %s: %s: %s",
                filepath.name,
                type(exc).__name__,
                exc,
            )

        if progress_callback is not None:
            progress_callback(idx + 1, total, filepath.name)

    # Phase 2: augmentation (operates on entire loaded set)
    if config.augment and waveforms:
        pipeline = AugmentationPipeline(
            sample_rate=config.target_sample_rate,
            config=config.augmentation_config,
        )
        waveforms = pipeline.expand_dataset(waveforms)

    # Phase 3: cache as .pt files
    if config.cache_dir is not None:
        cache_path = Path(config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        for i, tensor in enumerate(waveforms):
            torch.save(tensor, cache_path / f"{i:04d}.pt")

    return waveforms


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------


def load_cached_dataset(cache_dir: Path) -> list["torch.Tensor"]:
    """Load all ``.pt`` tensors from a cache directory.

    Parameters
    ----------
    cache_dir:
        Directory containing ``.pt`` files saved by :func:`preprocess_dataset`.

    Returns
    -------
    list[torch.Tensor]
        Tensors loaded in filename-sorted order.

    Raises
    ------
    FileNotFoundError
        If *cache_dir* does not exist or contains no ``.pt`` files.
    """
    import torch  # noqa: WPS433

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

    pt_files = sorted(cache_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in cache directory: {cache_dir}")

    return [torch.load(f, weights_only=True) for f in pt_files]


def clear_cache(cache_dir: Path) -> int:
    """Delete all ``.pt`` files in *cache_dir*.

    Parameters
    ----------
    cache_dir:
        Directory to clean.

    Returns
    -------
    int
        Number of files deleted.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0

    pt_files = list(cache_dir.glob("*.pt"))
    for f in pt_files:
        f.unlink()
    return len(pt_files)
