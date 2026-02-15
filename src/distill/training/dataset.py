"""PyTorch Dataset wrapper for audio training with file-level splitting.

Loads audio files on demand, chunks them into fixed-length segments
(default 1 second at 48kHz = 48000 samples), and returns raw waveform
tensors.  Mel spectrogram conversion is NOT done here -- it happens in
the training loop on GPU for efficiency.

**Critical design choice:** Validation split happens at the *file* level,
not the *chunk* level, to prevent data leakage (chunks from the same
file appearing in both train and validation sets).

Heavy dependencies (torch, torchaudio, audio.io) are imported lazily
inside method/function bodies, matching the project-wide pattern.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from distill.training.config import TrainingConfig

logger = logging.getLogger(__name__)

# Fixed seed for reproducible file-level splits across runs.
_SPLIT_SEED: int = 42


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class AudioTrainingDataset:
    """PyTorch-compatible Dataset that chunks audio files into fixed-length segments.

    Pre-scans files at construction time to build a chunk index
    ``(file_idx, chunk_start_sample)`` without loading waveform data.
    Audio is loaded lazily in ``__getitem__``.

    Parameters
    ----------
    file_paths:
        Paths to audio files (should all be loadable by ``audio.io.load_audio``).
    chunk_samples:
        Number of samples per chunk (default 48000 = 1 second at 48kHz).
    sample_rate:
        Target sample rate for loaded audio.
    augmentation_pipeline:
        Optional :class:`AugmentationPipeline` instance.  When provided,
        each chunk is augmented with 50% probability.
    """

    def __init__(
        self,
        file_paths: list[Path],
        chunk_samples: int = 48_000,
        sample_rate: int = 48_000,
        augmentation_pipeline: object | None = None,
    ) -> None:
        self.file_paths = list(file_paths)
        self.chunk_samples = chunk_samples
        self.sample_rate = sample_rate
        self.augmentation_pipeline = augmentation_pipeline

        # Build chunk index: list of (file_idx, chunk_start_sample)
        self._chunk_index: list[tuple[int, int]] = []
        self._build_chunk_index()

    # -----------------------------------------------------------------
    # Dataset protocol
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of fixed-length chunks across all files."""
        return len(self._chunk_index)

    def __getitem__(self, idx: int) -> "torch.Tensor":
        """Load and return a single chunk as ``[1, chunk_samples]`` tensor.

        Loads the parent audio file, extracts the chunk slice, zero-pads
        if the chunk is shorter than ``chunk_samples`` (final chunk of a
        file), and converts to mono.

        If ``augmentation_pipeline`` is set, applies augmentation with
        50% probability per chunk.

        Parameters
        ----------
        idx:
            Chunk index in ``[0, len(self))``.

        Returns
        -------
        torch.Tensor
            Float32 waveform tensor with shape ``[1, chunk_samples]``.
        """
        import torch  # noqa: WPS433

        from distill.audio.io import load_audio

        file_idx, chunk_start = self._chunk_index[idx]
        filepath = self.file_paths[file_idx]

        # Load full file (load_audio handles resampling + format)
        audio_file = load_audio(filepath, target_sample_rate=self.sample_rate)
        waveform = audio_file.waveform  # [channels, samples]

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Extract chunk
        chunk_end = chunk_start + self.chunk_samples
        chunk = waveform[:, chunk_start:chunk_end]

        # Zero-pad if this is the final (possibly short) chunk
        actual_samples = chunk.shape[1]
        if actual_samples < self.chunk_samples:
            padding = torch.zeros(
                1, self.chunk_samples - actual_samples,
                dtype=chunk.dtype,
            )
            chunk = torch.cat([chunk, padding], dim=1)

        # Apply augmentation with 50% probability
        if self.augmentation_pipeline is not None and random.random() < 0.5:
            chunk = self.augmentation_pipeline.augment(chunk)
            # Re-enforce fixed length after augmentation (speed perturbation
            # can change duration).
            if chunk.shape[1] > self.chunk_samples:
                chunk = chunk[:, : self.chunk_samples]
            elif chunk.shape[1] < self.chunk_samples:
                padding = torch.zeros(
                    1, self.chunk_samples - chunk.shape[1], dtype=chunk.dtype,
                )
                chunk = torch.cat([chunk, padding], dim=1)

        return chunk

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _build_chunk_index(self) -> None:
        """Pre-scan files to build ``(file_idx, chunk_start)`` pairs."""
        from distill.audio.io import get_metadata

        for file_idx, filepath in enumerate(self.file_paths):
            try:
                meta = get_metadata(filepath)
                # After resampling, the number of samples changes
                if meta.sample_rate != self.sample_rate:
                    total_samples = int(
                        meta.num_frames * self.sample_rate / meta.sample_rate
                    )
                else:
                    total_samples = meta.num_frames

                num_chunks = math.ceil(total_samples / self.chunk_samples)
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * self.chunk_samples
                    self._chunk_index.append((file_idx, chunk_start))
            except Exception as exc:
                logger.warning(
                    "Skipping file %s during chunk indexing: %s",
                    filepath.name,
                    exc,
                )


# ---------------------------------------------------------------------------
# Data Loader Factory
# ---------------------------------------------------------------------------


def create_data_loaders(
    files: list[Path],
    config: "TrainingConfig",
    augmentation_pipeline: object | None = None,
) -> "tuple[DataLoader, DataLoader]":
    """Create train and validation DataLoaders with file-level splitting.

    **File-level split** prevents data leakage: chunks from the same
    audio file never appear in both training and validation sets.

    Uses a fixed random seed (``_SPLIT_SEED``) for reproducibility.

    Parameters
    ----------
    files:
        All audio file paths for the dataset.
    config:
        Training configuration (provides ``val_fraction``, ``batch_size``,
        ``num_workers``, ``device``, ``chunk_duration_s``).
    augmentation_pipeline:
        Optional augmentation applied to training chunks only.
        Validation set is never augmented.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, val_loader)`` pair.
    """
    from torch.utils.data import DataLoader  # noqa: WPS433

    # Reproducible shuffle for file-level split
    rng = random.Random(_SPLIT_SEED)
    shuffled_files = list(files)
    rng.shuffle(shuffled_files)

    # Split at file level
    val_count = max(1, int(len(shuffled_files) * config.val_fraction))
    train_files = shuffled_files[val_count:]
    val_files = shuffled_files[:val_count]

    # Ensure at least 1 file in each split
    if not train_files and val_files:
        train_files = [val_files.pop()]
    if not val_files and train_files:
        val_files = [train_files.pop()]

    chunk_samples = int(config.chunk_duration_s * 48_000)
    pin_memory = config.device != "cpu"

    train_dataset = AudioTrainingDataset(
        file_paths=train_files,
        chunk_samples=chunk_samples,
        augmentation_pipeline=augmentation_pipeline,
    )
    val_dataset = AudioTrainingDataset(
        file_paths=val_files,
        chunk_samples=chunk_samples,
        augmentation_pipeline=None,  # Never augment validation data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
