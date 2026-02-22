"""Audio preprocessing pipeline for training with optional caching.

Handles resampling to the project baseline (48kHz), peak normalisation,
data augmentation via :class:`AugmentationPipeline`, and tensor caching
as ``.pt`` files for fast training restarts.

v2.0 adds :func:`preprocess_complex_spectrograms` which builds a disk
cache of 2-channel (magnitude + IF) mel spectrograms with manifest-based
change detection.  The cache is consumed by
:class:`~distill.training.dataset.CachedSpectrogramDataset`.

Heavy dependencies (torch, torchaudio) are imported inside function
bodies, matching the project-wide lazy-import pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from distill.audio.augmentation import AugmentationConfig

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

    from distill.audio.augmentation import AugmentationPipeline
    from distill.audio.io import load_audio

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


# ---------------------------------------------------------------------------
# v2.0 -- 2-Channel Complex Spectrogram Preprocessing & Caching
# ---------------------------------------------------------------------------


def load_cache_manifest(cache_dir: Path) -> dict | None:
    """Load and return the manifest dict from a cache directory.

    Parameters
    ----------
    cache_dir:
        Path to the cache directory containing ``manifest.json``.

    Returns
    -------
    dict | None
        Parsed manifest, or ``None`` if manifest is missing or invalid.
    """
    import json  # noqa: WPS433

    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def preprocess_complex_spectrograms(
    files: list[Path],
    dataset_dir: Path,
    complex_spectrogram_config: "ComplexSpectrogramConfig",
    augmentation_config: AugmentationConfig | None = None,
    augmentation_expansion: int = 0,
    chunk_samples: int = 48_000,
    sample_rate: int = 48_000,
    progress_callback: Callable[[str, int, int], None] | None = None,
    confirm_callback: Callable[[str], bool] | None = None,
) -> "tuple[Path, dict]":
    """Preprocess audio files into cached 2-channel spectrograms.

    Converts audio files into chunked, augmented, normalized 2-channel
    (magnitude + IF) mel spectrograms and caches them as ``.pt`` files.
    Includes manifest-based change detection for cache invalidation.

    Parameters
    ----------
    files:
        Audio file paths to process.
    dataset_dir:
        Root dataset directory.  Cache is stored at ``dataset_dir / ".cache"``.
    complex_spectrogram_config:
        Configuration for the ComplexSpectrogram computation.
    augmentation_config:
        Augmentation settings.  ``None`` disables augmentation.
    augmentation_expansion:
        Number of augmented copies per original chunk.
    chunk_samples:
        Chunk size in samples (default 48000 = 1s at 48kHz).
    sample_rate:
        Target sample rate (default 48000).
    progress_callback:
        ``callback(message, current, total)`` for progress updates.
    confirm_callback:
        ``callback(message) -> bool`` for disk usage confirmation.
        Called when estimated cache exceeds 100MB.

    Returns
    -------
    tuple[Path, dict]
        ``(cache_dir, normalization_stats)`` -- path to cache directory
        and the normalization statistics dict.

    Raises
    ------
    RuntimeError
        If user declines preprocessing via ``confirm_callback``.
    ValueError
        If no valid audio files can be loaded.
    """
    import json  # noqa: WPS433
    import math  # noqa: WPS433
    import os  # noqa: WPS433
    from datetime import datetime, timezone  # noqa: WPS433

    import torch  # noqa: WPS433

    from distill.audio.io import get_metadata, load_audio
    from distill.audio.spectrogram import ComplexSpectrogram

    # -----------------------------------------------------------------
    # 1. Cache directory setup
    # -----------------------------------------------------------------
    cache_dir = Path(dataset_dir) / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 2. Manifest check / change detection
    # -----------------------------------------------------------------
    manifest_path = cache_dir / "manifest.json"
    sorted_abs_paths = sorted(str(f.resolve()) for f in files)
    file_mtimes = {p: os.path.getmtime(p) for p in sorted_abs_paths}

    # Serialize configs for comparison
    spec_cfg_dict = {
        "if_masking_threshold": complex_spectrogram_config.if_masking_threshold,
        "n_fft": complex_spectrogram_config.n_fft,
        "hop_length": complex_spectrogram_config.hop_length,
        "n_mels": complex_spectrogram_config.n_mels,
    }
    aug_cfg_dict = None
    if augmentation_config is not None:
        from dataclasses import asdict as _asdict  # noqa: WPS433
        aug_cfg_dict = _asdict(augmentation_config)

    existing_manifest = load_cache_manifest(cache_dir)
    if existing_manifest is not None:
        # Compare file list
        existing_paths = [f["path"] for f in existing_manifest.get("files", [])]
        files_match = existing_paths == sorted_abs_paths
        # Compare mtimes
        mtimes_match = files_match and all(
            abs(f["mtime"] - file_mtimes[f["path"]]) < 0.01
            for f in existing_manifest.get("files", [])
        )
        # Compare configs
        spec_match = existing_manifest.get("complex_spectrogram_config") == spec_cfg_dict
        aug_match = existing_manifest.get("augmentation_config") == aug_cfg_dict
        expansion_match = existing_manifest.get("augmentation_expansion") == augmentation_expansion
        chunk_match = existing_manifest.get("chunk_samples") == chunk_samples
        sr_match = existing_manifest.get("sample_rate") == sample_rate

        if all([files_match, mtimes_match, spec_match, aug_match,
                expansion_match, chunk_match, sr_match]):
            print("Cache valid, skipping preprocessing", flush=True)
            return (cache_dir, existing_manifest["normalization_stats"])

        # Report what changed
        changes = []
        if not files_match:
            n_old = len(existing_paths)
            n_new = len(sorted_abs_paths)
            if n_new > n_old:
                changes.append(f"{n_new - n_old} files added")
            elif n_new < n_old:
                changes.append(f"{n_old - n_new} files removed")
            else:
                changes.append("file list changed")
        if not mtimes_match and files_match:
            changes.append("file modification times changed")
        if not spec_match:
            changes.append("spectrogram config changed")
        if not aug_match:
            changes.append("augmentation config changed")
        if not expansion_match:
            changes.append("augmentation expansion changed")
        if not chunk_match:
            changes.append("chunk size changed")
        if not sr_match:
            changes.append("sample rate changed")
        print(f"Cache invalidated: {', '.join(changes)}", flush=True)

        # Clear existing cache .pt files
        for pt_file in cache_dir.glob("*.pt"):
            pt_file.unlink()

    # -----------------------------------------------------------------
    # 3. Disk usage estimation
    # -----------------------------------------------------------------
    # Estimate n_chunks_per_file from first few files
    sample_count = min(3, len(files))
    total_samples_estimate = 0
    for i in range(sample_count):
        try:
            meta = get_metadata(files[i])
            if meta.sample_rate != sample_rate:
                file_samples = int(meta.num_frames * sample_rate / meta.sample_rate)
            else:
                file_samples = meta.num_frames
            total_samples_estimate += file_samples
        except Exception:
            continue

    if sample_count > 0 and total_samples_estimate > 0:
        avg_samples_per_file = total_samples_estimate / sample_count
        avg_chunks_per_file = math.ceil(avg_samples_per_file / chunk_samples)
    else:
        avg_chunks_per_file = 1

    n_mels = complex_spectrogram_config.n_mels
    hop_length = complex_spectrogram_config.hop_length
    time_frames = chunk_samples // hop_length + 1
    n_total_specs = len(files) * (1 + augmentation_expansion) * avg_chunks_per_file
    bytes_per_spec = 2 * n_mels * time_frames * 4  # 2 channels, float32
    estimated_bytes = int(n_total_specs * bytes_per_spec)

    # Human-readable size
    if estimated_bytes >= 1_073_741_824:
        size_str = f"{estimated_bytes / 1_073_741_824:.1f} GB"
    else:
        size_str = f"{estimated_bytes / 1_048_576:.1f} MB"
    print(f"Estimated cache size: {size_str}", flush=True)

    if confirm_callback is not None and estimated_bytes > 100 * 1_048_576:
        if not confirm_callback(f"Estimated cache size: {size_str}. Continue?"):
            raise RuntimeError("Preprocessing cancelled by user")

    # -----------------------------------------------------------------
    # 4. Load and chunk waveforms
    # -----------------------------------------------------------------
    all_chunks: list[torch.Tensor] = []
    valid_file_count = 0

    for file_idx, filepath in enumerate(files):
        try:
            audio_file = load_audio(filepath, target_sample_rate=sample_rate)
            waveform = audio_file.waveform  # [channels, samples]

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            total_file_samples = waveform.shape[1]
            n_chunks = math.ceil(total_file_samples / chunk_samples)

            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_samples
                end = start + chunk_samples
                chunk = waveform[:, start:end]

                # Zero-pad final chunk if shorter
                if chunk.shape[1] < chunk_samples:
                    padding = torch.zeros(1, chunk_samples - chunk.shape[1], dtype=chunk.dtype)
                    chunk = torch.cat([chunk, padding], dim=1)

                # Peak-normalize each chunk
                peak = chunk.abs().max()
                if peak > 1e-6:
                    chunk = chunk / peak

                all_chunks.append(chunk)

            valid_file_count += 1
        except Exception as exc:
            logger.warning(
                "Skipping corrupt file %s: %s: %s",
                filepath.name,
                type(exc).__name__,
                exc,
            )

        if progress_callback is not None:
            progress_callback("Loading audio", file_idx + 1, len(files))

    if not all_chunks:
        raise ValueError("No valid audio files found")

    # -----------------------------------------------------------------
    # 5. Augmentation
    # -----------------------------------------------------------------
    if augmentation_config is not None and augmentation_expansion > 0:
        from distill.audio.augmentation import AugmentationPipeline  # noqa: WPS433

        # Create pipeline with pitch shift disabled (too slow at 48kHz)
        aug_cfg = AugmentationConfig(
            pitch_shift_probability=0.0,
            speed_probability=augmentation_config.speed_probability
            if hasattr(augmentation_config, "speed_probability")
            else AugmentationConfig.speed_probability,
            noise_probability=augmentation_config.noise_probability
            if hasattr(augmentation_config, "noise_probability")
            else AugmentationConfig.noise_probability,
            volume_probability=augmentation_config.volume_probability
            if hasattr(augmentation_config, "volume_probability")
            else AugmentationConfig.volume_probability,
            expansion_ratio=augmentation_expansion,
        )
        pipeline = AugmentationPipeline(sample_rate=sample_rate, config=aug_cfg)

        original_count = len(all_chunks)
        augmented_chunks: list[torch.Tensor] = []
        for chunk in all_chunks:
            for _ in range(augmentation_expansion):
                aug_chunk = pipeline.augment(chunk)
                # Re-enforce fixed length after augmentation
                if aug_chunk.shape[1] > chunk_samples:
                    aug_chunk = aug_chunk[:, :chunk_samples]
                elif aug_chunk.shape[1] < chunk_samples:
                    pad = torch.zeros(1, chunk_samples - aug_chunk.shape[1], dtype=aug_chunk.dtype)
                    aug_chunk = torch.cat([aug_chunk, pad], dim=1)
                # Re-normalize after augmentation
                peak = aug_chunk.abs().max()
                if peak > 1e-6:
                    aug_chunk = aug_chunk / peak
                augmented_chunks.append(aug_chunk)
        all_chunks.extend(augmented_chunks)
        print(
            f"Augmentation: {original_count} original + {len(augmented_chunks)} augmented = {len(all_chunks)} total chunks",
            flush=True,
        )

    # -----------------------------------------------------------------
    # 6. Compute 2-channel spectrograms
    # -----------------------------------------------------------------
    complex_spec = ComplexSpectrogram(complex_spectrogram_config)

    all_spectrograms: list[torch.Tensor] = []
    batch_size = 32
    total_batches = math.ceil(len(all_chunks) / batch_size)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_chunks))
        batch_chunks = all_chunks[start:end]

        # Stack: each chunk is [1, samples] -> batch is [B, 1, samples]
        batch_tensor = torch.stack(batch_chunks, dim=0)  # [B, 1, samples]
        specs = complex_spec.waveform_to_complex_mel(batch_tensor)  # [B, 2, n_mels, time]

        # Split batch into individual spectrograms
        for i in range(specs.shape[0]):
            all_spectrograms.append(specs[i])  # [2, n_mels, time]

        if progress_callback is not None:
            progress_callback("Computing spectrograms", batch_idx + 1, total_batches)

    # -----------------------------------------------------------------
    # 7. Compute normalization statistics and normalize
    # -----------------------------------------------------------------
    norm_stats = complex_spec.compute_dataset_statistics(all_spectrograms)
    print(
        f"Normalization stats: mag_mean={norm_stats['mag_mean']:.4f}, "
        f"mag_std={norm_stats['mag_std']:.4f}, "
        f"if_mean={norm_stats['if_mean']:.4f}, "
        f"if_std={norm_stats['if_std']:.4f}",
        flush=True,
    )

    normalized_specs: list[torch.Tensor] = []
    for spec in all_spectrograms:
        normalized_specs.append(complex_spec.normalize(spec, norm_stats))

    # -----------------------------------------------------------------
    # 8. Save to cache
    # -----------------------------------------------------------------
    for idx, spec in enumerate(normalized_specs):
        torch.save(spec, cache_dir / f"{idx:06d}.pt")
        if progress_callback is not None:
            progress_callback("Saving cache", idx + 1, len(normalized_specs))

    # Actual cache size
    actual_cache_bytes = sum(
        f.stat().st_size for f in cache_dir.glob("*.pt")
    )

    # -----------------------------------------------------------------
    # 9. Write manifest
    # -----------------------------------------------------------------
    manifest = {
        "version": "2.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "file_count": valid_file_count,
        "files": [
            {"path": p, "mtime": file_mtimes[p]}
            for p in sorted_abs_paths
        ],
        "chunk_samples": chunk_samples,
        "sample_rate": sample_rate,
        "augmentation_expansion": augmentation_expansion,
        "augmentation_config": aug_cfg_dict,
        "complex_spectrogram_config": spec_cfg_dict,
        "normalization_stats": norm_stats,
        "total_spectrograms": len(normalized_specs),
        "cache_size_bytes": actual_cache_bytes,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Cache complete: {len(normalized_specs)} spectrograms saved to {cache_dir}",
        flush=True,
    )

    # -----------------------------------------------------------------
    # 10. Return
    # -----------------------------------------------------------------
    return (cache_dir, norm_stats)
