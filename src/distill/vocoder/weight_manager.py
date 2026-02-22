"""BigVGAN weight download, caching, and loading.

Uses HuggingFace Hub for resumable download with progress indication.
After first download, weights are cached and no network access is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BIGVGAN_REPO_ID = "nvidia/bigvgan_v2_44khz_128band_512x"


def ensure_bigvgan_weights() -> Path:
    """Ensure BigVGAN weights are available locally, downloading if needed.

    Returns the directory containing the model files (config.json,
    bigvgan_generator.pt, etc.).

    On first call: downloads ~489MB generator weights with progress bar.
    On subsequent calls: returns cached path instantly (no network).

    Returns
    -------
    Path
        Local directory path containing BigVGAN model files.

    Raises
    ------
    OSError
        If download fails and no cached version exists.
    """
    from huggingface_hub import snapshot_download

    # Try online download first (will use cache if already downloaded)
    try:
        logger.info(
            "Ensuring BigVGAN weights are available (repo: %s)...",
            BIGVGAN_REPO_ID,
        )
        local_dir = snapshot_download(repo_id=BIGVGAN_REPO_ID)
        logger.info("BigVGAN weights ready at: %s", local_dir)
        return Path(local_dir)
    except Exception as online_err:
        logger.debug("Online download failed: %s", online_err)

    # Fall back to local-only mode (cached from a previous download)
    try:
        logger.info("Network unavailable, trying cached weights...")
        local_dir = snapshot_download(
            repo_id=BIGVGAN_REPO_ID,
            local_files_only=True,
        )
        logger.info("BigVGAN weights loaded from cache: %s", local_dir)
        return Path(local_dir)
    except Exception as offline_err:
        raise OSError(
            f"BigVGAN weights not available. Download failed ({online_err}) "
            f"and no cached version found ({offline_err}). "
            "Please ensure you have internet access for the first download "
            f"of the BigVGAN model (~489 MB from '{BIGVGAN_REPO_ID}')."
        ) from offline_err


def is_bigvgan_cached() -> bool:
    """Check if BigVGAN weights are already cached (no download needed)."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=BIGVGAN_REPO_ID, local_files_only=True)
        return True
    except Exception:
        return False
