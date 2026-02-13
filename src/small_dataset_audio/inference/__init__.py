"""Audio generation from trained models.

Public API re-exports from :mod:`inference.generation`,
:mod:`inference.export`, :mod:`inference.chunking`,
:mod:`inference.stereo`, and :mod:`inference.quality`.
"""

from small_dataset_audio.inference.generation import (
    GenerationConfig,
    GenerationResult,
    GenerationPipeline,
)
from small_dataset_audio.inference.export import (
    export_wav,
    write_sidecar_json,
    BIT_DEPTH_MAP,
    SAMPLE_RATE_OPTIONS,
)
from small_dataset_audio.inference.chunking import (
    slerp,
    crossfade_chunks,
    generate_chunks_crossfade,
    generate_chunks_latent_interp,
)
from small_dataset_audio.inference.stereo import (
    apply_mid_side_widening,
    create_dual_seed_stereo,
    peak_normalize,
)
from small_dataset_audio.inference.quality import (
    compute_snr_db,
    detect_clipping,
    compute_quality_score,
)

__all__ = [
    # generation.py
    "GenerationConfig",
    "GenerationResult",
    "GenerationPipeline",
    # export.py
    "export_wav",
    "write_sidecar_json",
    "BIT_DEPTH_MAP",
    "SAMPLE_RATE_OPTIONS",
    # chunking.py
    "slerp",
    "crossfade_chunks",
    "generate_chunks_crossfade",
    "generate_chunks_latent_interp",
    # stereo.py
    "apply_mid_side_widening",
    "create_dual_seed_stereo",
    "peak_normalize",
    # quality.py
    "compute_snr_db",
    "detect_clipping",
    "compute_quality_score",
]
