"""Audio generation from trained models.

Public API re-exports from :mod:`inference.generation`,
:mod:`inference.export`, :mod:`inference.chunking`,
:mod:`inference.stereo`, :mod:`inference.spatial`,
:mod:`inference.quality`, and :mod:`inference.blending`.
"""

from distill.inference.generation import (
    GenerationConfig,
    GenerationResult,
    GenerationPipeline,
    generate_audio_from_prior,
)
from distill.inference.export import (
    ExportFormat,
    FORMAT_EXTENSIONS,
    export_wav,
    export_mp3,
    export_flac,
    export_ogg,
    export_audio,
    write_sidecar_json,
    BIT_DEPTH_MAP,
    SAMPLE_RATE_OPTIONS,
)
from distill.inference.chunking import (
    slerp,
    crossfade_chunks,
    generate_chunks_crossfade,
    generate_chunks_latent_interp,
)
from distill.inference.stereo import (
    apply_mid_side_widening,
    create_dual_seed_stereo,
    peak_normalize,
)
from distill.inference.spatial import (
    SpatialMode,
    SpatialConfig,
    apply_spatial,
    apply_spatial_to_dual_seed,
    migrate_stereo_config,
)
from distill.inference.quality import (
    compute_snr_db,
    detect_clipping,
    compute_quality_score,
)
from distill.inference.blending import (
    BlendMode,
    BlendEngine,
    ModelSlot,
    MAX_BLEND_MODELS,
)

__all__ = [
    # generation.py
    "GenerationConfig",
    "GenerationResult",
    "GenerationPipeline",
    "generate_audio_from_prior",
    # export.py
    "ExportFormat",
    "FORMAT_EXTENSIONS",
    "export_wav",
    "export_mp3",
    "export_flac",
    "export_ogg",
    "export_audio",
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
    # spatial.py
    "SpatialMode",
    "SpatialConfig",
    "apply_spatial",
    "apply_spatial_to_dual_seed",
    "migrate_stereo_config",
    # quality.py
    "compute_snr_db",
    "detect_clipping",
    "compute_quality_score",
    # blending.py
    "BlendMode",
    "BlendEngine",
    "ModelSlot",
    "MAX_BLEND_MODELS",
]
