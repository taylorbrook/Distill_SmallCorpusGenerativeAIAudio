"""Neural network model definitions (VAE v1.0 and VQ-VAE v1.1).

Public API re-exports from :mod:`models.vae`, :mod:`models.vqvae`,
:mod:`models.losses`, :mod:`models.persistence`, and
:mod:`training.config`.
"""

# ---------------------------------------------------------------------------
# v1.0 exports (VAE) -- keep until Phase 13 migration completes
# ---------------------------------------------------------------------------

from distill.models.losses import (
    compute_kl_divergence,
    get_kl_weight,
    vae_loss,
)
from distill.models.persistence import (
    MODEL_FILE_EXTENSION,
    LoadedModel,
    ModelMetadata,
    SAVED_MODEL_VERSION,
    delete_model,
    load_model,
    save_model,
    save_model_from_checkpoint,
)

# ---------------------------------------------------------------------------
# v1.1 persistence exports (VQ-VAE v2 format) -- added in Phase 13
# ---------------------------------------------------------------------------

from distill.models.persistence import (
    SAVED_MODEL_VERSION_V2,
    LoadedVQModel,
    load_model_v2,
    save_model_v2,
)
from distill.models.vae import ConvDecoder, ConvEncoder, ConvVAE

# ---------------------------------------------------------------------------
# v1.1 exports (VQ-VAE) -- added in Phase 12
# ---------------------------------------------------------------------------

from distill.models.losses import multi_scale_mel_loss, vqvae_loss
from distill.models.vqvae import ConvVQVAE, QuantizerWrapper, VQDecoder, VQEncoder
from distill.training.config import VQVAEConfig, get_adaptive_vqvae_config

# ---------------------------------------------------------------------------
# v1.1 Prior exports -- added in Phase 14
# ---------------------------------------------------------------------------

from distill.models.persistence import save_prior_to_model
from distill.models.prior import (
    CodePrior,
    extract_code_sequences,
    flatten_codes,
    sample_code_sequence,
    unflatten_codes,
)

__all__ = [
    # vae.py (v1.0)
    "ConvVAE",
    "ConvEncoder",
    "ConvDecoder",
    # losses.py (v1.0)
    "vae_loss",
    "get_kl_weight",
    "compute_kl_divergence",
    # persistence.py (v1.0)
    "ModelMetadata",
    "LoadedModel",
    "save_model",
    "load_model",
    "delete_model",
    "save_model_from_checkpoint",
    "MODEL_FILE_EXTENSION",
    "SAVED_MODEL_VERSION",
    # persistence.py (v1.1 -- v2 format)
    "SAVED_MODEL_VERSION_V2",
    "LoadedVQModel",
    "save_model_v2",
    "load_model_v2",
    # vqvae.py (v1.1)
    "ConvVQVAE",
    "VQEncoder",
    "VQDecoder",
    "QuantizerWrapper",
    # losses.py (v1.1)
    "vqvae_loss",
    "multi_scale_mel_loss",
    # config (v1.1)
    "VQVAEConfig",
    "get_adaptive_vqvae_config",
    # prior.py (v1.1)
    "CodePrior",
    "flatten_codes",
    "unflatten_codes",
    "extract_code_sequences",
    "sample_code_sequence",
    # persistence.py (v1.1 prior)
    "save_prior_to_model",
]
