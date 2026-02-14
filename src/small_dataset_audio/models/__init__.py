"""Neural network model definitions.

Public API re-exports from :mod:`models.vae`, :mod:`models.losses`,
and :mod:`models.persistence`.
"""

from small_dataset_audio.models.losses import (
    compute_kl_divergence,
    get_kl_weight,
    vae_loss,
)
from small_dataset_audio.models.persistence import (
    MODEL_FILE_EXTENSION,
    LoadedModel,
    ModelMetadata,
    SAVED_MODEL_VERSION,
    delete_model,
    load_model,
    save_model,
    save_model_from_checkpoint,
)
from small_dataset_audio.models.vae import ConvDecoder, ConvEncoder, ConvVAE

__all__ = [
    # vae.py
    "ConvVAE",
    "ConvEncoder",
    "ConvDecoder",
    # losses.py
    "vae_loss",
    "get_kl_weight",
    "compute_kl_divergence",
    # persistence.py
    "ModelMetadata",
    "LoadedModel",
    "save_model",
    "load_model",
    "delete_model",
    "save_model_from_checkpoint",
    "MODEL_FILE_EXTENSION",
    "SAVED_MODEL_VERSION",
]
