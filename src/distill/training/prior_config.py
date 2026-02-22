"""Prior model and training configuration with dataset-adaptive scaling.

Provides :class:`PriorConfig` (all user-facing hyperparameters for
autoregressive prior training) and :func:`get_adaptive_prior_config`
(auto-scales model size and regularisation to prevent memorisation on
small datasets).

Follows the same 3-tier adaptive pattern established by
:func:`~distill.training.config.get_adaptive_vqvae_config`:
smaller datasets get smaller models with heavier regularisation.

Design notes:

- Pure Python -- no torch dependency (matches
  :mod:`distill.training.config` pattern).
- ``from __future__ import annotations`` for modern type syntax.
- All fields documented for Gradio UI auto-generation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriorConfig:
    """Configuration for the autoregressive prior model and training.

    All user-facing parameters for the CodePrior transformer.  Defaults
    match the ``BALANCED`` tier (21-100 files).

    Attributes
    ----------
    hidden_size:
        Transformer hidden dimension.
    num_layers:
        Number of transformer encoder layers.
    num_heads:
        Number of attention heads.
    dropout:
        Dropout probability.
    max_epochs:
        Maximum training epochs.
    learning_rate:
        AdamW learning rate.
    weight_decay:
        AdamW weight decay coefficient.
    gradient_clip_norm:
        Maximum gradient L2 norm for clipping.
    batch_size:
        Mini-batch size for code sequence training.
    val_fraction:
        Fraction of code sequences held out for validation.
    device:
        Target device: ``"auto"`` resolves via hardware detection,
        or explicit ``"cpu"`` / ``"cuda"`` / ``"mps"``.
    """

    # Model architecture
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    # Training
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    batch_size: int = 32
    val_fraction: float = 0.2

    # Device
    device: str = "auto"


def get_adaptive_prior_config(file_count: int) -> PriorConfig:
    """Build a :class:`PriorConfig` adapted to the dataset size.

    Scales model size and regularisation to prevent memorisation on
    small datasets.  Follows the same 3-tier pattern as
    :func:`~distill.training.config.get_adaptive_vqvae_config`.

    ======== =========== ========== ========= ======= ====== =============
    Files    hidden_size num_layers num_heads dropout epochs learning_rate
    ======== =========== ========== ========= ======= ====== =============
    <= 20    128         2          4         0.3     50     3e-4
    21-100   256         4          4         0.2     100    1e-3
    > 100    512         6          8         0.1     150    1e-3
    ======== =========== ========== ========= ======= ====== =============

    ``val_fraction`` scales inversely with dataset size: 0.5 for < 10,
    0.3 for < 50, 0.2 for < 200, 0.1 for >= 200.

    ``batch_size`` is capped at ``min(32, max(1, file_count * 5 // 4))``
    (same heuristic as :func:`get_adaptive_vqvae_config`).

    Parameters
    ----------
    file_count:
        Number of audio files in the dataset.

    Returns
    -------
    PriorConfig
        Fully configured for the given dataset size.
    """
    # 1. Tier selection
    if file_count <= 20:
        hidden_size = 128
        num_layers = 2
        num_heads = 4
        dropout = 0.3
        max_epochs = 50
        learning_rate = 3e-4
    elif file_count <= 100:
        hidden_size = 256
        num_layers = 4
        num_heads = 4
        dropout = 0.2
        max_epochs = 100
        learning_rate = 1e-3
    else:  # > 100
        hidden_size = 512
        num_layers = 6
        num_heads = 8
        dropout = 0.1
        max_epochs = 150
        learning_rate = 1e-3

    # 2. Adaptive validation fraction (same as VQVAEConfig)
    if file_count < 10:
        val_fraction = 0.5
    elif file_count < 50:
        val_fraction = 0.3
    elif file_count < 200:
        val_fraction = 0.2
    else:
        val_fraction = 0.1

    # 3. Adaptive batch size (same heuristic as VQVAEConfig)
    batch_size = min(32, max(1, file_count * 5 // 4))

    return PriorConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        batch_size=batch_size,
        val_fraction=val_fraction,
    )
