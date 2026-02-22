"""Matplotlib figure builder for training loss curves.

Renders train and validation loss curves on a single axes for
display in ``gr.Plot``.  Called from the Timer tick handler in
the Train tab (never from the training thread).

Supports both v1.0 (EpochMetrics: train + val) and v1.1
(VQEpochMetrics: train + val + commitment) metric types.
VQ-VAE metrics are detected via duck-typing (``hasattr``
check for ``val_commit_loss``) to avoid circular import risk.

Uses ``matplotlib.use("Agg")`` for headless, thread-safe rendering
(project convention).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from distill.training.metrics import EpochMetrics, VQEpochMetrics


def build_loss_chart(
    epoch_metrics: list[Union[EpochMetrics, VQEpochMetrics]],
) -> Figure | None:
    """Build a matplotlib figure showing training loss curves.

    Detects VQ-VAE metrics by checking for ``val_commit_loss`` attribute
    and adds a third commitment loss line (green, dashed) when present.

    Parameters
    ----------
    epoch_metrics:
        List of :class:`EpochMetrics` or :class:`VQEpochMetrics`
        dataclasses accumulated during training.  May be empty
        (returns ``None``).

    Returns
    -------
    Figure | None
        A matplotlib Figure ready for ``gr.Plot``, or ``None`` if
        *epoch_metrics* is empty.
    """
    if not epoch_metrics:
        return None

    # Close any previous figures to prevent matplotlib memory leaks
    plt.close("all")

    epochs = [m.epoch + 1 for m in epoch_metrics]
    train_losses = [m.train_loss for m in epoch_metrics]
    val_losses = [m.val_loss for m in epoch_metrics]

    # Duck-type check: VQ-VAE metrics have val_commit_loss
    is_vqvae = hasattr(epoch_metrics[0], "val_commit_loss")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5, color="#1f77b4")
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=1.5, color="#ff7f0e")

    if is_vqvae:
        commit_losses = [m.val_commit_loss for m in epoch_metrics]
        ax.plot(
            epochs,
            commit_losses,
            label="Commitment Loss",
            linewidth=1.5,
            color="#2ca02c",
            linestyle="--",
        )
        ax.set_title("VQ-VAE Training Progress")
    else:
        ax.set_title("Training Progress")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()

    return fig
