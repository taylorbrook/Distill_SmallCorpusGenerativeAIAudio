"""Matplotlib figure builder for training loss curves.

Renders train and validation loss curves on a single axes for
display in ``gr.Plot``.  Called from the Timer tick handler in
the Train tab (never from the training thread).

Uses ``matplotlib.use("Agg")`` for headless, thread-safe rendering
(project convention).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from distill.training.metrics import EpochMetrics


def build_loss_chart(epoch_metrics: list[EpochMetrics]) -> Figure | None:
    """Build a matplotlib figure showing training loss curves.

    Parameters
    ----------
    epoch_metrics:
        List of :class:`EpochMetrics` dataclasses accumulated during
        training.  May be empty (returns ``None``).

    Returns
    -------
    Figure | None
        A matplotlib Figure ready for ``gr.Plot``, or ``None`` if
        *epoch_metrics* is empty.
    """
    if not epoch_metrics:
        return None

    epochs = [m.epoch + 1 for m in epoch_metrics]
    train_losses = [m.train_loss for m in epoch_metrics]
    val_losses = [m.val_loss for m in epoch_metrics]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    plt.tight_layout()

    return fig
