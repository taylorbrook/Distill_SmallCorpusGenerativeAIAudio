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
    from distill.vocoder.hifigan.trainer import VocoderEpochMetrics


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


def build_vocoder_loss_chart(
    vocoder_metrics: list[VocoderEpochMetrics],
) -> Figure | None:
    """Build a dual-axis matplotlib figure for GAN training progress.

    Shows generator and discriminator loss curves on separate y-axes,
    with mel reconstruction loss as a dashed line on the generator axis.

    Parameters
    ----------
    vocoder_metrics:
        List of :class:`VocoderEpochMetrics` dataclasses accumulated
        during vocoder training.  May be empty (returns ``None``).

    Returns
    -------
    Figure | None
        A matplotlib Figure ready for ``gr.Plot``, or ``None`` if
        *vocoder_metrics* is empty.
    """
    if not vocoder_metrics:
        return None

    epochs = [m.epoch + 1 for m in vocoder_metrics]
    gen_losses = [m.gen_loss for m in vocoder_metrics]
    disc_losses = [m.disc_loss for m in vocoder_metrics]
    mel_losses = [m.mel_loss for m in vocoder_metrics]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    # Generator losses on left axis
    ax1.plot(
        epochs, gen_losses, label="Generator Loss", color="#2196F3", linewidth=1.5
    )
    ax1.plot(
        epochs,
        mel_losses,
        label="Mel Loss",
        color="#2196F3",
        linewidth=1,
        linestyle="--",
        alpha=0.7,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Generator Loss", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    # Discriminator loss on right axis
    ax2.plot(
        epochs, disc_losses, label="Discriminator Loss", color="#F44336", linewidth=1.5
    )
    ax2.set_ylabel("Discriminator Loss", color="#F44336")
    ax2.tick_params(axis="y", labelcolor="#F44336")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title("Vocoder Training Progress")
    fig.tight_layout()
    return fig
