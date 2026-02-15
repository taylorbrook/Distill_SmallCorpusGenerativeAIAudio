"""Training metrics collection and history with callback-based events.

Defines typed event dataclasses emitted by the training loop and consumed
by the dashboard (Phase 6).  The callback pattern decouples training from
display: the loop emits ``StepMetrics`` / ``EpochMetrics`` / ``PreviewEvent``
and any subscriber can react.

:class:`MetricsHistory` accumulates metrics across a training run with
serialisation for checkpoint inclusion, loss curve extraction for
plotting, and overfitting detection.

Design notes:
- **Pure Python** -- no torch dependency (dataclasses + typing only).
- ``from __future__ import annotations`` for modern type syntax.
- All fields typed and documented for IDE autocompletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

# ---------------------------------------------------------------------------
# Event Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    """Emitted every training step (mini-batch).

    Attributes
    ----------
    epoch:
        Current epoch (0-indexed).
    step:
        Current step within the epoch.
    total_steps:
        Total steps in the epoch.
    train_loss:
        Combined loss for this step.
    recon_loss:
        Reconstruction loss component.
    kl_loss:
        KL divergence loss component (weighted).
    kl_weight:
        Current KL weight from warmup schedule.
    learning_rate:
        Current optimizer learning rate.
    step_time_s:
        Wall-clock time for this step in seconds.
    """

    epoch: int
    step: int
    total_steps: int
    train_loss: float
    recon_loss: float
    kl_loss: float
    kl_weight: float
    learning_rate: float
    step_time_s: float


@dataclass
class EpochMetrics:
    """Emitted every epoch after validation.

    Attributes
    ----------
    epoch:
        Current epoch (0-indexed).
    total_epochs:
        Total planned epochs.
    train_loss:
        Average training loss over the epoch.
    val_loss:
        Average validation loss.
    val_recon_loss:
        Validation reconstruction loss component.
    val_kl_loss:
        Validation KL divergence loss component (weighted).
    kl_divergence:
        Raw (unweighted) KL divergence for posterior collapse monitoring.
    overfitting_gap:
        ``(val_loss - train_loss) / train_loss`` -- positive means
        validation is worse.
    learning_rate:
        Current optimizer learning rate.
    eta_seconds:
        Estimated time remaining in seconds.
    elapsed_seconds:
        Total wall-clock time elapsed since training start.
    """

    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_recon_loss: float
    val_kl_loss: float
    kl_divergence: float
    overfitting_gap: float
    learning_rate: float
    eta_seconds: float
    elapsed_seconds: float


@dataclass
class PreviewEvent:
    """Emitted when an audio preview is generated during training.

    Attributes
    ----------
    epoch:
        Epoch at which the preview was generated.
    audio_path:
        Path to the generated audio file.
    sample_rate:
        Sample rate of the generated audio.
    """

    epoch: int
    audio_path: str
    sample_rate: int


@dataclass
class TrainingCompleteEvent:
    """Emitted when training finishes (normally or early-stopped).

    Attributes
    ----------
    total_epochs:
        Number of epochs completed.
    total_time_s:
        Total wall-clock training time in seconds.
    final_train_loss:
        Training loss at the final epoch.
    final_val_loss:
        Validation loss at the final epoch.
    best_val_loss:
        Lowest validation loss observed.
    best_epoch:
        Epoch with the lowest validation loss.
    """

    total_epochs: int
    total_time_s: float
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int


# ---------------------------------------------------------------------------
# Callback Type
# ---------------------------------------------------------------------------

MetricsCallback = Callable[
    [Union[StepMetrics, EpochMetrics, PreviewEvent, TrainingCompleteEvent]],
    None,
]
"""Type alias for metrics event subscribers.

A callback receives one of the four event types and processes it
(e.g., update UI, write log, etc.).
"""


# ---------------------------------------------------------------------------
# Metrics History
# ---------------------------------------------------------------------------


@dataclass
class MetricsHistory:
    """Accumulates training metrics for a full run.

    Stores step-level and epoch-level metrics, provides loss curve
    extraction for plotting, ETA estimation, overfitting detection,
    and serialisation for checkpoint inclusion.
    """

    step_metrics: list[StepMetrics] = field(default_factory=list)
    epoch_metrics: list[EpochMetrics] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------

    def add_step(self, metrics: StepMetrics) -> None:
        """Record a step-level metrics snapshot."""
        self.step_metrics.append(metrics)

    def add_epoch(self, metrics: EpochMetrics) -> None:
        """Record an epoch-level metrics snapshot."""
        self.epoch_metrics.append(metrics)

    # -----------------------------------------------------------------
    # Loss Curves
    # -----------------------------------------------------------------

    def get_loss_curves(self) -> dict[str, list[float]]:
        """Extract named loss curves for plotting.

        Returns
        -------
        dict[str, list[float]]
            Keys: ``train_losses``, ``val_losses``, ``kl_losses``,
            ``recon_losses``.  Each list has one entry per epoch.
        """
        return {
            "kl_losses": [m.val_kl_loss for m in self.epoch_metrics],
            "recon_losses": [m.val_recon_loss for m in self.epoch_metrics],
            "train_losses": [m.train_loss for m in self.epoch_metrics],
            "val_losses": [m.val_loss for m in self.epoch_metrics],
        }

    # -----------------------------------------------------------------
    # Best Epoch
    # -----------------------------------------------------------------

    def get_best_epoch(self) -> int:
        """Return the epoch index with the lowest validation loss.

        Returns
        -------
        int
            0-indexed epoch number, or 0 if no epochs recorded.
        """
        if not self.epoch_metrics:
            return 0
        best = min(self.epoch_metrics, key=lambda m: m.val_loss)
        return best.epoch

    # -----------------------------------------------------------------
    # ETA
    # -----------------------------------------------------------------

    def compute_eta(self, current_epoch: int, total_epochs: int) -> float:
        """Estimate remaining training time based on average epoch duration.

        Parameters
        ----------
        current_epoch:
            Current epoch (0-indexed).
        total_epochs:
            Total planned epochs.

        Returns
        -------
        float
            Estimated remaining seconds.  Returns 0.0 if no epoch data.
        """
        if not self.epoch_metrics or current_epoch <= 0:
            return 0.0

        latest = self.epoch_metrics[-1]
        avg_time_per_epoch = latest.elapsed_seconds / (current_epoch + 1)
        remaining_epochs = total_epochs - current_epoch - 1
        return max(0.0, avg_time_per_epoch * remaining_epochs)

    # -----------------------------------------------------------------
    # Overfitting Detection
    # -----------------------------------------------------------------

    def is_overfitting(self, threshold: float = 0.2) -> bool:
        """Check if the latest epoch shows overfitting.

        Parameters
        ----------
        threshold:
            Maximum acceptable ``overfitting_gap``.  Default 0.2 (20%).

        Returns
        -------
        bool
            ``True`` if the most recent ``overfitting_gap`` exceeds
            *threshold*.
        """
        if not self.epoch_metrics:
            return False
        return self.epoch_metrics[-1].overfitting_gap > threshold

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Nested structure with ``step_metrics`` and ``epoch_metrics``
            lists of plain dicts.
        """
        return {
            "step_metrics": [
                {
                    "epoch": m.epoch,
                    "step": m.step,
                    "total_steps": m.total_steps,
                    "train_loss": m.train_loss,
                    "recon_loss": m.recon_loss,
                    "kl_loss": m.kl_loss,
                    "kl_weight": m.kl_weight,
                    "learning_rate": m.learning_rate,
                    "step_time_s": m.step_time_s,
                }
                for m in self.step_metrics
            ],
            "epoch_metrics": [
                {
                    "epoch": m.epoch,
                    "total_epochs": m.total_epochs,
                    "train_loss": m.train_loss,
                    "val_loss": m.val_loss,
                    "val_recon_loss": m.val_recon_loss,
                    "val_kl_loss": m.val_kl_loss,
                    "kl_divergence": m.kl_divergence,
                    "overfitting_gap": m.overfitting_gap,
                    "learning_rate": m.learning_rate,
                    "eta_seconds": m.eta_seconds,
                    "elapsed_seconds": m.elapsed_seconds,
                }
                for m in self.epoch_metrics
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MetricsHistory:
        """Restore a :class:`MetricsHistory` from a serialised dictionary.

        Parameters
        ----------
        data:
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        MetricsHistory
            Restored instance with all metrics.
        """
        history = cls()
        for d in data.get("step_metrics", []):
            history.add_step(StepMetrics(**d))
        for d in data.get("epoch_metrics", []):
            history.add_epoch(EpochMetrics(**d))
        return history
