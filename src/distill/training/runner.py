"""Training runner with background thread management.

Wraps the core training loop in a daemon thread with clean cancellation
via ``threading.Event``.  Cancel triggers an immediate checkpoint save
inside the training loop (no waiting for epoch boundaries).

Usage::

    runner = TrainingRunner()
    runner.start(config, file_paths, output_dir, device, callback=on_event)

    # ... later ...
    runner.cancel()          # request graceful stop
    runner.wait(timeout=30)  # wait for thread to finish

    if runner.last_error:
        print(f"Training failed: {runner.last_error}")
    else:
        result = runner.result  # {model, metrics_history, ...}

Design notes:
- Daemon thread: dies automatically if main process exits.
- ``_is_running`` is set before thread start and cleared in finally block.
- ``cancel()`` only sets the event -- does NOT join the thread (let it
  finish its checkpoint save operation).
- ``resume()`` is identical to ``start()`` but passes a checkpoint path
  to ``train()`` for state restoration.
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from distill.training.config import TrainingConfig
    from distill.training.metrics import MetricsCallback

logger = logging.getLogger(__name__)


class TrainingRunner:
    """Manages VAE training in a background thread.

    Provides start/cancel/resume/wait operations with thread-safe
    state tracking.  Only one training run is allowed at a time.

    Attributes
    ----------
    is_running:
        Whether training is currently in progress.
    last_error:
        The exception from the most recent failed run, or ``None``.
    result:
        The result dict from the most recent successful run, or ``None``.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._is_running: bool = False
        self._last_error: Exception | None = None
        self._result: dict | None = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def start(
        self,
        config: "TrainingConfig",
        file_paths: list[Path],
        output_dir: Path,
        device: "torch.device",
        callback: "MetricsCallback | None" = None,
        models_dir: "Path | None" = None,
        dataset_name: str = "",
        model_name: str = "",
    ) -> None:
        """Start training in a background thread.

        Parameters
        ----------
        config:
            Training configuration.
        file_paths:
            Paths to audio files.
        output_dir:
            Root output directory.
        device:
            Target device.
        callback:
            Optional metrics event subscriber.
        models_dir:
            Directory to save the .distill model file into.
        dataset_name:
            Name of the dataset (used for model metadata).
        model_name:
            User-specified model name (used for saved model name).

        Raises
        ------
        RuntimeError
            If training is already running.
        """
        if self._is_running:
            raise RuntimeError("Training is already running. Cancel first.")

        self._cancel_event.clear()
        self._last_error = None
        self._result = None
        self._is_running = True

        self._thread = threading.Thread(
            target=self._run_training,
            args=(config, file_paths, output_dir, device, callback, None,
                  models_dir, dataset_name, model_name),
            daemon=True,
            name="training-runner",
        )
        self._thread.start()

    def cancel(self) -> None:
        """Request training cancellation.

        Sets the cancel event which the training loop checks at each
        step and epoch boundary.  The loop saves an immediate checkpoint
        before stopping.  Does NOT join the thread.
        """
        self._cancel_event.set()
        logger.info("Training cancellation requested")

    def resume(
        self,
        config: "TrainingConfig",
        file_paths: list[Path],
        output_dir: Path,
        device: "torch.device",
        checkpoint_path: Path,
        callback: "MetricsCallback | None" = None,
        models_dir: "Path | None" = None,
        dataset_name: str = "",
        model_name: str = "",
    ) -> None:
        """Resume training from a checkpoint.

        Same as ``start()`` but passes ``resume_checkpoint`` to the
        training loop for state restoration.

        Parameters
        ----------
        config:
            Training configuration.
        file_paths:
            Paths to audio files.
        output_dir:
            Root output directory.
        device:
            Target device.
        checkpoint_path:
            Path to the ``.pt`` checkpoint file to resume from.
        callback:
            Optional metrics event subscriber.
        models_dir:
            Directory to save the .distill model file into.
        dataset_name:
            Name of the dataset (used for model metadata).
        model_name:
            User-specified model name (used for saved model name).

        Raises
        ------
        RuntimeError
            If training is already running.
        """
        if self._is_running:
            raise RuntimeError("Training is already running. Cancel first.")

        self._cancel_event.clear()
        self._last_error = None
        self._result = None
        self._is_running = True

        self._thread = threading.Thread(
            target=self._run_training,
            args=(config, file_paths, output_dir, device, callback, checkpoint_path,
                  models_dir, dataset_name, model_name),
            daemon=True,
            name="training-runner-resume",
        )
        self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the training thread to finish.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  ``None`` means wait indefinitely.

        Returns
        -------
        bool
            ``True`` if the thread completed (or was never started).
        """
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    @property
    def is_running(self) -> bool:
        """Whether training is currently in progress."""
        return self._is_running

    @property
    def last_error(self) -> Exception | None:
        """The exception from the most recent failed run, or ``None``."""
        return self._last_error

    @property
    def result(self) -> dict | None:
        """The result dict from the most recent successful run, or ``None``."""
        return self._result

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _run_training(
        self,
        config: "TrainingConfig",
        file_paths: list[Path],
        output_dir: Path,
        device: "torch.device",
        callback: "MetricsCallback | None",
        resume_checkpoint: Path | None,
        models_dir: "Path | None" = None,
        dataset_name: str = "",
        model_name: str = "",
    ) -> None:
        """Thread target: run the training loop with error handling."""
        from distill.training.loop import train

        try:
            result = train(
                config=config,
                file_paths=file_paths,
                output_dir=output_dir,
                device=device,
                callback=callback,
                cancel_event=self._cancel_event,
                resume_checkpoint=resume_checkpoint,
                models_dir=models_dir,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            self._result = result
        except Exception as exc:
            self._last_error = exc
            logger.error(
                "Training failed: %s\n%s",
                exc, traceback.format_exc(),
            )
        finally:
            self._is_running = False
