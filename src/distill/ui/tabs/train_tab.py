"""Train tab: configuration, live loss chart, previews, and cancel/resume.

Surfaces the :class:`TrainingRunner` (Phase 3) with a Timer-polled
dashboard.  Training runs in a daemon thread; the Timer reads the
shared ``app_state.metrics_buffer`` every 2 seconds and pushes updates
to the loss chart, stats panel, and preview audio slots.

Key design points:
- One ``TrainingRunner`` lives on ``app_state``; Start/Resume refuse to
  launch a second run, and a reloaded page reattaches to a running one.
- Each run writes checkpoints and previews to its own directory under
  ``app_state.training_dir`` (never inside the dataset, which is scanned
  for audio).  Resume continues the newest run from its latest checkpoint.
- ``gr.Timer(active=False)`` -- activated only when training starts.
- 20 pre-created ``gr.Audio`` slots (hidden) revealed as previews arrive.
- Preset dropdown auto-populates epochs/LR/advanced from
  ``get_adaptive_config`` + ``OverfittingPreset`` mapping.
- Empty state guidance when no dataset is loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from distill.training.checkpoint import get_latest_checkpoint, new_run_dir
from distill.training.config import (
    OverfittingPreset,
    TrainingConfig,
    get_adaptive_config,
)
from distill.training.metrics import (
    EpochMetrics,
    PreviewEvent,
    TrainingCompleteEvent,
)
from distill.training.runner import TrainingRunner
from distill.ui.components.guided_nav import get_empty_state_message
from distill.ui.components.loss_chart import build_loss_chart
from distill.ui.state import app_state, reset_metrics_buffer

logger = logging.getLogger(__name__)

# Number of pre-created audio preview slots
_MAX_PREVIEW_SLOTS = 20

# Preset name -> OverfittingPreset enum mapping
_PRESET_MAP: dict[str, OverfittingPreset] = {
    "Conservative": OverfittingPreset.CONSERVATIVE,
    "Balanced": OverfittingPreset.BALANCED,
    "Aggressive": OverfittingPreset.AGGRESSIVE,
}

# Preset -> default parameter values (subset of _PRESET_PARAMS in config.py)
_PRESET_DEFAULTS: dict[str, dict[str, float | int]] = {
    "Conservative": {
        "max_epochs": 100,
        "learning_rate": 5e-4,
        "dropout": 0.4,
        "weight_decay": 0.05,
        "kl_weight_max": 0.005,
    },
    "Balanced": {
        "max_epochs": 200,
        "learning_rate": 1e-3,
        "dropout": 0.2,
        "weight_decay": 0.01,
        "kl_weight_max": 0.01,
    },
    "Aggressive": {
        "max_epochs": 500,
        "learning_rate": 2e-3,
        "dropout": 0.1,
        "weight_decay": 0.001,
        "kl_weight_max": 0.02,
    },
}


# -------------------------------------------------------------------
# Training callback (runs in training thread, stores events)
# -------------------------------------------------------------------


def _training_callback(event: object) -> None:
    """MetricsCallback that stores events for the Timer to read.

    Thread-safe: ``list.append`` is atomic under CPython GIL.
    """
    if isinstance(event, EpochMetrics):
        app_state.metrics_buffer["epoch_metrics"].append(event)
    elif isinstance(event, PreviewEvent):
        app_state.metrics_buffer["previews"].append(event)
    elif isinstance(event, TrainingCompleteEvent):
        app_state.metrics_buffer["complete"] = True


# -------------------------------------------------------------------
# Helper: format ETA
# -------------------------------------------------------------------


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-friendly ETA string."""
    if seconds <= 0:
        return "done"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


# -------------------------------------------------------------------
# Helpers: run directories and the shared runner
# -------------------------------------------------------------------


def _dataset_training_dir() -> Path:
    """Parent directory of the current dataset's training runs.

    Lives under ``app_state.training_dir``, not under ``datasets_dir``:
    dataset directories are scanned for audio, so preview WAVs written
    there would become training data.
    """
    ds = app_state.current_dataset
    name = ds.name if ds else "default"
    return app_state.training_dir / name


def _latest_checkpoint() -> Path | None:
    """The newest checkpoint across the current dataset's runs.

    That is where training last left off, so it is what Resume continues
    (a cancel checkpoint is never the best-val-loss one).
    """
    root = _dataset_training_dir()
    if not root.is_dir():
        return None
    candidates = [
        get_latest_checkpoint(run / "checkpoints")
        for run in root.iterdir()
        if run.is_dir()
    ]
    candidates = [c for c in candidates if c is not None]
    return max(candidates, key=lambda c: c.stat().st_mtime, default=None)


def _get_runner() -> TrainingRunner:
    """The app's single training runner (created on first use)."""
    if app_state.training_runner is None:
        app_state.training_runner = TrainingRunner()
    return app_state.training_runner


def _training_in_progress() -> bool:
    runner = app_state.training_runner
    return runner is not None and runner.is_running


def _build_config(
    max_epochs: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    kl_weight_max: float,
    preset_name: str,
) -> TrainingConfig:
    """Adaptive config for the current dataset with the UI's overrides."""
    ds = app_state.current_dataset
    file_count = len(ds.valid_files) if hasattr(ds, "valid_files") else 10
    config = get_adaptive_config(file_count)
    config.max_epochs = int(max_epochs)
    config.learning_rate = float(learning_rate)
    config.regularization.dropout = float(dropout)
    config.regularization.weight_decay = float(weight_decay)
    config.kl_weight_max = float(kl_weight_max)
    if preset_name in _PRESET_MAP:
        config.preset = _PRESET_MAP[preset_name]
    return config


def _control_updates(running: bool) -> list:
    """Updates for (timer, start_btn, cancel_btn, resume_btn)."""
    if running:
        cancelled = app_state.metrics_buffer.get("cancelled", False)
        return [
            gr.Timer(active=True),
            gr.update(interactive=False),
            gr.update(interactive=not cancelled),
            gr.update(interactive=False),
        ]
    return [
        gr.Timer(active=False),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True, visible=_latest_checkpoint() is not None),
    ]


def _no_preview_updates() -> list:
    return [gr.update() for _ in range(_MAX_PREVIEW_SLOTS)]


def _already_running_outputs() -> list:
    """Outputs for a Start/Resume click while a run is in progress."""
    return [
        gr.update(),
        "**Training is already running.** Cancel it before starting another run.",
        *_control_updates(running=True),
    ] + _no_preview_updates()


# -------------------------------------------------------------------
# Tab builder
# -------------------------------------------------------------------


def build_train_tab() -> dict:
    """Build the Train tab UI within the current Blocks context.

    Layout:
    - Empty state message (shown when no dataset)
    - Training config with preset selector and advanced accordion
    - Start / Cancel / Resume buttons
    - Loss chart (gr.Plot) + stats panel (gr.Markdown)
    - Timer for polling
    - 20 preview audio slots

    Returns:
        Dict of component references for cross-tab wiring.
    """
    # Empty state
    empty_state = gr.Markdown(
        value=get_empty_state_message("train"),
        visible=True,
    )

    # Main training UI container (hidden until dataset loaded)
    with gr.Column(visible=False) as train_ui:
        gr.Markdown("## Train")

        # -- Model name -----------------------------------------------------
        model_name_input = gr.Textbox(
            label="Model Name",
            placeholder="e.g. Ambient Pads v1",
            value="",
        )

        # -- Config section -----------------------------------------------
        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=["Conservative", "Balanced", "Aggressive"],
                value="Balanced",
                label="Training Preset",
                scale=1,
            )
            max_epochs_num = gr.Number(
                value=200,
                label="Max Epochs",
                precision=0,
                scale=1,
            )
            learning_rate_num = gr.Number(
                value=1e-3,
                label="Learning Rate",
                precision=6,
                scale=1,
            )

        with gr.Accordion("Advanced Training Settings", open=False):
            dropout_slider = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                step=0.05,
                value=0.2,
                label="Dropout",
            )
            weight_decay_slider = gr.Slider(
                minimum=0.0,
                maximum=0.1,
                step=0.001,
                value=0.01,
                label="Weight Decay",
            )
            kl_weight_slider = gr.Slider(
                minimum=0.0,
                maximum=0.1,
                step=0.001,
                value=0.01,
                label="KL Weight (Beta)",
            )

        # -- Control buttons ----------------------------------------------
        with gr.Row():
            start_btn = gr.Button("Train", variant="primary")
            cancel_btn = gr.Button("Cancel", variant="stop", interactive=False)
            resume_btn = gr.Button("Resume Training", visible=False)

        # -- Training dashboard -------------------------------------------
        loss_plot = gr.Plot(label="Loss Curves")
        stats_md = gr.Markdown(value="Waiting for training to start...")

        # Timer (inactive until training starts)
        timer = gr.Timer(value=2, active=False)

        # -- Preview audio section ----------------------------------------
        gr.Markdown("### Audio Previews")
        preview_audios: list[gr.Audio] = []
        for i in range(_MAX_PREVIEW_SLOTS):
            audio = gr.Audio(
                label=f"Preview {i + 1}",
                visible=False,
                interactive=False,
            )
            preview_audios.append(audio)

    # -------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------

    def _on_preset_change(preset_name: str) -> tuple:
        """Auto-populate config fields from preset defaults."""
        defaults = _PRESET_DEFAULTS.get(preset_name, _PRESET_DEFAULTS["Balanced"])
        return (
            defaults["max_epochs"],
            defaults["learning_rate"],
            defaults["dropout"],
            defaults["weight_decay"],
            defaults["kl_weight_max"],
        )

    preset_dd.change(
        fn=_on_preset_change,
        inputs=[preset_dd],
        outputs=[
            max_epochs_num,
            learning_rate_num,
            dropout_slider,
            weight_decay_slider,
            kl_weight_slider,
        ],
    )

    def _check_dataset_ready() -> tuple:
        """Check if a dataset is loaded and toggle empty/main UI."""
        has_ds = app_state.current_dataset is not None
        # Also check for existing checkpoints to show Resume button
        show_resume = has_ds and _latest_checkpoint() is not None
        return (
            gr.update(visible=not has_ds),   # empty_state
            gr.update(visible=has_ds),        # train_ui
            gr.update(visible=show_resume),   # resume_btn
        )

    def _reattach_to_run() -> list:
        """Page load: if a run is in progress, resume polling it.

        Without this a reloaded page shows an enabled Start button and an
        idle dashboard while the previous run is still training.
        """
        if not _training_in_progress():
            return [gr.update() for _ in range(5)]
        return [
            "Training in progress...",
            *_control_updates(running=True),
        ]

    # Poll dataset readiness when the tab renders via a load event
    app_context = gr.context.Context.root_block
    if app_context is not None:
        app_context.load(
            fn=_check_dataset_ready,
            inputs=None,
            outputs=[empty_state, train_ui, resume_btn],
        )
        app_context.load(
            fn=_reattach_to_run,
            inputs=None,
            outputs=[stats_md, timer, start_btn, cancel_btn, resume_btn],
        )

    def _start_training(
        model_name: str,
        max_epochs: int,
        learning_rate: float,
        dropout: float,
        weight_decay: float,
        kl_weight_max: float,
        preset_name: str,
    ) -> list:
        """Start training in a background thread."""
        ds = app_state.current_dataset
        if ds is None:
            return [
                gr.update(),          # loss_plot
                "No dataset loaded. Go to the Data tab first.",  # stats_md
                *_control_updates(running=False),
            ] + _no_preview_updates()

        runner = _get_runner()
        if runner.is_running:
            return _already_running_outputs()

        config = _build_config(
            max_epochs, learning_rate, dropout, weight_decay,
            kl_weight_max, preset_name,
        )
        reset_metrics_buffer()
        output_dir = new_run_dir(_dataset_training_dir())
        file_paths = list(ds.valid_files) if hasattr(ds, "valid_files") else []

        try:
            runner.start(
                config=config,
                file_paths=file_paths,
                output_dir=output_dir,
                device=app_state.device,
                callback=_training_callback,
                models_dir=app_state.models_dir,
                dataset_name=ds.name if ds else "untitled",
                model_name=model_name.strip() if model_name else "",
            )
        except RuntimeError:
            # Another Start/Resume click won the race
            return _already_running_outputs()
        app_state.training_active = True

        # Return: activate timer, disable Start, enable Cancel, hide previews
        return [
            gr.update(value=None),           # loss_plot (clear)
            "Training started...",           # stats_md
            *_control_updates(running=True),
        ] + [gr.update(visible=False, value=None) for _ in range(_MAX_PREVIEW_SLOTS)]

    _train_inputs = [
        model_name_input,
        max_epochs_num,
        learning_rate_num,
        dropout_slider,
        weight_decay_slider,
        kl_weight_slider,
        preset_dd,
    ]
    _run_outputs = [
        loss_plot, stats_md, timer, start_btn, cancel_btn, resume_btn,
    ] + preview_audios

    start_btn.click(
        fn=_start_training,
        inputs=_train_inputs,
        outputs=_run_outputs,
    )

    def _cancel_training() -> list:
        """Request cancellation; the timer keeps polling until the run ends.

        The training thread still saves a checkpoint and (after at least
        one epoch) runs the latent analysis and saves the model, so Start
        stays disabled until ``_poll_training`` sees the completion event.
        """
        if app_state.training_runner is not None:
            app_state.training_runner.cancel()
        app_state.metrics_buffer["cancelled"] = True

        return [
            gr.update(),                    # timer (keep polling)
            gr.update(interactive=False),   # start_btn
            gr.update(interactive=False),   # cancel_btn
            "Cancelling -- saving checkpoint and model...",  # stats_md
        ]

    cancel_btn.click(
        fn=_cancel_training,
        inputs=None,
        outputs=[timer, start_btn, cancel_btn, stats_md],
    )

    def _resume_training(
        model_name: str,
        max_epochs: int,
        learning_rate: float,
        dropout: float,
        weight_decay: float,
        kl_weight_max: float,
        preset_name: str,
    ) -> list:
        """Continue the newest run from its latest checkpoint."""
        ds = app_state.current_dataset
        if ds is None:
            return [
                gr.update(),
                "No dataset loaded.",
                *_control_updates(running=False),
            ] + _no_preview_updates()

        runner = _get_runner()
        if runner.is_running:
            return _already_running_outputs()

        checkpoint = _latest_checkpoint()
        if checkpoint is None:
            return [
                gr.update(),
                "No checkpoint found to resume from.",
                *_control_updates(running=False),
            ] + _no_preview_updates()

        config = _build_config(
            max_epochs, learning_rate, dropout, weight_decay,
            kl_weight_max, preset_name,
        )
        reset_metrics_buffer()
        # Keep writing into the run being resumed: its checkpoints belong
        # to the same run, so retention and "best" selection stay correct.
        output_dir = checkpoint.parent.parent
        file_paths = list(ds.valid_files) if hasattr(ds, "valid_files") else []

        try:
            runner.resume(
                config=config,
                file_paths=file_paths,
                output_dir=output_dir,
                device=app_state.device,
                checkpoint_path=checkpoint,
                callback=_training_callback,
                models_dir=app_state.models_dir,
                dataset_name=ds.name,
                model_name=model_name.strip() if model_name else "",
            )
        except RuntimeError:
            return _already_running_outputs()
        app_state.training_active = True

        return [
            gr.update(value=None),
            f"Resuming {output_dir.name} from {checkpoint.name}",
            *_control_updates(running=True),
        ] + [gr.update(visible=False, value=None) for _ in range(_MAX_PREVIEW_SLOTS)]

    resume_btn.click(
        fn=_resume_training,
        inputs=_train_inputs,
        outputs=_run_outputs,
    )

    def _poll_training() -> list:
        """Timer tick: read metrics buffer and update dashboard."""
        buf = app_state.metrics_buffer
        epoch_metrics = buf.get("epoch_metrics", [])
        previews = buf.get("previews", [])
        is_complete = buf.get("complete", False)

        # Detect training thread crash: thread died without emitting
        # a TrainingCompleteEvent (exception was caught by runner).
        runner = app_state.training_runner
        if (
            runner is not None
            and not runner.is_running
            and not is_complete
            and app_state.training_active
        ):
            app_state.training_active = False
            error_msg = "**Training failed.**"
            if runner.last_error is not None:
                error_msg += f"\n\n`{runner.last_error}`"
            return [
                gr.update(),                    # loss_plot
                error_msg,                      # stats_md
                *_control_updates(running=False),
            ] + _no_preview_updates()

        # Build loss chart
        chart = build_loss_chart(epoch_metrics)

        # Build stats string
        if epoch_metrics:
            latest = epoch_metrics[-1]
            eta_str = _format_eta(latest.eta_seconds)
            stats = (
                f"**Epoch {latest.epoch + 1}/{latest.total_epochs}** | "
                f"Train: {latest.train_loss:.4f} | Val: {latest.val_loss:.4f} | "
                f"LR: {latest.learning_rate:.2e} | ETA: {eta_str}"
            )
        else:
            stats = "Waiting for first epoch..."
        cancelled = buf.get("cancelled", False)
        if cancelled and not is_complete:
            stats = "Cancelling -- saving checkpoint and model..."

        # Build preview audio updates
        audio_updates = []
        for i in range(_MAX_PREVIEW_SLOTS):
            if i < len(previews):
                p = previews[i]
                audio_updates.append(
                    gr.update(
                        visible=True,
                        value=str(p.audio_path),
                        label=f"Epoch {p.epoch + 1}",
                    )
                )
            else:
                audio_updates.append(gr.update())

        # Handle training completion
        if is_complete:
            if epoch_metrics:
                latest = epoch_metrics[-1]
                stats = (
                    f"**Training complete!** "
                    f"Final -- Train: {latest.train_loss:.4f} | "
                    f"Val: {latest.val_loss:.4f} | "
                    f"Epochs: {latest.epoch + 1}"
                )
            else:
                stats = "Training complete."
            if cancelled:
                stats = stats.replace("**Training complete!**", "**Training cancelled.**")
                if not epoch_metrics:
                    stats = "Training cancelled before the first epoch finished."
            app_state.training_active = False

            # Refresh model library catalog so Library tab sees the new model
            try:
                from distill.library.catalog import ModelLibrary

                if app_state.models_dir:
                    app_state.model_library = ModelLibrary(app_state.models_dir)
            except Exception:
                logger.warning("Failed to refresh model library after training", exc_info=True)
            return [
                chart,                          # loss_plot
                stats,                          # stats_md
                *_control_updates(running=False),
            ] + audio_updates

        return [
            chart,       # loss_plot
            stats,       # stats_md
            gr.update(), # timer (keep active)
            gr.update(), # start_btn (keep disabled)
            gr.update(), # cancel_btn (keep enabled)
            gr.update(), # resume_btn (keep disabled)
        ] + audio_updates

    timer.tick(
        fn=_poll_training,
        inputs=None,
        outputs=_run_outputs,
    )

    return {
        "empty_state": empty_state,
        "train_ui": train_ui,
        "resume_btn": resume_btn,
        "check_dataset_ready": _check_dataset_ready,
    }
