"""Train tab: VQ-VAE configuration, live loss chart, previews, and cancel.

Surfaces the :class:`TrainingRunner` (Phase 3) with a Timer-polled
dashboard.  Training runs in a daemon thread; the Timer reads the
shared ``app_state.metrics_buffer`` every 2 seconds and pushes updates
to the loss chart, stats panel, and preview audio slots.

Key design points:
- ``gr.Timer(active=False)`` -- activated only when training starts.
- 20 pre-created ``gr.Audio`` slots (hidden) revealed as previews arrive.
- VQ-VAE controls: RVQ Levels slider, Commitment Weight input,
  auto-determined Codebook Size display.
- Per-level codebook health (utilization, perplexity, dead codes) shown
  during training in the stats panel.
- Empty state guidance when no dataset is loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from distill.training.config import (
    VQVAEConfig,
    get_adaptive_vqvae_config,
)
from distill.training.metrics import (
    EpochMetrics,
    PreviewEvent,
    TrainingCompleteEvent,
    VQEpochMetrics,
)
from distill.training.runner import TrainingRunner
from distill.ui.components.guided_nav import get_empty_state_message
from distill.ui.components.loss_chart import build_loss_chart
from distill.ui.state import app_state, reset_metrics_buffer

logger = logging.getLogger(__name__)

# Number of pre-created audio preview slots
_MAX_PREVIEW_SLOTS = 20


# -------------------------------------------------------------------
# Training callback (runs in training thread, stores events)
# -------------------------------------------------------------------


def _training_callback(event: object) -> None:
    """MetricsCallback that stores events for the Timer to read.

    Thread-safe: ``list.append`` is atomic under CPython GIL.
    Handles both v1.0 (EpochMetrics) and v1.1 (VQEpochMetrics) events.
    """
    if isinstance(event, (EpochMetrics, VQEpochMetrics)):
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
# Helper: output directory for current dataset
# -------------------------------------------------------------------


def _get_output_dir() -> Path:
    """Return the training output directory for the current dataset."""
    ds = app_state.current_dataset
    name = ds.name if ds else "default"
    return app_state.datasets_dir / name / "training"


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

        # -- VQ-VAE Config section ----------------------------------------
        with gr.Row():
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

        with gr.Row():
            rvq_levels_slider = gr.Slider(
                minimum=2,
                maximum=4,
                step=1,
                value=3,
                label="RVQ Levels",
                info="Fewer levels = faster, more levels = finer detail",
            )
            commitment_weight_num = gr.Slider(
                minimum=0.01,
                maximum=1.0,
                step=0.01,
                value=0.25,
                label="Commitment Weight",
                info="Controls codebook learning rate. Most users leave this at 0.25.",
            )

        codebook_size_display = gr.Textbox(
            label="Codebook Size (auto)",
            interactive=False,
            value="Auto-determined from dataset size",
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

        # -- Control buttons ----------------------------------------------
        with gr.Row():
            start_btn = gr.Button("Train VQ-VAE", variant="primary")
            cancel_btn = gr.Button("Cancel", variant="stop", interactive=False)
            # Resume hidden for VQ-VAE -- checkpoint resume needs runner adaptation
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

    def _check_dataset_ready() -> tuple:
        """Check if a dataset is loaded and toggle empty/main UI."""
        has_ds = app_state.current_dataset is not None
        # Resume hidden for VQ-VAE (checkpoint resume needs adaptation)
        return (
            gr.update(visible=not has_ds),   # empty_state
            gr.update(visible=has_ds),        # train_ui
            gr.update(visible=False),         # resume_btn (always hidden for VQ-VAE)
        )

    # Poll dataset readiness when the tab renders via a load event
    app_context = gr.context.Context.root_block
    if app_context is not None:
        app_context.load(
            fn=_check_dataset_ready,
            inputs=None,
            outputs=[empty_state, train_ui, resume_btn],
        )

    def _start_training(
        model_name: str,
        max_epochs: int,
        learning_rate: float,
        dropout: float,
        weight_decay: float,
        rvq_levels: int,
        commitment_weight: float,
    ) -> list:
        """Start VQ-VAE training in a background thread."""
        ds = app_state.current_dataset
        if ds is None:
            return [
                gr.update(),          # loss_plot
                "No dataset loaded. Go to the Data tab first.",  # stats_md
                gr.Timer(active=False),  # timer
                gr.update(interactive=True),   # start_btn
                gr.update(interactive=False),  # cancel_btn
                gr.update(),          # codebook_size_display
            ] + [gr.update() for _ in range(_MAX_PREVIEW_SLOTS)]

        # Build VQVAEConfig from adaptive base, override with UI values
        file_count = len(ds.valid_files) if hasattr(ds, "valid_files") else 10
        config = get_adaptive_vqvae_config(file_count)
        config.max_epochs = int(max_epochs)
        config.learning_rate = float(learning_rate)
        config.dropout = float(dropout)
        config.weight_decay = float(weight_decay)
        config.num_quantizers = int(rvq_levels)
        config.commitment_weight = float(commitment_weight)

        # Clear metrics buffer
        reset_metrics_buffer()

        # Create runner and start
        output_dir = _get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        runner = TrainingRunner()
        app_state.training_runner = runner
        app_state.training_active = True

        file_paths = list(ds.valid_files) if hasattr(ds, "valid_files") else []

        try:
            runner.start_vqvae(
                config=config,
                file_paths=file_paths,
                output_dir=output_dir,
                device=app_state.device,
                callback=_training_callback,
                models_dir=app_state.models_dir,
                dataset_name=ds.name if ds else "untitled",
                model_name=model_name.strip() if model_name else "",
            )
        except RuntimeError as exc:
            app_state.training_active = False
            return [
                gr.update(),
                f"Failed to start training: {exc}",
                gr.Timer(active=False),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(),          # codebook_size_display
            ] + [gr.update() for _ in range(_MAX_PREVIEW_SLOTS)]

        # Show auto-determined codebook size
        cb_info = f"{config.codebook_size} entries x {config.num_quantizers} levels"

        # Return: activate timer, disable Start, enable Cancel, show codebook size
        return [
            gr.update(value=None),           # loss_plot (clear)
            "Training started...",           # stats_md
            gr.Timer(active=True),           # timer
            gr.update(interactive=False),    # start_btn
            gr.update(interactive=True),     # cancel_btn
            gr.update(value=cb_info),        # codebook_size_display
        ] + [gr.update(visible=False, value=None) for _ in range(_MAX_PREVIEW_SLOTS)]

    start_btn.click(
        fn=_start_training,
        inputs=[
            model_name_input,
            max_epochs_num,
            learning_rate_num,
            dropout_slider,
            weight_decay_slider,
            rvq_levels_slider,
            commitment_weight_num,
        ],
        outputs=[
            loss_plot, stats_md, timer, start_btn, cancel_btn,
            codebook_size_display,
        ] + preview_audios,
    )

    def _cancel_training() -> list:
        """Cancel training and deactivate timer."""
        if app_state.training_runner is not None:
            app_state.training_runner.cancel()
        app_state.training_active = False

        return [
            gr.Timer(active=False),         # timer
            gr.update(interactive=True),    # start_btn
            gr.update(interactive=False),   # cancel_btn
            "Training cancelled.",          # stats_md
        ]

    cancel_btn.click(
        fn=_cancel_training,
        inputs=None,
        outputs=[timer, start_btn, cancel_btn, stats_md],
    )

    # --- Resume is hidden for VQ-VAE (checkpoint resume needs adaptation) ---

    # Resume button is hidden -- VQ-VAE checkpoint resume needs runner
    # adaptation.  The button remains in the component tree but is never
    # shown (visible=False is set at construction and _check_dataset_ready
    # never reveals it for VQ-VAE).


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
                gr.Timer(active=False),         # timer
                gr.update(interactive=True),    # start_btn
                gr.update(interactive=False),   # cancel_btn
            ] + [gr.update() for _ in range(_MAX_PREVIEW_SLOTS)]

        # Build loss chart
        chart = build_loss_chart(epoch_metrics)

        # Build stats string -- detect VQ-VAE vs v1.0 metrics
        if epoch_metrics:
            latest = epoch_metrics[-1]
            eta_str = _format_eta(latest.eta_seconds)

            if isinstance(latest, VQEpochMetrics):
                # VQ-VAE stats with codebook health
                stats = (
                    f"**Epoch {latest.epoch + 1}/{latest.total_epochs}** | "
                    f"Train: {latest.train_loss:.4f} | Val: {latest.val_loss:.4f} | "
                    f"Recon: {latest.val_recon_loss:.4f} | "
                    f"Commit: {latest.val_commit_loss:.4f} | "
                    f"LR: {latest.learning_rate:.2e} | ETA: {eta_str}"
                )

                # Codebook health table
                if latest.codebook_health:
                    stats += "\n\n**Codebook Health:**\n"
                    stats += "| Level | Utilization | Perplexity | Dead Codes |\n"
                    stats += "|-------|-------------|------------|------------|\n"
                    for level_key in sorted(latest.codebook_health.keys()):
                        h = latest.codebook_health[level_key]
                        util_pct = h.get("utilization", 0) * 100
                        perplexity = h.get("perplexity", 0)
                        dead_codes = h.get("dead_codes", 0)
                        stats += (
                            f"| {level_key} | {util_pct:.0f}% | "
                            f"{perplexity:.1f} | {dead_codes} |\n"
                        )

                # Utilization warnings
                if latest.utilization_warnings:
                    stats += "\n**Warnings:** " + " | ".join(
                        latest.utilization_warnings
                    )
            else:
                # v1.0 stats (original format)
                stats = (
                    f"**Epoch {latest.epoch + 1}/{latest.total_epochs}** | "
                    f"Train: {latest.train_loss:.4f} | Val: {latest.val_loss:.4f} | "
                    f"LR: {latest.learning_rate:.2e} | ETA: {eta_str}"
                )
        else:
            stats = "Waiting for first epoch..."

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
                if isinstance(latest, VQEpochMetrics):
                    stats = (
                        f"**Training complete!** "
                        f"Final -- Train: {latest.train_loss:.4f} | "
                        f"Val: {latest.val_loss:.4f} | "
                        f"Recon: {latest.val_recon_loss:.4f} | "
                        f"Commit: {latest.val_commit_loss:.4f} | "
                        f"Epochs: {latest.epoch + 1}"
                    )
                else:
                    stats = (
                        f"**Training complete!** "
                        f"Final -- Train: {latest.train_loss:.4f} | "
                        f"Val: {latest.val_loss:.4f} | "
                        f"Epochs: {latest.epoch + 1}"
                    )
            else:
                stats = "Training complete."
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
                gr.Timer(active=False),         # timer
                gr.update(interactive=True),    # start_btn
                gr.update(interactive=False),   # cancel_btn
            ] + audio_updates

        return [
            chart,       # loss_plot
            stats,       # stats_md
            gr.update(), # timer (keep active)
            gr.update(), # start_btn (keep disabled)
            gr.update(), # cancel_btn (keep enabled)
        ] + audio_updates

    timer.tick(
        fn=_poll_training,
        inputs=None,
        outputs=[loss_plot, stats_md, timer, start_btn, cancel_btn] + preview_audios,
    )

    return {
        "empty_state": empty_state,
        "train_ui": train_ui,
        "resume_btn": resume_btn,
        "check_dataset_ready": _check_dataset_ready,
    }
