"""CLI command for training models on audio datasets.

Provides ``sda train DATASET_DIR`` with Rich progress bars, SIGINT
graceful cancellation, training preset selection, and individual
parameter overrides.

Calls ``training.loop.train()`` directly (NOT TrainingRunner) -- the
background-thread pattern is for the GUI.  The CLI is the main process.

All heavy imports (torch, training modules) are lazy inside the command
function body for fast ``sda train --help`` response.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def train_cmd(
    dataset_dir: Path = typer.Argument(
        ...,
        help="Path to audio dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    preset: str = typer.Option(
        "auto",
        "--preset",
        help="Training preset: auto, conservative, balanced, aggressive",
    ),
    epochs: Annotated[Optional[int], typer.Option("--epochs", "-e", help="Override max epochs")] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--lr", help="Override learning rate")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="Override batch size")] = None,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", "-o", help="Training output directory")] = None,
    device: str = typer.Option("auto", "--device", help="Compute device: auto, mps, cuda, cpu"),
    config: Annotated[Optional[Path], typer.Option("--config", help="Config file path")] = None,
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    model_name: Annotated[
        Optional[str],
        typer.Option("--model-name", help="Name for saved model (default: dataset_name_timestamp)"),
    ] = None,
) -> None:
    """Train a model on an audio dataset."""
    import json as json_mod
    import signal
    import threading
    from datetime import datetime

    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    from small_dataset_audio.audio.validation import collect_audio_files
    from small_dataset_audio.cli import bootstrap
    from small_dataset_audio.config.settings import resolve_path
    from small_dataset_audio.training.config import (
        OverfittingPreset,
        TrainingConfig,
        get_adaptive_config,
    )

    console = Console(stderr=True)

    # ------------------------------------------------------------------
    # Bootstrap config + device
    # ------------------------------------------------------------------
    app_config, torch_device, config_path = bootstrap(config, device)

    # ------------------------------------------------------------------
    # Resolve output directory
    # ------------------------------------------------------------------
    if output_dir is None:
        models_base = resolve_path(
            app_config["paths"].get("models", "data/models"),
            base_dir=config_path.parent,
        )
        output_dir = models_base / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect audio files
    # ------------------------------------------------------------------
    file_paths = collect_audio_files(dataset_dir)
    if not file_paths:
        console.print(
            f"[bold red]Error:[/bold red] No audio files found in {dataset_dir}"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[bold]Dataset:[/bold] {len(file_paths)} audio files from {dataset_dir}"
    )

    # ------------------------------------------------------------------
    # Build TrainingConfig
    # ------------------------------------------------------------------
    preset_lower = preset.lower()
    valid_presets = {"auto", "conservative", "balanced", "aggressive"}
    if preset_lower not in valid_presets:
        console.print(
            f"[bold red]Error:[/bold red] Invalid preset '{preset}'. "
            f"Choose from: {', '.join(sorted(valid_presets))}"
        )
        raise typer.Exit(code=1)

    training_config: TrainingConfig = get_adaptive_config(len(file_paths))

    if preset_lower != "auto":
        # Override preset selection
        target_preset = OverfittingPreset(preset_lower)
        training_config.preset = target_preset
        # Apply preset parameters
        from small_dataset_audio.training.config import _PRESET_PARAMS

        params = _PRESET_PARAMS[target_preset]
        training_config.max_epochs = int(params["max_epochs"])
        training_config.learning_rate = float(params["learning_rate"])
        training_config.kl_warmup_fraction = float(params["kl_warmup_fraction"])
        training_config.regularization.dropout = float(params["dropout"])
        training_config.regularization.weight_decay = float(params["weight_decay"])
        training_config.regularization.augmentation_expansion = int(params["augmentation_expansion"])
        training_config.regularization.gradient_clip_norm = float(params["gradient_clip_norm"])

    # Apply individual overrides
    if epochs is not None:
        training_config.max_epochs = epochs
    if learning_rate is not None:
        training_config.learning_rate = learning_rate
    if batch_size is not None:
        training_config.batch_size = batch_size

    console.print(
        f"[bold]Config:[/bold] preset={training_config.preset.value}, "
        f"epochs={training_config.max_epochs}, "
        f"lr={training_config.learning_rate:.1e}, "
        f"batch_size={training_config.batch_size}"
    )
    console.print(f"[bold]Output:[/bold] {output_dir}")
    console.print()

    # ------------------------------------------------------------------
    # SIGINT handling
    # ------------------------------------------------------------------
    cancel_event = threading.Event()

    def handle_sigint(signum: int, frame: object) -> None:
        console.print("\n[yellow]Cancelling training (saving checkpoint)...[/yellow]")
        cancel_event.set()

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)

    # ------------------------------------------------------------------
    # Rich progress bar + training
    # ------------------------------------------------------------------
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"),
            BarColumn(),
            TextColumn("train={task.fields[train_loss]:.4f}"),
            TextColumn("val={task.fields[val_loss]:.4f}"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "Training",
                total=training_config.max_epochs,
                epoch=0,
                total_epochs=training_config.max_epochs,
                train_loss=0.0,
                val_loss=0.0,
            )

            def cli_callback(event: object) -> None:
                from small_dataset_audio.training.metrics import (
                    EpochMetrics,
                    PreviewEvent,
                    TrainingCompleteEvent,
                )

                if isinstance(event, EpochMetrics):
                    progress.update(
                        task_id,
                        completed=event.epoch + 1,
                        epoch=event.epoch + 1,
                        total_epochs=event.total_epochs,
                        train_loss=event.train_loss,
                        val_loss=event.val_loss,
                    )
                elif isinstance(event, PreviewEvent):
                    console.print(f"  [dim]Preview saved: {event.audio_path}[/dim]")
                elif isinstance(event, TrainingCompleteEvent):
                    pass  # Handled after train() returns

            # Call train() directly (NOT TrainingRunner)
            from small_dataset_audio.training.loop import train as run_training

            result = run_training(
                config=training_config,
                file_paths=file_paths,
                output_dir=output_dir,
                device=torch_device,
                callback=cli_callback,
                cancel_event=cancel_event,
            )
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)

    # ------------------------------------------------------------------
    # Post-training output
    # ------------------------------------------------------------------
    if cancel_event.is_set():
        console.print("\n[yellow]Training cancelled.[/yellow] Checkpoint saved.")
        raise typer.Exit(code=3)

    # ------------------------------------------------------------------
    # Auto-save model to library
    # ------------------------------------------------------------------
    best_checkpoint = result.get("best_checkpoint_path")
    saved_model_path = None
    auto_name = None

    if best_checkpoint is not None:
        from small_dataset_audio.models.persistence import ModelMetadata, save_model_from_checkpoint

        # Generate model name: user override or dataset_name_timestamp
        if model_name:
            auto_name = model_name
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_name = f"{dataset_dir.name}_{ts}"

        meta = ModelMetadata(
            name=auto_name,
            dataset_name=dataset_dir.name,
            dataset_file_count=len(file_paths),
            training_date=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        )

        try:
            models_base = resolve_path(
                app_config["paths"].get("models", "data/models"),
                base_dir=config_path.parent,
            )
            models_base.mkdir(parents=True, exist_ok=True)

            saved_model_path = save_model_from_checkpoint(
                checkpoint_path=best_checkpoint,
                metadata=meta,
                models_dir=models_base,
            )
            console.print(f"\n[bold green]Model saved to library![/bold green]")
            console.print(f"  Name: {auto_name}")
            console.print(f"  Path: {saved_model_path}")
            console.print(f"  ID:   {meta.model_id}")
        except Exception as exc:
            console.print(f"\n[yellow]Warning:[/yellow] Failed to auto-save model: {exc}")
            logger.warning("Auto-save model failed", exc_info=True)

    # Build result summary
    metrics_history = result.get("metrics_history")

    epochs_completed = 0
    final_train_loss = 0.0
    final_val_loss = 0.0
    best_val_loss = float("inf")
    best_epoch = 0

    if metrics_history and metrics_history.epoch_metrics:
        last = metrics_history.epoch_metrics[-1]
        epochs_completed = last.epoch + 1
        final_train_loss = last.train_loss
        final_val_loss = last.val_loss
        best_val_loss = min(m.val_loss for m in metrics_history.epoch_metrics)
        best_epoch = metrics_history.get_best_epoch()

    result_summary = {
        "output_dir": str(output_dir),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "epochs_completed": epochs_completed,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "model_path": str(saved_model_path) if saved_model_path else None,
        "model_name": auto_name if saved_model_path else None,
    }

    if json_output:
        print(json_mod.dumps(result_summary, indent=2))
    else:
        console.print()
        console.print("[bold green]Training complete![/bold green]")
        console.print(f"  Epochs: {epochs_completed}")
        console.print(f"  Final train loss: {final_train_loss:.4f}")
        console.print(f"  Final val loss:   {final_val_loss:.4f}")
        console.print(f"  Best val loss:    {best_val_loss:.4f} (epoch {best_epoch})")
        if best_checkpoint:
            console.print(f"  Best checkpoint:  {best_checkpoint}")
        if saved_model_path:
            console.print(f"  Model:            {saved_model_path}")
        console.print(f"  Output dir:       {output_dir}")
        # Machine-readable output to stdout
        print(str(output_dir))
