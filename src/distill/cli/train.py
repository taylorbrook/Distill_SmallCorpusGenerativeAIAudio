"""CLI command for training VQ-VAE models on audio datasets.

Provides ``distill train DATASET_DIR`` with Rich progress bars, SIGINT
graceful cancellation, codebook-specific CLI flags (--codebook-size,
--rvq-levels, --commitment-weight), per-level codebook health display
during training, and a comprehensive end-of-training summary.

Calls ``training.loop.train_vqvae()`` directly (NOT TrainingRunner) --
the background-thread pattern is for the GUI.  The CLI is the main
process.

All heavy imports (torch, training modules) are lazy inside the command
function body for fast ``distill train --help`` response.
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
    epochs: Annotated[Optional[int], typer.Option("--epochs", "-e", help="Override max epochs")] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--lr", help="Override learning rate")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="Override batch size")] = None,
    codebook_size: Annotated[
        Optional[int],
        typer.Option("--codebook-size", help="Override auto-determined codebook size (64/128/256)"),
    ] = None,
    rvq_levels: Annotated[
        Optional[int],
        typer.Option("--rvq-levels", help="Number of RVQ quantizer levels (2-4, default: 3)"),
    ] = None,
    commitment_weight: Annotated[
        Optional[float],
        typer.Option("--commitment-weight", help="Commitment loss weight (default: 0.25)"),
    ] = None,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", "-o", help="Training output directory")] = None,
    device: str = typer.Option("auto", "--device", help="Compute device: auto, mps, cuda, cpu"),
    config: Annotated[Optional[Path], typer.Option("--config", help="Config file path")] = None,
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    model_name: Annotated[
        Optional[str],
        typer.Option("--model-name", help="Name for saved model (default: dataset_name_timestamp)"),
    ] = None,
) -> None:
    """Train a VQ-VAE model on an audio dataset."""
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

    from distill.audio.validation import collect_audio_files
    from distill.cli import bootstrap
    from distill.config.settings import resolve_path
    from distill.training.config import VQVAEConfig, get_adaptive_vqvae_config

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
    # Build VQVAEConfig (auto-adaptive, then apply overrides)
    # ------------------------------------------------------------------
    vqvae_config: VQVAEConfig = get_adaptive_vqvae_config(len(file_paths))
    auto_codebook_size = vqvae_config.codebook_size

    # Apply individual overrides
    if epochs is not None:
        vqvae_config.max_epochs = epochs
    if learning_rate is not None:
        vqvae_config.learning_rate = learning_rate
    if batch_size is not None:
        vqvae_config.batch_size = batch_size
    if codebook_size is not None:
        vqvae_config.codebook_size = codebook_size
    if rvq_levels is not None:
        vqvae_config.num_quantizers = rvq_levels
    if commitment_weight is not None:
        vqvae_config.commitment_weight = commitment_weight

    # Determine codebook size source for display
    cs_source = "(override)" if codebook_size is not None else "(auto)"

    console.print(
        f"[bold]Config:[/bold] codebook_size={vqvae_config.codebook_size} {cs_source}, "
        f"rvq_levels={vqvae_config.num_quantizers}, "
        f"commitment_weight={vqvae_config.commitment_weight}, "
        f"epochs={vqvae_config.max_epochs}, "
        f"lr={vqvae_config.learning_rate:.1e}, "
        f"batch_size={vqvae_config.batch_size}"
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
    # Track warnings across epochs for end-of-training summary
    # ------------------------------------------------------------------
    all_warnings: list[str] = []
    last_codebook_health: dict | None = None

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
                total=vqvae_config.max_epochs,
                epoch=0,
                total_epochs=vqvae_config.max_epochs,
                train_loss=0.0,
                val_loss=0.0,
            )

            def cli_callback(event: object) -> None:
                nonlocal last_codebook_health
                from distill.training.metrics import (
                    PreviewEvent,
                    TrainingCompleteEvent,
                    VQEpochMetrics,
                    VQStepMetrics,
                )

                if isinstance(event, VQEpochMetrics):
                    progress.update(
                        task_id,
                        completed=event.epoch + 1,
                        epoch=event.epoch + 1,
                        total_epochs=event.total_epochs,
                        train_loss=event.train_loss,
                        val_loss=event.val_loss,
                    )
                    # Print per-level codebook health after each epoch
                    if event.codebook_health:
                        last_codebook_health = event.codebook_health
                        console.print("  [dim]Codebook Health:[/dim]")
                        for level_name, stats in sorted(event.codebook_health.items()):
                            util = stats.get("utilization", 0)
                            perp = stats.get("perplexity", 0)
                            dead = stats.get("dead_codes", 0)
                            util_color = "green" if util >= 0.5 else ("yellow" if util >= 0.3 else "red")
                            console.print(
                                f"    {level_name}: "
                                f"[{util_color}]{util:.0%}[/{util_color}] util, "
                                f"{perp:.1f} perplexity, "
                                f"{dead} dead codes"
                            )
                    # Print warnings if any
                    if event.utilization_warnings:
                        for w in event.utilization_warnings:
                            console.print(f"  [bold yellow]Warning:[/bold yellow] {w}")
                            all_warnings.append(f"Epoch {event.epoch}: {w}")
                elif isinstance(event, PreviewEvent):
                    console.print(f"  [dim]Preview saved: {event.audio_path}[/dim]")
                elif isinstance(event, TrainingCompleteEvent):
                    pass  # Handled after train_vqvae() returns

            # Resolve models_base for train_vqvae
            models_base = resolve_path(
                app_config["paths"].get("models", "data/models"),
                base_dir=config_path.parent,
            )
            models_base.mkdir(parents=True, exist_ok=True)

            # Call train_vqvae() directly (NOT TrainingRunner)
            from distill.training.loop import train_vqvae

            result = train_vqvae(
                config=vqvae_config,
                file_paths=file_paths,
                output_dir=output_dir,
                device=torch_device,
                callback=cli_callback,
                cancel_event=cancel_event,
                models_dir=models_base,
                dataset_name=dataset_dir.name,
                model_name=model_name or "",
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
    # Build result summary
    # ------------------------------------------------------------------
    metrics_history = result.get("metrics_history")
    final_codebook_health = result.get("final_codebook_health") or last_codebook_health

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

    # Determine saved model path from output dir
    # train_vqvae() saves to models_dir, path is printed during training
    saved_model_path = None
    if models_base.exists():
        # Find the most recently created .distill file
        distill_files = sorted(models_base.glob("*.distill"), key=lambda p: p.stat().st_mtime, reverse=True)
        if distill_files:
            saved_model_path = distill_files[0]

    result_summary = {
        "output_dir": str(output_dir),
        "epochs_completed": epochs_completed,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "model_path": str(saved_model_path) if saved_model_path else None,
        "model_name": model_name or None,
        "codebook_size": vqvae_config.codebook_size,
        "codebook_size_source": "override" if codebook_size is not None else "auto",
        "rvq_levels": vqvae_config.num_quantizers,
        "commitment_weight": vqvae_config.commitment_weight,
        "codebook_health": final_codebook_health,
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
        console.print()

        # Codebook health (final)
        if final_codebook_health:
            console.print("  [bold]Codebook Health (Final):[/bold]")
            for level_name, stats in sorted(final_codebook_health.items()):
                util = stats.get("utilization", 0)
                perp = stats.get("perplexity", 0)
                dead = stats.get("dead_codes", 0)
                level_label = level_name.replace("level_", "Level ")
                util_color = "green" if util >= 0.5 else ("yellow" if util >= 0.3 else "red")
                console.print(
                    f"    {level_label}: "
                    f"[{util_color}]{util:.0%}[/{util_color}] utilization, "
                    f"{perp:.1f} perplexity, "
                    f"{dead} dead codes"
                )
            console.print()

        # Config summary
        console.print(
            f"  [bold]Config:[/bold] codebook_size={vqvae_config.codebook_size} {cs_source}, "
            f"rvq_levels={vqvae_config.num_quantizers}, "
            f"commitment_weight={vqvae_config.commitment_weight}"
        )
        if saved_model_path:
            console.print(f"  [bold]Model saved:[/bold] {saved_model_path}")
        console.print(f"  [bold]Output dir:[/bold] {output_dir}")

        # Warnings during training
        if all_warnings:
            console.print()
            console.print("  [bold yellow]Warnings during training:[/bold yellow]")
            for w in all_warnings:
                console.print(f"    {w}")

        # Machine-readable output to stdout
        print(str(output_dir))
