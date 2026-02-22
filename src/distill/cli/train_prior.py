"""CLI command for training an autoregressive prior on a VQ-VAE model.

Provides ``distill train-prior MODEL_PATH DATASET_DIR`` with Rich per-epoch
progress display, SIGINT graceful cancellation, per-epoch perplexity and
memorization warnings, and prior bundling into the model file after training.

Follows the same patterns as ``cli/train.py``:

- Lazy heavy imports inside command body (fast ``--help``).
- Rich console on stderr for progress, JSON on stdout for scripting.
- SIGINT handling with cancel event.
- ``(auto)`` / ``(override)`` suffix on each config parameter.

All heavy imports (torch, training modules) are lazy inside the command
function body for fast ``distill train-prior --help`` response.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def train_prior_cmd(
    model_path: Path = typer.Argument(
        ...,
        help="Path to a trained VQ-VAE .distill model file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    dataset_dir: Path = typer.Argument(
        ...,
        help="Path to audio dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    epochs: Annotated[
        Optional[int], typer.Option("--epochs", "-e", help="Override max epochs")
    ] = None,
    hidden_size: Annotated[
        Optional[int], typer.Option("--hidden-size", help="Override transformer hidden dimension")
    ] = None,
    layers: Annotated[
        Optional[int], typer.Option("--layers", help="Override number of transformer layers")
    ] = None,
    heads: Annotated[
        Optional[int], typer.Option("--heads", help="Override number of attention heads")
    ] = None,
    lr: Annotated[
        Optional[float], typer.Option("--lr", help="Override learning rate")
    ] = None,
    device: str = typer.Option(
        "auto", "--device", help="Compute device: auto, mps, cuda, cpu"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON"
    ),
) -> None:
    """Train an autoregressive prior on a VQ-VAE model."""
    import json as json_mod
    import signal
    import threading

    from rich.console import Console

    from distill.audio.validation import collect_audio_files
    from distill.training.prior_config import get_adaptive_prior_config

    console = Console(stderr=True)

    # ------------------------------------------------------------------
    # Count audio files to determine adaptive config
    # ------------------------------------------------------------------
    file_paths = collect_audio_files(dataset_dir)
    if not file_paths:
        console.print(
            f"[bold red]Error:[/bold red] No audio files found in {dataset_dir}"
        )
        raise typer.Exit(code=1)

    file_count = len(file_paths)

    # ------------------------------------------------------------------
    # Build adaptive config, then apply CLI overrides
    # ------------------------------------------------------------------
    prior_config = get_adaptive_prior_config(file_count)
    prior_config.device = device

    # Track which params were overridden for display
    overrides: dict[str, bool] = {
        "epochs": epochs is not None,
        "hidden_size": hidden_size is not None,
        "layers": layers is not None,
        "heads": heads is not None,
        "lr": lr is not None,
    }

    if epochs is not None:
        prior_config.max_epochs = epochs
    if hidden_size is not None:
        prior_config.hidden_size = hidden_size
    if layers is not None:
        prior_config.num_layers = layers
    if heads is not None:
        prior_config.num_heads = heads
    if lr is not None:
        prior_config.learning_rate = lr

    # ------------------------------------------------------------------
    # Display config summary
    # ------------------------------------------------------------------
    def _suffix(key: str) -> str:
        return "(override)" if overrides.get(key) else "(auto)"

    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print(f"[bold]Dataset:[/bold] {file_count} audio files from {dataset_dir}")
    console.print(
        f"[bold]Config:[/bold] "
        f"epochs={prior_config.max_epochs} {_suffix('epochs')}, "
        f"hidden_size={prior_config.hidden_size} {_suffix('hidden_size')}, "
        f"layers={prior_config.num_layers} {_suffix('layers')}, "
        f"heads={prior_config.num_heads} {_suffix('heads')}, "
        f"lr={prior_config.learning_rate:.1e} {_suffix('lr')}"
    )
    console.print()

    # ------------------------------------------------------------------
    # SIGINT handling
    # ------------------------------------------------------------------
    cancel_event = threading.Event()

    def handle_sigint(signum: int, frame: object) -> None:
        console.print("\n[yellow]Cancelling prior training...[/yellow]")
        cancel_event.set()

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)

    # ------------------------------------------------------------------
    # Track memorization warnings across epochs
    # ------------------------------------------------------------------
    all_memorization_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Training with per-epoch progress display
    # ------------------------------------------------------------------
    try:
        from distill.training.metrics import PriorEpochMetrics, PriorTrainingCompleteEvent
        from distill.training.prior_loop import train_prior

        def cli_callback(event: object) -> None:
            if isinstance(event, PriorEpochMetrics):
                epoch_display = event.epoch + 1
                console.print(
                    f"  Epoch {epoch_display}/{event.total_epochs} | "
                    f"train_loss: {event.train_loss:.4f} | "
                    f"val_loss: {event.val_loss:.4f} | "
                    f"perplexity: {event.val_perplexity:.1f} | "
                    f"best: {event.best_perplexity:.1f}"
                )
                if event.is_memorizing:
                    console.print(
                        f"  [bold yellow]Warning:[/bold yellow] "
                        f"{event.memorization_message}"
                    )
                    all_memorization_warnings.append(
                        f"Epoch {epoch_display}: {event.memorization_message}"
                    )
            elif isinstance(event, PriorTrainingCompleteEvent):
                pass  # Handled after train_prior() returns

        result = train_prior(
            model_path=model_path,
            dataset_dir=dataset_dir,
            prior_config=prior_config,
            callback=cli_callback,
            cancel_event=cancel_event,
        )
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)

    # ------------------------------------------------------------------
    # Handle cancellation
    # ------------------------------------------------------------------
    if cancel_event.is_set():
        console.print("\n[yellow]Prior training cancelled.[/yellow]")
        raise typer.Exit(code=3)

    # ------------------------------------------------------------------
    # Save prior into model file
    # ------------------------------------------------------------------
    from distill.models.persistence import save_prior_to_model

    save_prior_to_model(
        model_path=model_path,
        prior_model=result["prior_model"],
        prior_config=result["prior_config"],
        prior_metadata=result["prior_metadata"],
    )

    # ------------------------------------------------------------------
    # Post-training output
    # ------------------------------------------------------------------
    prior_meta = result["prior_metadata"]
    prior_cfg = result["prior_config"]

    result_summary = {
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "file_count": file_count,
        "epochs_trained": prior_meta["epochs_trained"],
        "final_perplexity": prior_meta["final_perplexity"],
        "best_perplexity": prior_meta["best_perplexity"],
        "hidden_size": prior_cfg["hidden_size"],
        "num_layers": prior_cfg["num_layers"],
        "num_heads": prior_cfg["num_heads"],
        "learning_rate": prior_cfg["learning_rate"],
        "memorization_warnings": len(all_memorization_warnings),
    }

    if json_output:
        print(json_mod.dumps(result_summary, indent=2))
    else:
        console.print()
        console.print("[bold green]Prior training complete![/bold green]")
        console.print(f"  Epochs trained: {prior_meta['epochs_trained']}")
        console.print(f"  Final perplexity: {prior_meta['final_perplexity']:.2f}")
        console.print(f"  Best perplexity:  {prior_meta['best_perplexity']:.2f}")
        console.print()
        console.print(
            f"  [bold]Prior config:[/bold] "
            f"hidden={prior_cfg['hidden_size']}, "
            f"layers={prior_cfg['num_layers']}, "
            f"heads={prior_cfg['num_heads']}, "
            f"lr={prior_cfg['learning_rate']:.1e}"
        )
        console.print(f"  [bold]Model updated:[/bold] {model_path}")

        if all_memorization_warnings:
            console.print()
            console.print("  [bold yellow]Memorization warnings during training:[/bold yellow]")
            for w in all_memorization_warnings:
                console.print(f"    {w}")
