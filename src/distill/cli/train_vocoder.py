"""CLI command for training per-model HiFi-GAN V2 vocoder.

Provides ``distill train-vocoder MODEL_PATH AUDIO_DIR`` with Rich live
progress display, SIGINT graceful cancellation, and resume support.

Calls VocoderTrainer.train() directly (not via background thread -- CLI
is the main process, same pattern as distill train).

All heavy imports lazy inside the command function body for fast --help.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def train_vocoder_cmd(
    model_path: Path = typer.Argument(
        ...,
        help="Path to .distillgan model file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    audio_dir: Path = typer.Argument(
        ...,
        help="Path to training audio directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    epochs: int = typer.Option(200, "--epochs", "-e", help="Training epochs"),
    learning_rate: float = typer.Option(0.0002, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size"),
    checkpoint_interval: int = typer.Option(
        50, "--checkpoint-interval", help="Save checkpoint every N epochs"
    ),
    device: str = typer.Option(
        "auto", "--device", help="Compute device: auto, cuda, mps, cpu"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from existing checkpoint"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Replace existing vocoder without confirmation"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON"
    ),
) -> None:
    """Train a per-model HiFi-GAN V2 vocoder on training audio."""
    # All heavy imports here (lazy)
    import json as json_mod
    import signal
    import threading

    from rich.console import Console
    from rich.live import Live
    from rich.table import Table

    from distill.models.persistence import load_model
    from distill.vocoder.hifigan import (
        HiFiGANConfig,
        VocoderEpochMetrics,
        VocoderPreviewEvent,
        VocoderTrainer,
        VocoderTrainingCompleteEvent,
    )

    console = Console(stderr=True)

    # 1. Load model and validate
    console.print(f"Loading model: {model_path.name}")
    loaded = load_model(model_path, device=device)

    # 2. Check for existing vocoder
    if loaded.vocoder_state is not None:
        if loaded.vocoder_state.get("checkpoint") and not resume:
            # Has checkpoint -- ask about resume
            epoch = loaded.vocoder_state["checkpoint"]["epoch"]
            if not force:
                console.print(
                    f"[yellow]Found checkpoint at epoch {epoch}.[/yellow]"
                )
                choice = typer.prompt(
                    "Resume from checkpoint or start fresh?",
                    type=str,
                    default="resume",
                )
                if choice.lower().startswith("r"):
                    resume = True
        elif not loaded.vocoder_state.get("checkpoint"):
            # Has completed vocoder -- warn about replacement
            if not force:
                epochs_trained = (
                    loaded.vocoder_state.get("training_metadata", {}).get(
                        "epochs", "?"
                    )
                )
                console.print(
                    f"[yellow]This model already has a trained vocoder "
                    f"({epochs_trained} epochs).[/yellow]"
                )
                if not typer.confirm("Replace existing vocoder?"):
                    raise typer.Abort()

    # 3. Create config from CLI parameters
    config = HiFiGANConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # 4. Set up SIGINT handler for graceful cancel
    cancel_event = threading.Event()
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(sig, frame):
        console.print("\n[yellow]Cancelling... saving checkpoint...[/yellow]")
        cancel_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # 5. Set up Rich Live table progress display
    #    Uses Live(Table) pattern -- table updates in-place each epoch (no scroll).

    def _build_table(epoch, total, gen_loss, disc_loss, mel_loss, lr, eta, pct):
        """Build a Rich Table showing current training metrics."""
        table = Table(title="HiFi-GAN Vocoder Training", expand=True)
        table.add_column("Epoch", justify="right", style="bold blue", width=14)
        table.add_column("Gen Loss", justify="right", width=12)
        table.add_column("Disc Loss", justify="right", width=12)
        table.add_column("Mel Loss", justify="right", width=12)
        table.add_column("LR", justify="right", width=12)
        table.add_column("ETA", justify="right", width=10)
        eta_str = f"{eta:.0f}s" if eta > 0 else "\u2014"
        table.add_row(
            f"{epoch}/{total}",
            f"{gen_loss:.4f}",
            f"{disc_loss:.4f}",
            f"{mel_loss:.4f}",
            f"{lr:.6f}",
            eta_str,
        )
        return table

    live_context = Live(
        _build_table(0, epochs, 0.0, 0.0, 0.0, learning_rate, 0.0, 0.0),
        console=console,
        refresh_per_second=4,
    )

    def _callback(event):
        if isinstance(event, VocoderEpochMetrics):
            live_context.update(
                _build_table(
                    event.epoch + 1,
                    event.total_epochs,
                    event.gen_loss,
                    event.disc_loss,
                    event.mel_loss,
                    event.learning_rate,
                    event.eta_seconds,
                    (event.epoch + 1) / event.total_epochs,
                )
            )
        elif isinstance(event, VocoderPreviewEvent):
            # Save preview WAV to disk (useful for headless/SSH training)
            import soundfile as sf

            preview_path = (
                model_path.parent
                / f"vocoder_preview_epoch{event.epoch:04d}.wav"
            )
            sf.write(str(preview_path), event.audio, event.sample_rate)
            console.print(
                f"  [dim]Preview saved: {preview_path.name}[/dim]"
            )
        elif isinstance(event, VocoderTrainingCompleteEvent):
            console.print(
                f"\n[bold green]Training complete![/bold green] "
                f"{event.epochs_completed} epochs, "
                f"gen_loss={event.final_gen_loss:.4f}, "
                f"disc_loss={event.final_disc_loss:.4f}"
            )

    # 6. Run training
    console.print(f"\nTraining HiFi-GAN V2 vocoder on {audio_dir}")
    console.print(f"  Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}")
    console.print()

    trainer = VocoderTrainer(config, device=device)
    # Pass the full vocoder_state dict -- trainer extracts checkpoint internally
    checkpoint_data = loaded.vocoder_state if resume and loaded.vocoder_state else None

    try:
        with live_context:
            result = trainer.train(
                model_path=model_path,
                audio_dir=audio_dir,
                callback=_callback,
                cancel_event=cancel_event,
                checkpoint=checkpoint_data,
                epochs=epochs,
                preview_interval=checkpoint_interval,
            )
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    # 7. Report results
    if cancel_event.is_set():
        saved_epoch = result.get("checkpoint", {}).get("epoch", "?")
        console.print(
            f"\n[yellow]Checkpoint saved at epoch {saved_epoch}. "
            f"Resume anytime.[/yellow]"
        )
    else:
        console.print(f"\nVocoder saved to: {model_path}")

    # 8. JSON output
    if json_output:
        output = {
            "model_path": str(model_path),
            "audio_dir": str(audio_dir),
            "cancelled": cancel_event.is_set(),
            "epochs_completed": result.get("training_metadata", {}).get(
                "epochs", 0
            ),
            "final_gen_loss": result.get("training_metadata", {}).get(
                "final_loss", None
            ),
        }
        typer.echo(json_mod.dumps(output, indent=2))
