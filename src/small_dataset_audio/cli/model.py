"""``sda model`` commands -- list, info, and delete saved models.

Provides model library management from the terminal with Rich-formatted
tables and JSON output support.

Design notes:
- ``Console(stderr=True)`` for all Rich output (tables/status to stderr).
- ``print()`` for machine-readable output to stdout (JSON data).
- All heavy imports lazy inside command functions (project pattern).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer()


# ---------------------------------------------------------------------------
# Helper: find model entry by name or ID
# ---------------------------------------------------------------------------


def _find_model_entry(
    model_ref: str,
    models_dir: Path,
) -> "ModelEntry":
    """Find a model entry by name or ID.

    Parameters
    ----------
    model_ref : str
        Model name or ID string.
    models_dir : Path
        Directory containing saved models.

    Returns
    -------
    ModelEntry
        The matched entry.

    Raises
    ------
    typer.BadParameter
        If model is not found or reference is ambiguous.
    """
    from small_dataset_audio.library.catalog import ModelLibrary

    library = ModelLibrary(models_dir)

    # Try ID lookup first
    entry = library.get(model_ref)
    if entry is not None:
        return entry

    # Try name search
    results = library.search(query=model_ref)
    if len(results) == 1:
        return results[0]
    if len(results) > 1:
        names = ", ".join(f"'{r.name}'" for r in results)
        raise typer.BadParameter(
            f"Ambiguous model reference '{model_ref}'. "
            f"Matches: {names}. Use the model ID to be specific."
        )

    raise typer.BadParameter(f"Model not found: {model_ref}")


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


@app.command("list")
def list_models(
    device: str = typer.Option("auto", "--device", help="Compute device"),
    config: Annotated[
        Optional[Path], typer.Option("--config", help="Config file path")
    ] = None,
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON to stdout"
    ),
) -> None:
    """List all models in the library."""
    from rich.console import Console
    from rich.table import Table

    from small_dataset_audio.cli import bootstrap
    from small_dataset_audio.config.settings import resolve_path
    from small_dataset_audio.library.catalog import ModelLibrary

    console = Console(stderr=True)

    # Bootstrap config
    app_config, _torch_device, config_path = bootstrap(config, device)

    # Resolve models directory
    models_dir = resolve_path(
        app_config["paths"]["models"], base_dir=config_path.parent
    )

    # Load library
    library = ModelLibrary(models_dir)
    entries = library.list_all()

    # JSON output
    if json_output:
        print(json.dumps([asdict(e) for e in entries], indent=2, default=str))
        return

    # No models
    if not entries:
        console.print("No models found in library.")
        return

    # Rich table
    table = Table(title="Model Library")
    table.add_column("Name", style="bold")
    table.add_column("Dataset")
    table.add_column("Epochs", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Saved")
    table.add_column("ID", style="dim")

    for entry in entries:
        table.add_row(
            entry.name,
            entry.dataset_name,
            str(entry.training_epochs),
            f"{entry.final_val_loss:.4f}",
            entry.save_date[:10] if entry.save_date else "",
            entry.model_id[:8],
        )

    console.print(table)


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


@app.command("info")
def model_info(
    model: str = typer.Argument(..., help="Model name, ID, or .sda file path"),
    device: str = typer.Option("auto", "--device", help="Compute device"),
    config: Annotated[
        Optional[Path], typer.Option("--config", help="Config file path")
    ] = None,
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON to stdout"
    ),
) -> None:
    """Show detailed information about a model."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from small_dataset_audio.cli import bootstrap
    from small_dataset_audio.config.settings import resolve_path

    console = Console(stderr=True)

    # Bootstrap config
    app_config, _torch_device, config_path = bootstrap(config, device)

    # Resolve models directory
    models_dir = resolve_path(
        app_config["paths"]["models"], base_dir=config_path.parent
    )

    # Find model
    try:
        entry = _find_model_entry(model, models_dir)
    except typer.BadParameter as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(2)

    # JSON output
    if json_output:
        print(json.dumps(asdict(entry), indent=2, default=str))
        return

    # Format file size
    size_bytes = entry.file_size_bytes
    if size_bytes >= 1_000_000:
        size_str = f"{size_bytes / 1_000_000:.1f} MB"
    elif size_bytes >= 1_000:
        size_str = f"{size_bytes / 1_000:.1f} KB"
    else:
        size_str = f"{size_bytes} B"

    # Build detail table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Name", entry.name)
    table.add_row("Description", entry.description or "(none)")
    table.add_row("Model ID", entry.model_id)
    table.add_row("File", entry.file_path)
    table.add_row("File Size", size_str)
    table.add_row("Dataset", entry.dataset_name or "(unknown)")
    table.add_row("Dataset Files", str(entry.dataset_file_count))
    table.add_row(
        "Dataset Duration",
        f"{entry.dataset_total_duration_s:.1f}s"
        if entry.dataset_total_duration_s > 0
        else "(unknown)",
    )
    table.add_row("Training Date", entry.training_date[:10] if entry.training_date else "(unknown)")
    table.add_row("Save Date", entry.save_date[:10] if entry.save_date else "(unknown)")
    table.add_row("Epochs", str(entry.training_epochs))
    table.add_row("Train Loss", f"{entry.final_train_loss:.4f}")
    table.add_row("Val Loss", f"{entry.final_val_loss:.4f}")
    table.add_row("Has Analysis", "Yes" if entry.has_analysis else "No")
    table.add_row("Active PCA Components", str(entry.n_active_components))
    table.add_row("Tags", ", ".join(entry.tags) if entry.tags else "(none)")

    panel = Panel(table, title=f"[bold]{entry.name}[/bold]", expand=False)
    console.print(panel)


# ---------------------------------------------------------------------------
# delete command
# ---------------------------------------------------------------------------


@app.command("delete")
def delete_model_cmd(
    model: str = typer.Argument(..., help="Model name or ID"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation"
    ),
    device: str = typer.Option("auto", "--device", help="Compute device"),
    config: Annotated[
        Optional[Path], typer.Option("--config", help="Config file path")
    ] = None,
) -> None:
    """Delete a model from the library."""
    from rich.console import Console

    from small_dataset_audio.cli import bootstrap
    from small_dataset_audio.config.settings import resolve_path
    from small_dataset_audio.models.persistence import delete_model

    console = Console(stderr=True)

    # Bootstrap config
    app_config, _torch_device, config_path = bootstrap(config, device)

    # Resolve models directory
    models_dir = resolve_path(
        app_config["paths"]["models"], base_dir=config_path.parent
    )

    # Find model
    try:
        entry = _find_model_entry(model, models_dir)
    except typer.BadParameter as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(2)

    # Confirmation
    if not force:
        typer.confirm(
            f"Delete model '{entry.name}' ({entry.model_id[:8]})? "
            "This cannot be undone.",
            abort=True,
        )

    # Delete
    success = delete_model(entry.model_id, models_dir)
    if success:
        console.print(
            f"[green]Deleted:[/green] {entry.name} ({entry.model_id[:8]})"
        )
    else:
        console.print(f"[red]Error:[/red] Failed to delete model.")
        raise SystemExit(1)
