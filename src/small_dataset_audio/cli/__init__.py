"""CLI entry point for Small Dataset Audio.

Provides the ``sda`` command with subcommands: generate, train, model, ui.
Bare ``sda`` (no subcommand) launches the Gradio GUI for backward
compatibility.

Public API:
- app: The Typer application instance
- main: Entry point callable (same as app, used by pyproject.toml console_script)
- bootstrap: Shared config + device loader for all commands
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="sda",
    help="Small Dataset Audio - Generative audio from small personal datasets",
    no_args_is_help=False,
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Module-level shared state (populated by the callback, read by subcommands)
# ---------------------------------------------------------------------------

_cli_state: dict = {}


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------


def bootstrap(
    config_path: Path | None = None,
    device: str = "auto",
) -> tuple:
    """Load config and select device.  Shared by all commands.

    Returns
    -------
    tuple of (config_dict, torch_device, config_path)
    """
    from small_dataset_audio.config.settings import get_config_path, load_config
    from small_dataset_audio.hardware.device import select_device

    path = config_path or get_config_path()
    config = load_config(path)

    # Resolve device: CLI flag overrides config
    device_preference = device
    if device_preference == "auto":
        config_device = config.get("hardware", {}).get("device", "auto")
        if config_device != "auto":
            device_preference = config_device

    torch_device = select_device(device_preference)
    return config, torch_device, path


# ---------------------------------------------------------------------------
# App-level callback (runs before any subcommand, or alone if no subcommand)
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    device: str = typer.Option(
        "auto",
        "--device",
        help="Compute device: auto, mps, cuda, cpu",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed startup information",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Override config file path",
    ),
) -> None:
    """Small Dataset Audio - Generative audio from small personal datasets."""
    # Store global options for subcommands
    _cli_state["device"] = device
    _cli_state["verbose"] = verbose
    _cli_state["config"] = config

    # If no subcommand was given, launch the GUI (backward compatible)
    if ctx.invoked_subcommand is None:
        _launch_gui(device=device, verbose=verbose, config_path=config)


# ---------------------------------------------------------------------------
# Default action: launch GUI (same as legacy app.py:main)
# ---------------------------------------------------------------------------


def _launch_gui(
    *,
    device: str = "auto",
    verbose: bool = False,
    config_path: Path | None = None,
) -> None:
    """Run the full legacy startup flow and launch Gradio."""
    from rich.console import Console

    from small_dataset_audio import __version__
    from small_dataset_audio.app import first_run_setup
    from small_dataset_audio.config.settings import (
        get_config_path,
        load_config,
        resolve_path,
    )
    from small_dataset_audio.hardware.device import (
        format_device_report,
        get_device_info,
        select_device,
    )
    from small_dataset_audio.ui import launch_ui
    from small_dataset_audio.validation.startup import run_startup_validation

    console = Console()

    # Determine config path
    path = config_path or get_config_path()

    # First-run check
    if not path.exists():
        config = first_run_setup(path)
    else:
        config = load_config(path)
        if not config.get("general", {}).get("first_run_complete", False):
            config = first_run_setup(path)

    # Startup validation
    if not run_startup_validation(config, verbose=verbose):
        console.print(
            "\n[bold red]Startup validation failed.[/bold red] "
            "Fix the issues above and try again."
        )
        sys.exit(1)

    # Device selection
    device_preference = device
    if device_preference == "auto":
        config_device = config.get("hardware", {}).get("device", "auto")
        if config_device != "auto":
            device_preference = config_device

    torch_device = select_device(device_preference)
    info = get_device_info(torch_device)

    report = format_device_report(torch_device, info, verbose=verbose)
    console.print(f"[bold]Device:[/bold] {report}")

    # Ensure data directories exist
    for _key, path_str in config.get("paths", {}).items():
        resolved = resolve_path(path_str, base_dir=path.parent)
        resolved.mkdir(parents=True, exist_ok=True)

    # Launch
    console.print()
    console.print(
        f"[bold green]Small Dataset Audio v{__version__} ready.[/bold green] "
        f"({torch_device.type.upper()})"
    )
    console.print("Launching Gradio UI...")
    launch_ui(config=config, device=torch_device)


# ---------------------------------------------------------------------------
# Register sub-typers (gracefully skip missing modules during development)
# ---------------------------------------------------------------------------

try:
    from small_dataset_audio.cli.ui import app as ui_app

    app.add_typer(ui_app, name="ui", help="Launch the Gradio web UI")
except ImportError:
    pass

try:
    from small_dataset_audio.cli.generate import app as generate_app

    app.add_typer(generate_app, name="generate", help="Generate audio from trained models")
except ImportError:
    pass

try:
    from small_dataset_audio.cli.train import app as train_app

    app.add_typer(train_app, name="train", help="Train models on audio datasets")
except ImportError:
    pass

try:
    from small_dataset_audio.cli.model import app as model_app

    app.add_typer(model_app, name="model", help="Manage saved models")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Entry point callable
# ---------------------------------------------------------------------------

main = app

__all__ = ["app", "main", "bootstrap"]
