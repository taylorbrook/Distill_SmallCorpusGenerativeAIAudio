"""Application bootstrap, first-run experience, and CLI entry point.

This module ties together configuration, hardware detection, and
environment validation into the user-facing application.  It provides:

* CLI argument parsing (``--device``, ``--verbose``, ``--benchmark``,
  ``--config``)
* Guided first-run setup
* Every-launch environment validation
* Device selection and reporting

After this module, ``python -m distill`` (or the ``distill``
console script) launches a working application.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with ``device``, ``verbose``, ``benchmark``,
        and ``config`` attributes.
    """
    parser = argparse.ArgumentParser(
        prog="distill",
        description="Distill - Small custom-corpus generative AI audio",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed startup information",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run hardware benchmark and exit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Override config file path (default: config.toml in project root)",
    )
    return parser.parse_args(argv)


def first_run_setup(config_path: Path) -> dict:
    """Guided first-run experience.

    Walks the user through initial configuration: data directory
    paths, device detection, and an optional hardware benchmark.

    Per locked decision: "Guided first-run experience: walk user
    through initial config (paths, device check, create directories)."

    Args:
        config_path: Path where config.toml will be saved.

    Returns:
        The populated configuration dictionary (already saved to disk).
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table

    from distill import __version__
    from distill.config.defaults import DEFAULT_CONFIG
    from distill.config.settings import resolve_path, save_config
    from distill.hardware.device import (
        format_device_report,
        get_device_info,
        select_device,
    )

    console = Console()
    config = DEFAULT_CONFIG.copy()
    config["general"] = DEFAULT_CONFIG["general"].copy()
    config["paths"] = DEFAULT_CONFIG["paths"].copy()
    config["hardware"] = DEFAULT_CONFIG["hardware"].copy()

    # Welcome
    console.print()
    console.print(
        Panel(
            f"[bold]Distill v{__version__}[/bold]\n"
            "Small custom-corpus generative AI audio\n\n"
            "This appears to be your first run. "
            "Let's configure a few things.",
            title="Welcome",
            border_style="green",
        )
    )
    console.print()

    # Data directory paths
    console.print("[bold]Data Directories[/bold]")
    console.print("Where should datasets, models, and generated audio be stored?")
    console.print()

    for key in ("datasets", "models", "generated"):
        default = DEFAULT_CONFIG["paths"][key]
        path_str = Prompt.ask(
            f"  {key} directory",
            default=default,
        )
        config["paths"][key] = path_str

    console.print()

    # Create directories
    console.print("Creating data directories...")
    for key, path_str in config["paths"].items():
        resolved = resolve_path(path_str, base_dir=config_path.parent)
        resolved.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]v[/green] {resolved}")
    console.print()

    # Device detection
    console.print("[bold]Device Detection[/bold]")
    device = select_device("auto")
    info = get_device_info(device)
    report = format_device_report(device, info, verbose=True)
    console.print(f"  Detected: {report}")
    config["hardware"]["device"] = device.type if device.type != "cpu" else "auto"
    console.print()

    # Optional benchmark
    if device.type != "cpu":
        run_bench = Confirm.ask(
            "  Run hardware benchmark? (recommended for GPU)",
            default=True,
        )
    else:
        run_bench = Confirm.ask(
            "  Run hardware benchmark? (CPU-only, uses default batch size)",
            default=False,
        )

    if run_bench:
        from distill.hardware.benchmark import run_benchmark

        console.print()
        console.print("  Running benchmark...")
        result = run_benchmark(device, verbose=True)
        config["hardware"]["max_batch_size"] = result["max_batch_size"]
        config["hardware"]["memory_limit_gb"] = result["memory_total_gb"]
        console.print()

    # Mark first run complete
    config["general"]["first_run_complete"] = True

    # Save config
    save_config(config, config_path)
    console.print(f"[green]v[/green] Configuration saved to {config_path}")
    console.print()

    # Summary table
    table = Table(title="Configuration Summary")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Config file", str(config_path))
    for key, path_str in config["paths"].items():
        resolved = resolve_path(path_str, base_dir=config_path.parent)
        table.add_row(f"Path: {key}", str(resolved))
    table.add_row("Device", info["name"])
    if config["hardware"]["max_batch_size"] > 0:
        table.add_row("Max batch size", str(config["hardware"]["max_batch_size"]))

    console.print(table)
    console.print()

    return config


def main(argv: list[str] | None = None) -> None:
    """Application entry point.

    This is called by the ``distill`` console script and by
    ``python -m distill``.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).
    """
    from rich.console import Console

    from distill import __version__
    from distill.config.settings import (
        get_config_path,
        load_config,
        resolve_path,
        save_config,
    )
    from distill.hardware.benchmark import run_benchmark
    from distill.hardware.device import (
        format_device_report,
        get_device_info,
        select_device,
    )
    from distill.validation.startup import run_startup_validation

    args = parse_args(argv)
    console = Console()

    # Configure logging so training/inference messages are visible
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root_logger.addHandler(handler)
    # Ensure distill loggers propagate
    logging.getLogger("distill").setLevel(log_level)

    # Determine config path
    config_path = args.config if args.config else get_config_path()

    # First-run check
    if not config_path.exists():
        config = first_run_setup(config_path)
    else:
        config = load_config(config_path)
        if not config.get("general", {}).get("first_run_complete", False):
            config = first_run_setup(config_path)

    # Run startup validation (every launch)
    if not run_startup_validation(config, verbose=args.verbose):
        console.print(
            "\n[bold red]Startup validation failed.[/bold red] "
            "Fix the issues above and try again."
        )
        sys.exit(1)

    # Select device (CLI override > config > auto)
    device_preference = args.device
    if device_preference == "auto":
        # Check if config has a non-auto device preference
        config_device = config.get("hardware", {}).get("device", "auto")
        if config_device != "auto":
            device_preference = config_device

    device = select_device(device_preference)
    info = get_device_info(device)

    # Display device report (always, per locked decision)
    report = format_device_report(device, info, verbose=args.verbose)
    console.print(f"[bold]Device:[/bold] {report}")

    # Benchmark mode
    if args.benchmark:
        console.print()
        console.print("[bold]Running hardware benchmark...[/bold]")
        result = run_benchmark(device, verbose=args.verbose)
        config["hardware"]["max_batch_size"] = result["max_batch_size"]
        config["hardware"]["memory_limit_gb"] = result["memory_total_gb"]
        save_config(config, config_path)
        console.print()
        console.print(
            f"[green]v[/green] Benchmark results saved to {config_path}"
        )
        return

    # Ensure data directories exist (non-first-run path)
    for key, path_str in config.get("paths", {}).items():
        resolved = resolve_path(path_str, base_dir=config_path.parent)
        resolved.mkdir(parents=True, exist_ok=True)

    # Launch the Gradio UI
    console.print()
    console.print(
        f"[bold green]Distill v{__version__} ready.[/bold green] "
        f"({device.type.upper()})"
    )
    console.print("Launching Gradio UI...")

    from distill.ui import launch_ui

    launch_ui(config=config, device=device)
