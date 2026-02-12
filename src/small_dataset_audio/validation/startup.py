"""Full startup validation sequence.

Orchestrates all environment checks and displays results using
``rich`` formatting.  The :func:`run_startup_validation` function is
called on every application launch from :mod:`small_dataset_audio.app`.

Per locked decision: "Report and exit on critical failures -- clear
message with fix instructions, no auto-fixing."
"""

from __future__ import annotations


def run_startup_validation(config: dict, verbose: bool = False) -> bool:
    """Run the full startup validation sequence.

    Calls :func:`~small_dataset_audio.validation.environment.validate_environment`
    and displays results using a ``rich`` console.

    * **Errors** are shown in red with fix instructions.
    * **Warnings** are shown in yellow.
    * **Success** is shown with a green checkmark.

    If *verbose* is ``True``, successful checks also display details
    (Python version, PyTorch version, device info, configured paths).

    Args:
        config: Application configuration dictionary.
        verbose: Show all check details even on success.

    Returns:
        ``True`` if the environment is acceptable (no errors),
        ``False`` if there are fatal errors (caller should exit).
    """
    from rich.console import Console

    from small_dataset_audio.validation.environment import validate_environment

    console = Console(stderr=True)
    errors, warnings = validate_environment(config)

    # Display errors
    if errors:
        console.print()
        console.print("[bold red]Environment validation failed:[/bold red]")
        for error in errors:
            console.print(f"  [red]x[/red] {error}")
        console.print()

    # Display warnings
    if warnings:
        if not errors:
            console.print()
        for warning in warnings:
            console.print(f"  [yellow]![/yellow] {warning}")
        console.print()

    # Display success or verbose details
    if not errors:
        if verbose:
            _print_verbose_details(console, config)
        else:
            console.print("[green]v[/green] Environment OK")
        return True

    return False


def _print_verbose_details(console: Console, config: dict) -> None:
    """Print detailed environment information on successful validation.

    Args:
        console: Rich console for output.
        config: Application configuration dictionary.
    """
    import sys

    from rich.console import Console

    console.print("[green]v[/green] Environment OK (verbose details below)")
    console.print()

    # Python version
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console.print(f"  Python: {version}")

    # PyTorch version
    try:
        import torch

        console.print(f"  PyTorch: {torch.__version__}")
    except ImportError:
        pass

    # TorchAudio version
    try:
        import torchaudio

        console.print(f"  TorchAudio: {torchaudio.__version__}")
    except ImportError:
        pass

    # Configured paths
    from small_dataset_audio.config.settings import resolve_path

    paths_config = config.get("paths", {})
    if paths_config:
        console.print("  Paths:")
        for key, path_str in paths_config.items():
            resolved = resolve_path(path_str)
            exists = "[green]exists[/green]" if resolved.exists() else "[yellow]pending[/yellow]"
            console.print(f"    {key}: {resolved} ({exists})")
