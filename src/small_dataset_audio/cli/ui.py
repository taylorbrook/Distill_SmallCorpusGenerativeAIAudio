"""``sda ui`` command -- launch the Gradio web UI.

Mirrors the default bare-``sda`` behavior but available explicitly
as a subcommand for clarity in scripts.
"""

from __future__ import annotations

import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def ui(ctx: typer.Context) -> None:
    """Launch the Gradio web UI."""
    from small_dataset_audio.cli import _cli_state, _launch_gui

    device = _cli_state.get("device", "auto")
    verbose = _cli_state.get("verbose", False)
    config_path = _cli_state.get("config")

    _launch_gui(device=device, verbose=verbose, config_path=config_path)
