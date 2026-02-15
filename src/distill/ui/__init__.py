"""User interfaces (Gradio, CLI).

Public API:
- launch_ui: Start the Gradio web UI
- create_app: Build the Gradio Blocks app (without launching)
"""

from __future__ import annotations

from typing import Any


def launch_ui(
    config: dict[str, Any] | None = None,
    device: Any = None,
) -> None:
    """Launch the Gradio web UI.

    Parameters
    ----------
    config : dict | None
        Pre-loaded config dict.  If ``None``, ``create_app`` loads fresh
        from disk.
    device : torch.device | None
        Pre-selected device.  If ``None``, ``create_app`` auto-detects.
    """
    from distill.ui.app import launch_ui as _launch

    _launch(config=config, device=device)


def create_app(
    config: dict[str, Any] | None = None,
    device: Any = None,
):  # noqa: ANN201
    """Build the Gradio Blocks application.

    Parameters
    ----------
    config : dict | None
        Pre-loaded config dict.
    device : torch.device | None
        Pre-selected device.
    """
    from distill.ui.app import create_app as _create

    return _create(config=config, device=device)


__all__ = ["launch_ui", "create_app"]
