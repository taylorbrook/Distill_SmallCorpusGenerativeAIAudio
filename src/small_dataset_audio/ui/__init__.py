"""User interfaces (Gradio, CLI).

Public API:
- launch_ui: Start the Gradio web UI
- create_app: Build the Gradio Blocks app (without launching)
"""


def launch_ui() -> None:
    """Launch the Gradio web UI."""
    from small_dataset_audio.ui.app import launch_ui as _launch

    _launch()


def create_app():  # noqa: ANN201
    """Build the Gradio Blocks application."""
    from small_dataset_audio.ui.app import create_app as _create

    return _create()


__all__ = ["launch_ui", "create_app"]
