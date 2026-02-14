"""Gradio Blocks application assembly.

Builds the 4-tab layout (Data, Train, Generate, Library), wires tab
builders, and provides the :func:`launch_ui` entry point.
"""

from __future__ import annotations

import gradio as gr

from small_dataset_audio.ui.components.guided_nav import get_empty_state_message
from small_dataset_audio.ui.tabs.data_tab import build_data_tab
from small_dataset_audio.ui.tabs.train_tab import build_train_tab


def create_app() -> gr.Blocks:
    """Build the complete Gradio Blocks application.

    Initialises application state from config before assembling the
    layout.  Returns the Blocks object (does not launch).
    """
    from small_dataset_audio.config.settings import load_config
    from small_dataset_audio.hardware.device import select_device
    from small_dataset_audio.ui.state import init_state

    config = load_config()
    device = select_device(config.get("hardware", {}).get("device", "auto"))
    init_state(config, device)

    with gr.Blocks(
        title="Small Dataset Audio",
        fill_width=True,
    ) as app:
        gr.Markdown("# Small Dataset Audio")

        with gr.Tabs():
            with gr.Tab("Data", id="data"):
                build_data_tab()
            with gr.Tab("Train", id="train"):
                build_train_tab()
            with gr.Tab("Generate", id="generate"):
                gr.Markdown(get_empty_state_message("generate"))
            with gr.Tab("Library", id="library"):
                gr.Markdown(get_empty_state_message("library"))

    return app


def launch_ui() -> None:
    """Load config, select device, build app, and launch in browser."""
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
