"""Gradio Blocks application assembly.

Builds the 4-tab layout (Data, Train, Generate, Library), wires tab
builders, and provides the :func:`launch_ui` entry point.  Cross-tab
wiring connects the Library load action to Generate tab slider updates.
"""

from __future__ import annotations

import gradio as gr

from small_dataset_audio.ui.tabs.data_tab import build_data_tab
from small_dataset_audio.ui.tabs.generate_tab import build_generate_tab
from small_dataset_audio.ui.tabs.library_tab import build_library_tab
from small_dataset_audio.ui.tabs.train_tab import build_train_tab


def create_app() -> gr.Blocks:
    """Build the complete Gradio Blocks application.

    Initialises application state from config before assembling the
    layout.  After all tabs are built, wires cross-tab events so that
    loading a model from the Library also updates the Generate tab
    sliders.  Returns the Blocks object (does not launch).
    """
    from small_dataset_audio.config.settings import load_config
    from small_dataset_audio.hardware.device import select_device
    from small_dataset_audio.ui.state import init_state
    from small_dataset_audio.ui.tabs.generate_tab import _update_sliders_for_model

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
                gen_refs = build_generate_tab()
            with gr.Tab("Library", id="library"):
                lib_refs = build_library_tab()

        # -- Cross-tab wiring --
        # After loading a model from Library, update Generate tab sliders.
        # The Library load handler sets app_state, then this chained event
        # reads app_state and updates slider visibility/labels.
        lib_refs["load_btn"].click(
            fn=_update_sliders_for_model,
            inputs=None,
            outputs=(
                gen_refs["sliders"]
                + [
                    gen_refs["preset_dd"],
                    gen_refs["controls_section"],
                    gen_refs["empty_msg"],
                ]
            ),
        )

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
