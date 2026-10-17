"""Gradio Blocks application assembly.

Builds the 4-tab layout (Data, Train, Generate, Library), wires tab
builders, and provides the :func:`launch_ui` entry point.  Cross-tab
wiring connects:

- Library load -> Generate tab slider updates
- Library load -> Generate tab blend model dropdown refresh
- Data import -> Train tab empty-state/controls visibility
- Library load -> Generate tab empty-state/controls visibility

Accepts optional pre-loaded ``config`` and ``device`` so the main
application entry point (``distill`` command) can pass through startup
results without duplicating config load and device detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from distill.ui.tabs.data_tab import build_data_tab
from distill.ui.tabs.generate_tab import build_generate_tab
from distill.ui.tabs.library_tab import build_library_tab
from distill.ui.tabs.train_tab import build_train_tab


def create_app(
    config: dict[str, Any] | None = None,
    device: Any = None,
) -> gr.Blocks:
    """Build the complete Gradio Blocks application.

    Initialises application state from config before assembling the
    layout.  After all tabs are built, wires cross-tab events so that
    loading a model from the Library also updates the Generate tab
    sliders.  Returns the Blocks object (does not launch).

    Parameters
    ----------
    config : dict | None
        Pre-loaded config dict.  If ``None``, loads from disk.
    device : torch.device | None
        Pre-selected device.  If ``None``, auto-detects.
    """
    from distill.config.settings import load_config
    from distill.hardware.device import select_device
    from distill.ui.state import init_state
    from distill.ui.tabs.generate_tab import (
        _update_sliders_for_model,
        _refresh_blend_model_choices,
    )

    if config is None:
        config = load_config()
    if device is None:
        device = select_device(config.get("hardware", {}).get("device", "auto"))
    init_state(config, device)

    with gr.Blocks(
        title="Distill",
        fill_width=True,
    ) as app:
        gr.Markdown("# Distill")

        with gr.Tabs():
            with gr.Tab("Data", id="data"):
                data_refs = build_data_tab()
            with gr.Tab("Train", id="train"):
                train_refs = build_train_tab()
            with gr.Tab("Generate", id="generate"):
                gen_refs = build_generate_tab()
            with gr.Tab("Library", id="library"):
                lib_refs = build_library_tab()

        # -- Cross-tab wiring --

        # After importing files in Data tab, update Train tab visibility.
        # The Data import handler sets app_state.current_dataset, then this
        # chained .then() reads it and toggles empty-state / training UI.
        _train_outputs = [
            train_refs["empty_state"],
            train_refs["train_ui"],
            train_refs["resume_btn"],
        ]
        data_refs["file_upload_event"].then(
            fn=train_refs["check_dataset_ready"],
            inputs=None,
            outputs=_train_outputs,
        )
        data_refs["folder_upload_event"].then(
            fn=train_refs["check_dataset_ready"],
            inputs=None,
            outputs=_train_outputs,
        )
        data_refs["clear_event"].then(
            fn=train_refs["check_dataset_ready"],
            inputs=None,
            outputs=_train_outputs,
        )

        # After loading a model from Library, update Generate tab sliders.
        # The Library load handler sets app_state, then this chained event
        # reads app_state and updates slider visibility/labels.  It must be
        # chained with .then(): a separate listener on the button can run
        # before the model has loaded and leave the sliders empty.
        _gen_slider_outputs = (
            gen_refs["sliders"]
            + [
                gen_refs["preset_dd"],
                gen_refs["controls_section"],
                gen_refs["empty_msg"],
            ]
        )

        lib_refs["load_event"].then(
            fn=_update_sliders_for_model,
            inputs=None,
            outputs=_gen_slider_outputs,
        ).then(
            fn=_refresh_blend_model_choices,
            inputs=None,
            outputs=gen_refs["blend_model_dds"],
        )

        # Card click -> select in dropdown, load model, update Generate sliders
        from distill.ui.tabs.library_tab import _load_model_handler

        def _card_click_select(evt: gr.EventData):
            """Set dropdown to clicked card name."""
            name = getattr(evt, "model_name", None)
            if not name:
                return gr.update()
            return gr.update(value=name)

        lib_refs["card_html"].click(
            fn=_card_click_select,
            inputs=None,
            outputs=[lib_refs["model_dropdown"]],
        ).then(
            fn=_load_model_handler,
            inputs=[lib_refs["model_dropdown"]],
            outputs=[lib_refs["status_msg"]],
        ).then(
            fn=_update_sliders_for_model,
            inputs=None,
            outputs=_gen_slider_outputs,
        ).then(
            fn=_refresh_blend_model_choices,
            inputs=None,
            outputs=gen_refs["blend_model_dds"],
        )

    return app


def launch_ui(
    config: dict[str, Any] | None = None,
    device: Any = None,
) -> None:
    """Build app and launch in browser.

    Parameters
    ----------
    config : dict | None
        Pre-loaded config dict (forwarded to ``create_app``).
    device : torch.device | None
        Pre-selected device (forwarded to ``create_app``).
    """
    app = create_app(config=config, device=device)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        allowed_paths=allowed_paths(),
    )


def allowed_paths() -> list[str]:
    """Data directories whose files the UI hands to the browser.

    Gradio only serves files under the working directory or its temp dir
    unless listed here, so history audio, thumbnails, training previews
    and exports failed with InvalidPathError when the data directories
    were configured elsewhere.  Call after :func:`create_app`.
    """
    from distill.ui.state import app_state

    dirs: list[Path] = [
        app_state.datasets_dir,
        app_state.models_dir,
        app_state.generated_dir,
        app_state.presets_dir,
        app_state.history_dir,
        app_state.training_dir,
    ]
    return [str(Path(d).resolve()) for d in dirs]
