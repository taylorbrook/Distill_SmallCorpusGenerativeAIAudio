"""Library tab: model browsing, loading, deletion, and saving.

Provides dual-view model browsing (card grid default, sortable table
toggle), model loading for generation, deletion, and saving trained
models to the library.

Uses the ``ModelLibrary`` from Phase 6 for catalog management and
``load_model`` / ``delete_model`` / ``save_model_from_checkpoint``
from :mod:`distill.models.persistence`.
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr

from distill.ui.components.guided_nav import get_empty_state_message
from distill.ui.components.model_card import render_model_cards
from distill.ui.state import app_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_models(query: str = "") -> list:
    """Fetch models from library, optionally filtered by search query.

    Returns
    -------
    list[ModelEntry]
        Matching models, sorted by save_date descending.
    """
    if app_state.model_library is None:
        return []
    if query.strip():
        return app_state.model_library.search(query=query.strip())
    return app_state.model_library.list_all()


def _models_to_table(models: list) -> list[list]:
    """Convert ModelEntry list to a list-of-lists for gr.Dataframe.

    Columns: Name, Dataset, Files, Epochs, Date, Components
    """
    rows = []
    for m in models:
        rows.append([
            m.name,
            m.dataset_name or "",
            m.dataset_file_count,
            m.training_epochs,
            m.training_date[:10] if m.training_date else "",
            m.n_active_components,
        ])
    return rows


def _model_dropdown_choices(models: list) -> list[str]:
    """Build dropdown choices from model list (name strings)."""
    return [m.name for m in models]


def _find_model_by_name(name: str):
    """Find a ModelEntry by name from the current library.

    Returns
    -------
    ModelEntry | None
    """
    if not name or app_state.model_library is None:
        return None
    for entry in app_state.model_library.list_all():
        if entry.name == name:
            return entry
    return None


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _refresh_library(query: str = "") -> tuple:
    """Refresh card grid, table, dropdown, and empty/content visibility.

    Returns
    -------
    tuple
        (empty_state_visible, content_visible, card_html,
         table_data, dropdown_choices, status_msg)
    """
    models = _get_models(query)
    card_html = render_model_cards(models)
    table_data = _models_to_table(models)
    choices = _model_dropdown_choices(models)

    has_models = len(models) > 0
    return (
        gr.update(visible=not has_models),  # empty state
        gr.update(visible=has_models),      # content area
        card_html,
        table_data,
        gr.update(choices=choices, value=None),
        "",
    )


def _toggle_view(view_choice: str) -> tuple:
    """Toggle between card grid and table views.

    Returns
    -------
    tuple
        (card_section_visible, table_section_visible)
    """
    is_cards = view_choice == "Cards"
    return gr.update(visible=is_cards), gr.update(visible=not is_cards)


def _search_models(query: str) -> tuple:
    """Search models and refresh both views."""
    return _refresh_library(query)


def _load_model_handler(selected_name: str) -> tuple:
    """Load a model from the library for generation.

    Sets up ``app_state.loaded_model``, ``app_state.pipeline``,
    ``app_state.preset_manager``, and ``app_state.history_store``.

    Returns
    -------
    tuple
        (status_message,)
    """
    if not selected_name:
        return ("No model selected.",)

    entry = _find_model_by_name(selected_name)
    if entry is None:
        return (f"Model '{selected_name}' not found in library.",)

    try:
        from distill.history.store import GenerationHistory
        from distill.inference.generation import GenerationPipeline
        from distill.models.persistence import load_model
        from distill.presets.manager import PresetManager

        model_path = app_state.models_dir / entry.file_path
        device_str = str(app_state.device) if app_state.device else "cpu"
        loaded = load_model(model_path, device=device_str)

        app_state.loaded_model = loaded
        app_state.pipeline = GenerationPipeline(
            model=loaded.model,
            spectrogram=loaded.spectrogram,
            device=loaded.device,
        )
        app_state.preset_manager = PresetManager(
            model_id=entry.model_id,
            presets_dir=app_state.presets_dir,
        )
        app_state.history_dir.mkdir(parents=True, exist_ok=True)
        app_state.history_store = GenerationHistory(
            history_dir=app_state.history_dir,
        )

        logger.info("Loaded model '%s' from library", entry.name)
        return (f"Model loaded: {entry.name}",)

    except Exception as exc:
        logger.error("Failed to load model '%s': %s", selected_name, exc)
        return (f"Failed to load model: {exc}",)


def _delete_model_handler(selected_name: str) -> tuple:
    """Delete a model from the library and disk.

    Returns
    -------
    tuple
        (empty_visible, content_visible, card_html, table_data,
         dropdown, status_msg)
    """
    if not selected_name:
        return _refresh_library("") + ("No model selected.",)  # type: ignore[return-value]

    entry = _find_model_by_name(selected_name)
    if entry is None:
        result = _refresh_library("")
        return (*result[:5], f"Model '{selected_name}' not found.")

    try:
        from distill.models.persistence import delete_model

        delete_model(entry.model_id, app_state.models_dir)

        # If the deleted model was currently loaded, clear state
        if (
            app_state.loaded_model is not None
            and app_state.loaded_model.metadata.model_id == entry.model_id
        ):
            app_state.loaded_model = None
            app_state.pipeline = None
            app_state.preset_manager = None
            app_state.history_store = None

        # Reload the library catalog from disk after deletion
        from distill.library.catalog import ModelLibrary

        app_state.model_library = ModelLibrary(app_state.models_dir)

        result = _refresh_library("")
        return (*result[:5], f"Deleted: {entry.name}")

    except Exception as exc:
        logger.error("Failed to delete model '%s': %s", selected_name, exc)
        result = _refresh_library("")
        return (*result[:5], f"Delete failed: {exc}")


def _save_model_handler(name: str, description: str, tags_str: str) -> tuple:
    """Save the most recently trained model to the library.

    Requires that training has completed (``app_state.training_runner.result``
    contains a ``best_checkpoint_path``).

    Returns
    -------
    tuple
        (empty_visible, content_visible, card_html, table_data,
         dropdown, status_msg)
    """
    if not name or not name.strip():
        result = _refresh_library("")
        return (*result[:5], "Please enter a model name.")

    runner = app_state.training_runner
    if runner is None or runner.result is None:
        result = _refresh_library("")
        return (*result[:5], "No training result available. Train a model first.")

    checkpoint_path = runner.result.get("best_checkpoint_path")
    if checkpoint_path is None:
        result = _refresh_library("")
        return (*result[:5], "No checkpoint found from training.")

    try:
        from distill.models.persistence import (
            ModelMetadata,
            save_model_from_checkpoint,
        )

        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        metadata = ModelMetadata(
            name=name.strip(),
            description=description.strip() if description else "",
            tags=tags,
        )

        # Populate dataset info if available
        if app_state.current_dataset is not None:
            metadata.dataset_name = app_state.current_dataset.name
            metadata.dataset_file_count = len(app_state.current_dataset.valid_files)

        save_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            models_dir=app_state.models_dir,
        )

        # Reload the library catalog from disk after save
        from distill.library.catalog import ModelLibrary

        app_state.model_library = ModelLibrary(app_state.models_dir)

        result = _refresh_library("")
        return (*result[:5], f"Saved: {name.strip()}")

    except Exception as exc:
        logger.error("Failed to save model '%s': %s", name, exc)
        result = _refresh_library("")
        return (*result[:5], f"Save failed: {exc}")


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

_TABLE_HEADERS = ["Name", "Dataset", "Files", "Epochs", "Date", "Components"]


def build_library_tab() -> dict[str, Any]:
    """Build the Library tab UI within the current Blocks context.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``"load_btn"`` key for cross-tab wiring in
        ``app.py``.  Other tabs may need to trigger on model load.
    """
    # -- Empty state (shown when library has no models) --
    empty_state = gr.Markdown(
        value=get_empty_state_message("library"),
        visible=True,
    )

    # -- Content area (shown when library has models) --
    with gr.Column(visible=False) as content_area:
        # Header with search and view toggle
        gr.Markdown("## Model Library")
        with gr.Row():
            search_box = gr.Textbox(
                label="Search",
                placeholder="Search models...",
                scale=3,
            )
            view_toggle = gr.Radio(
                choices=["Cards", "Table"],
                value="Cards",
                label="View",
                scale=1,
            )

        # Save model accordion
        with gr.Accordion("Save Current Model", open=False):
            save_name = gr.Textbox(
                label="Model Name",
                placeholder="My Model",
            )
            save_description = gr.Textbox(
                label="Description",
                placeholder="Optional description",
                lines=2,
            )
            save_tags = gr.Textbox(
                label="Tags",
                placeholder="ambient, texture, pad (comma-separated)",
            )
            save_btn = gr.Button("Save to Library", variant="primary")

        # Hidden textbox for card click -> dropdown selection (JS bridge)
        card_selected_name = gr.Textbox(
            value="",
            visible=False,
            elem_id="model-card-selected-name",
        )

        # Card grid view (default visible)
        with gr.Column(visible=True) as card_section:
            card_html = gr.HTML(value=render_model_cards([]))

        # Table view (hidden by default)
        with gr.Column(visible=False) as table_section:
            table_view = gr.Dataframe(
                headers=_TABLE_HEADERS,
                datatype=["str", "str", "number", "number", "str", "number"],
                interactive=False,
                value=[],
            )

        # Model selection and actions
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=[],
                scale=3,
            )
            load_btn = gr.Button("Load Selected", variant="primary", scale=1)
            delete_btn = gr.Button("Delete Selected", variant="stop", scale=1)

        status_msg = gr.Textbox(
            label="Status",
            interactive=False,
            value="",
        )

    # -- Wire events --

    # Refresh outputs (shared by multiple handlers)
    refresh_outputs = [
        empty_state, content_area, card_html, table_view,
        model_dropdown, status_msg,
    ]

    # View toggle
    view_toggle.change(
        fn=_toggle_view,
        inputs=[view_toggle],
        outputs=[card_section, table_section],
    )

    # Search
    search_box.change(
        fn=_search_models,
        inputs=[search_box],
        outputs=refresh_outputs,
    )

    # Card click -> load handled via cross-tab wiring in app.py

    # Load model
    load_btn.click(
        fn=_load_model_handler,
        inputs=[model_dropdown],
        outputs=[status_msg],
    )

    # Delete model
    delete_btn.click(
        fn=_delete_model_handler,
        inputs=[model_dropdown],
        outputs=refresh_outputs,
    )

    # Save model
    save_btn.click(
        fn=_save_model_handler,
        inputs=[save_name, save_description, save_tags],
        outputs=refresh_outputs,
    )

    # Refresh library on page load so newly trained models appear
    app_context = gr.context.Context.root_block
    if app_context is not None:
        app_context.load(
            fn=lambda: _refresh_library(""),
            inputs=None,
            outputs=refresh_outputs,
        )

    return {
        "load_btn": load_btn,
        "card_selected_name": card_selected_name,
        "model_dropdown": model_dropdown,
        "status_msg": status_msg,
    }
