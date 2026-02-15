"""Guided navigation helpers and empty state messages.

Provides empty-state messages for tabs whose prerequisites are not
yet met, and convenience checks on :data:`app_state` to determine
readiness.
"""

from __future__ import annotations

_EMPTY_STATE_MESSAGES: dict[str, str] = {
    "data": (
        "## Import Your Audio Dataset\n\n"
        "Drag and drop audio files above or use the folder browse "
        "button to import a dataset."
    ),
    "train": (
        "## Import a Dataset to Start Training\n\n"
        "Go to the **Data** tab to import audio files first."
    ),
    "generate": (
        "## Load a Model to Start Generating\n\n"
        "Go to the **Library** tab to load a trained model, "
        "or **Train** a new one first."
    ),
    "library": (
        "## No Models Yet\n\n"
        "Train a model on the **Train** tab, then save it to "
        "build your library."
    ),
}


def get_empty_state_message(tab_name: str) -> str:
    """Return Markdown text for an empty/unready tab.

    Args:
        tab_name: One of ``"data"``, ``"train"``, ``"generate"``,
                  ``"library"``.

    Returns:
        Markdown string with guidance on what the user should do.
    """
    return _EMPTY_STATE_MESSAGES.get(
        tab_name,
        f"## {tab_name.title()}\n\nThis tab is not yet available.",
    )


def has_dataset() -> bool:
    """Check whether a dataset is currently loaded."""
    from distill.ui.state import app_state

    return app_state.current_dataset is not None


def has_model() -> bool:
    """Check whether a model is currently loaded."""
    from distill.ui.state import app_state

    return app_state.loaded_model is not None


def has_training_data() -> bool:
    """Check whether data and model exist for training."""
    return has_dataset()
