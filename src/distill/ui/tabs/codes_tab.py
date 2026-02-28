"""Codes tab: encode/decode audio with VQ-VAE, interactive code grid, and audio previews.

Provides the primary interface for understanding how a VQ-VAE represents
audio through discrete codes.  Users can:

- Select a trained VQ-VAE model (v2 format only) from a dropdown
- Upload audio and encode it to see the interactive code grid
- Click any cell to hear that codebook entry previewed
- Click a column header to hear the full time-slice decoded
- Click Play on a row label to hear that level's contribution
- Click Decode to hear the full reconstruction for A/B comparison
- Edit level labels (Structure/Timbre/Detail) via a collapsible accordion

Uses ``app_state`` singleton for model and device access.  Event handlers
are module-level functions (not lambdas) following the generate_tab.py
pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from distill.ui.state import app_state

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum number of level label textboxes (covers up to 4 quantizer levels)
MAX_LEVEL_LABELS = 4

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_current_encode: dict | None = None
"""Stores the latest encode result (indices, spatial_shape, mel_shape, etc.)."""

_current_labels: list[str] | None = None
"""Stores the current level labels for grid re-rendering."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_vqvae_model_choices() -> list[str]:
    """Scan models_dir for VQ-VAE (v2) .distill files and return their names.

    Uses a lightweight peek at each file (torch.load with map_location=cpu,
    reading only version + model_type + metadata) following the
    ``_detect_model_version`` pattern from ``cli/generate.py``.
    """
    import torch  # noqa: WPS433 -- lazy import

    models_dir = app_state.models_dir
    if not models_dir.exists():
        return []

    choices: list[str] = []
    for path in sorted(models_dir.glob("*.distill")):
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            version = saved.get("version", 1)
            model_type = saved.get("model_type", "vae")
            if version >= 2 and model_type == "vqvae":
                meta = saved.get("metadata", {})
                name = meta.get("name", path.stem)
                choices.append(name)
        except Exception:
            logger.debug("Skipping unreadable .distill file: %s", path)
    return choices


def _find_model_path_by_name(name: str) -> Path | None:
    """Find the .distill file path for a VQ-VAE model by its metadata name."""
    import torch  # noqa: WPS433 -- lazy import

    models_dir = app_state.models_dir
    if not models_dir.exists():
        return None

    for path in models_dir.glob("*.distill"):
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            version = saved.get("version", 1)
            model_type = saved.get("model_type", "vae")
            if version >= 2 and model_type == "vqvae":
                meta = saved.get("metadata", {})
                if meta.get("name", path.stem) == name:
                    return path
        except Exception:
            continue
    return None


def _refresh_model_dropdown():
    """Return a gr.update for the model dropdown with current VQ-VAE choices."""
    choices = _get_vqvae_model_choices()
    return gr.update(choices=choices)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _load_model(model_name: str):
    """Load a VQ-VAE model by name from the models directory.

    Stores the loaded model in ``app_state.loaded_vq_model``.

    Returns
    -------
    str
        Status message.
    """
    if not model_name:
        return "No model selected."

    model_path = _find_model_path_by_name(model_name)
    if model_path is None:
        return f"Model '{model_name}' not found."

    try:
        from distill.models.persistence import load_model_v2

        device_str = str(app_state.device) if app_state.device else "cpu"
        loaded = load_model_v2(model_path, device=device_str)
        app_state.loaded_vq_model = loaded
        logger.info("Loaded VQ-VAE model '%s' for Codes tab", model_name)
        return f"Loaded: {model_name}"
    except Exception as exc:
        logger.error("Failed to load VQ-VAE model '%s': %s", model_name, exc)
        return f"Failed to load model: {exc}"


def _encode_audio(audio_path, model_name):
    """Encode uploaded audio through the loaded VQ-VAE model.

    Returns a list of updates for:
    [grid_html, original_audio, decoded_audio, label_0..label_N, status]
    """
    global _current_encode, _current_labels  # noqa: WPS420

    # Guards
    if app_state.loaded_vq_model is None:
        empty = [gr.update()] * (3 + MAX_LEVEL_LABELS) + ["No model loaded. Select a model first."]
        return empty

    if audio_path is None or (isinstance(audio_path, str) and not audio_path.strip()):
        empty = [gr.update()] * (3 + MAX_LEVEL_LABELS) + ["No audio file uploaded."]
        return empty

    try:
        from distill.inference.codes import encode_audio_file, decode_code_grid
        from distill.ui.components.code_grid import render_code_grid, get_level_labels

        # Encode
        result = encode_audio_file(Path(audio_path), app_state.loaded_vq_model)
        _current_encode = result

        # Get level labels
        labels = get_level_labels(result["num_quantizers"])
        _current_labels = labels

        # Render grid
        grid_html = render_code_grid(
            indices=result["indices"],
            num_quantizers=result["num_quantizers"],
            codebook_size=result["codebook_size"],
            spatial_shape=result["spatial_shape"],
            level_labels=labels,
            selected_cell=None,
            duration_s=result["duration_s"],
        )

        # Auto-decode for immediate A/B comparison
        wav_array = decode_code_grid(
            result["indices"],
            result["spatial_shape"],
            result["mel_shape"],
            app_state.loaded_vq_model,
        )

        # Build return list: grid_html, original_audio, decoded_audio
        updates: list = [
            grid_html,                   # grid_html
            audio_path,                  # original audio player
            (48000, wav_array),          # decoded audio player
        ]

        # Label textbox updates
        num_q = result["num_quantizers"]
        for i in range(MAX_LEVEL_LABELS):
            if i < num_q:
                updates.append(gr.update(
                    visible=True,
                    value=labels[i],
                ))
            else:
                updates.append(gr.update(visible=False, value=""))

        # Status
        seq_len = result["spatial_shape"][0] * result["spatial_shape"][1]
        updates.append(
            f"Encoded: {num_q} levels, {seq_len} positions, "
            f"{result['codebook_size']} codebook size, "
            f"{result['duration_s']:.1f}s"
        )

        return updates

    except Exception as exc:
        logger.exception("Encode failed")
        empty = [gr.update()] * (3 + MAX_LEVEL_LABELS) + [f"Encode failed: {exc}"]
        return empty


def _decode_current():
    """Decode the current code grid back to audio.

    Returns
    -------
    tuple
        ``(48000, wav_array)`` for the decoded audio player, or None.
    """
    if _current_encode is None:
        return None

    if app_state.loaded_vq_model is None:
        return None

    try:
        from distill.inference.codes import decode_code_grid

        wav_array = decode_code_grid(
            _current_encode["indices"],
            _current_encode["spatial_shape"],
            _current_encode["mel_shape"],
            app_state.loaded_vq_model,
        )
        return (48000, wav_array)
    except Exception as exc:
        logger.exception("Decode failed")
        return None


def _handle_cell_click(click_info: str):
    """Parse click info from the hidden textbox and dispatch to preview functions.

    Click info format:
    - ``"cell,{level},{position}"`` -- preview a single codebook entry
    - ``"col,{position}"`` -- preview all levels at one time position
    - ``"row,{level}"`` -- play one level's contribution across all time

    Returns
    -------
    list
        [preview_audio, grid_html]
    """
    if not click_info or _current_encode is None or app_state.loaded_vq_model is None:
        return [gr.update(), gr.update()]

    try:
        from distill.inference.codes import (
            preview_single_code,
            preview_time_slice,
            play_row_audio,
        )
        from distill.ui.components.code_grid import render_code_grid

        parts = click_info.strip().split(",")
        event_type = parts[0]

        loaded = app_state.loaded_vq_model
        enc = _current_encode
        labels = _current_labels or []

        if event_type == "cell" and len(parts) >= 3:
            level = int(parts[1])
            position = int(parts[2])
            code_index = int(enc["indices"][0, position, level].item())

            audio = preview_single_code(
                level, code_index, loaded,
                enc["spatial_shape"], enc["mel_shape"],
            )

            # Re-render grid with selection highlight
            grid_html = render_code_grid(
                indices=enc["indices"],
                num_quantizers=enc["num_quantizers"],
                codebook_size=enc["codebook_size"],
                spatial_shape=enc["spatial_shape"],
                level_labels=labels,
                selected_cell=(level, position),
                duration_s=enc["duration_s"],
            )

            return [(48000, audio), grid_html]

        elif event_type == "col" and len(parts) >= 2:
            position = int(parts[1])

            audio = preview_time_slice(
                position, enc["indices"], loaded,
                enc["spatial_shape"], enc["mel_shape"],
            )

            # Re-render grid (no specific cell selected, but could highlight column)
            grid_html = render_code_grid(
                indices=enc["indices"],
                num_quantizers=enc["num_quantizers"],
                codebook_size=enc["codebook_size"],
                spatial_shape=enc["spatial_shape"],
                level_labels=labels,
                selected_cell=None,
                duration_s=enc["duration_s"],
            )

            return [(48000, audio), grid_html]

        elif event_type == "row" and len(parts) >= 2:
            level = int(parts[1])

            audio = play_row_audio(
                level, enc["indices"], loaded,
                enc["spatial_shape"], enc["mel_shape"],
            )

            # Re-render grid (no cell selected)
            grid_html = render_code_grid(
                indices=enc["indices"],
                num_quantizers=enc["num_quantizers"],
                codebook_size=enc["codebook_size"],
                spatial_shape=enc["spatial_shape"],
                level_labels=labels,
                selected_cell=None,
                duration_s=enc["duration_s"],
            )

            return [(48000, audio), grid_html]

        else:
            return [gr.update(), gr.update()]

    except Exception as exc:
        logger.exception("Cell click handler failed")
        return [gr.update(), gr.update()]


def _update_level_labels(*label_texts):
    """Re-render the code grid with custom level labels.

    Parameters
    ----------
    *label_texts : str
        One label per textbox (up to MAX_LEVEL_LABELS).

    Returns
    -------
    str
        Updated grid HTML.
    """
    global _current_labels  # noqa: WPS420

    if _current_encode is None:
        return gr.update()

    from distill.ui.components.code_grid import render_code_grid

    enc = _current_encode
    num_q = enc["num_quantizers"]

    # Collect only the first num_q labels
    labels = [str(t).strip() for t in label_texts[:num_q]]
    _current_labels = labels

    grid_html = render_code_grid(
        indices=enc["indices"],
        num_quantizers=enc["num_quantizers"],
        codebook_size=enc["codebook_size"],
        spatial_shape=enc["spatial_shape"],
        level_labels=labels,
        selected_cell=None,
        duration_s=enc["duration_s"],
    )

    return grid_html


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_codes_tab() -> dict:
    """Build the Codes tab UI within the current Blocks context.

    Layout (top to bottom):
    1. Controls row: model selector, audio upload, encode/decode buttons
    2. Level label editor (collapsible accordion)
    3. Code grid display (HTML + hidden JS bridge textbox)
    4. Audio players row: original, decoded reconstruction, preview

    Returns
    -------
    dict
        Component references for cross-tab wiring:
        ``model_dropdown``, ``grid_html``, ``status``.
    """
    # -- Controls row --
    with gr.Row():
        model_dropdown = gr.Dropdown(
            label="VQ-VAE Model",
            choices=_get_vqvae_model_choices(),
            filterable=True,
            scale=2,
        )
        audio_input = gr.Audio(
            label="Audio File",
            sources=["upload"],
            type="filepath",
            scale=2,
        )
        encode_btn = gr.Button("Encode", variant="primary", scale=1)
        decode_btn = gr.Button("Decode", variant="secondary", scale=1)

    # Status message
    status_msg = gr.Textbox(
        label="Status",
        interactive=False,
        value="",
    )

    # -- Level label editor (collapsible) --
    with gr.Accordion("Level Labels", open=False):
        gr.Markdown(
            "Edit level labels for the code grid. "
            "Default: Structure (coarsest) to Detail (finest)."
        )
        label_textboxes: list[gr.Textbox] = []
        for i in range(MAX_LEVEL_LABELS):
            tb = gr.Textbox(
                label=f"Level {i}",
                value="",
                visible=False,
                interactive=True,
            )
            label_textboxes.append(tb)
        apply_labels_btn = gr.Button("Apply Labels", size="sm")

    # -- Code grid display --
    grid_html = gr.HTML(
        value=(
            '<div style="display: flex; align-items: center; justify-content: center; '
            'min-height: 200px; color: #888; font-size: 1.1em; text-align: center; '
            'padding: 40px;">'
            "<p>Upload an audio file and click Encode to see codes here.</p>"
            "</div>"
        ),
        elem_id="code-grid-container",
    )

    # Hidden textbox for JS->Python bridge (code grid click events)
    cell_clicked = gr.Textbox(
        value="",
        visible=False,
        elem_id="code-grid-cell-clicked",
    )

    # -- Audio players row --
    with gr.Row():
        original_audio = gr.Audio(
            label="Original Audio",
            interactive=False,
        )
        decoded_audio = gr.Audio(
            label="Decoded Reconstruction",
            interactive=False,
        )
        preview_audio = gr.Audio(
            label="Preview",
            interactive=False,
            autoplay=True,
        )

    # -- Wire event handlers --

    # Model selection -> load model
    model_dropdown.change(
        fn=_load_model,
        inputs=[model_dropdown],
        outputs=[status_msg],
    )

    # Encode button -> encode audio, render grid, auto-decode
    encode_outputs = [grid_html, original_audio, decoded_audio] + label_textboxes + [status_msg]
    encode_btn.click(
        fn=_encode_audio,
        inputs=[audio_input, model_dropdown],
        outputs=encode_outputs,
    )

    # Decode button -> decode current grid to audio
    decode_btn.click(
        fn=_decode_current,
        inputs=None,
        outputs=[decoded_audio],
    )

    # Cell/column/row click from JS bridge -> preview audio + grid update
    cell_clicked.change(
        fn=_handle_cell_click,
        inputs=[cell_clicked],
        outputs=[preview_audio, grid_html],
    )

    # Apply custom labels -> re-render grid
    apply_labels_btn.click(
        fn=_update_level_labels,
        inputs=label_textboxes,
        outputs=[grid_html],
    )

    return {
        "model_dropdown": model_dropdown,
        "grid_html": grid_html,
        "status": status_msg,
    }
