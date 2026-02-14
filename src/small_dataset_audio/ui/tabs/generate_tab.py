"""Generate tab: slider-controlled audio generation, playback, export, and presets.

Provides the core user interaction surface -- the generate-listen-export
workflow.  Slider controls map to the PCA-derived latent space from Phase 5.
Pre-creates ``MAX_SLIDERS`` Gradio sliders distributed across 3 columns
(timbral / temporal / spatial); visibility updates dynamically when a model
is loaded.

Uses ``app_state`` singleton for backend access (GenerationPipeline,
PresetManager, etc.).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from small_dataset_audio.ui.components.guided_nav import get_empty_state_message
from small_dataset_audio.ui.state import app_state

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum PCA components any model will have
MAX_SLIDERS = 12

# Category keyword mapping for column assignment
_TIMBRAL_KEYWORDS = {"bright", "warm", "rough", "harsh", "soft", "tone", "timbre"}
_TEMPORAL_KEYWORDS = {"rhythm", "pulse", "tempo", "attack", "decay", "speed", "fast"}
_SPATIAL_KEYWORDS = {"space", "reverb", "dense", "wide", "stereo", "room", "depth"}


def _assign_column(suggested_label: str) -> int:
    """Assign a slider to a column based on its suggested label keywords.

    Returns 0 (timbral), 1 (temporal), or 2 (spatial).
    Falls back to sequential distribution if no keyword matches.
    """
    lower = suggested_label.lower()
    for kw in _TIMBRAL_KEYWORDS:
        if kw in lower:
            return 0
    for kw in _TEMPORAL_KEYWORDS:
        if kw in lower:
            return 1
    for kw in _SPATIAL_KEYWORDS:
        if kw in lower:
            return 2
    # No match -- will be assigned sequentially in the caller
    return -1


def _quality_badge_markdown(quality: dict) -> str:
    """Build a Markdown badge from a quality score dict."""
    rating = quality.get("rating", "unknown")
    reason = quality.get("rating_reason", "")
    snr = quality.get("snr_db", 0.0)

    if rating == "green":
        icon = "🟢"
    elif rating == "yellow":
        icon = "🟡"
    elif rating == "red":
        icon = "🔴"
    else:
        icon = "⚪"

    snr_str = f"{snr:.1f}" if snr != float("inf") else ">60"
    return f"**Quality:** {icon} {rating.upper()} -- SNR {snr_str} dB | {reason}"


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------


def _update_sliders_for_model():
    """Update slider labels, visibility, and preset dropdown for loaded model.

    Returns a list of gr.update dicts for all MAX_SLIDERS sliders plus
    the preset dropdown, plus the controls column visibility, plus the
    empty state message visibility.
    """
    from small_dataset_audio.controls.mapping import get_slider_info
    from small_dataset_audio.presets.manager import PresetManager
    from small_dataset_audio.history.store import GenerationHistory

    if app_state.loaded_model is None or app_state.loaded_model.analysis is None:
        # No model loaded -- hide controls, show empty state
        updates = [gr.update(visible=False) for _ in range(MAX_SLIDERS)]
        # preset dropdown
        updates.append(gr.update(choices=["Custom"], value="Custom"))
        # controls column visible
        updates.append(gr.update(visible=False))
        # empty state visible
        updates.append(gr.update(visible=True))
        return updates

    analysis = app_state.loaded_model.analysis
    slider_infos = get_slider_info(analysis)
    n_active = len(slider_infos)

    # Initialize preset manager for this model if needed
    if app_state.preset_manager is None:
        model_id = app_state.loaded_model.metadata.model_id
        app_state.presets_dir.mkdir(parents=True, exist_ok=True)
        app_state.preset_manager = PresetManager(
            app_state.presets_dir, model_id
        )

    # Initialize history store if needed
    if app_state.history_store is None:
        app_state.history_dir.mkdir(parents=True, exist_ok=True)
        app_state.history_store = GenerationHistory(app_state.history_dir)

    # Build slider updates
    updates = []
    for i in range(MAX_SLIDERS):
        if i < n_active:
            info = slider_infos[i]
            label = info["suggested_label"]
            variance_pct = info["variance_explained_pct"]
            updates.append(gr.update(
                visible=True,
                label=f"{label} ({variance_pct:.1f}%)",
                minimum=info["min_step"],
                maximum=info["max_step"],
                value=0,
            ))
        else:
            updates.append(gr.update(visible=False, value=0))

    # Preset dropdown choices
    presets = app_state.preset_manager.list_presets()
    preset_choices = ["Custom"] + [p.name for p in presets]
    updates.append(gr.update(choices=preset_choices, value="Custom"))

    # Controls column visible
    updates.append(gr.update(visible=True))
    # Empty state hidden
    updates.append(gr.update(visible=False))

    return updates


def _generate_audio(*args):
    """Generate audio from slider values and generation config.

    Arguments are unpacked as:
        slider_0 ... slider_{MAX_SLIDERS-1}, duration, stereo_mode,
        stereo_width, seed
    """
    from small_dataset_audio.controls.mapping import SliderState, sliders_to_latent
    from small_dataset_audio.inference.generation import GenerationConfig
    from small_dataset_audio.inference.quality import compute_quality_score

    # Unpack arguments
    slider_values = list(args[:MAX_SLIDERS])
    duration = args[MAX_SLIDERS]
    stereo_mode = args[MAX_SLIDERS + 1]
    stereo_width = args[MAX_SLIDERS + 2]
    seed_val = args[MAX_SLIDERS + 3]

    # Validate model loaded
    if app_state.loaded_model is None or app_state.pipeline is None:
        return (
            None,
            "**No model loaded.** Go to the Library tab to load a model.",
            gr.update(visible=False),
        )

    analysis = app_state.loaded_model.analysis
    if analysis is None:
        return (
            None,
            "**Model has no latent space analysis.** Re-analyze the model.",
            gr.update(visible=False),
        )

    n_active = analysis.n_active_components

    # Build SliderState from slider values (only first N active)
    positions = [int(slider_values[i]) for i in range(n_active)]
    slider_state = SliderState(positions=positions, n_components=n_active)

    # Convert to latent vector
    latent_vector = sliders_to_latent(slider_state, analysis)

    # Parse seed
    seed = int(seed_val) if seed_val is not None and seed_val != "" else None

    # Build GenerationConfig
    config = GenerationConfig(
        latent_vector=latent_vector,
        duration_s=float(duration),
        stereo_mode=stereo_mode,
        stereo_width=float(stereo_width),
        seed=seed,
    )

    # Generate
    try:
        result = app_state.pipeline.generate(config)
    except Exception as exc:
        logger.exception("Generation failed")
        return (
            None,
            f"**Generation failed:** {exc}",
            gr.update(visible=False),
        )

    # Store result in app_state for export
    app_state.metrics_buffer["last_result"] = result

    # Add to history if history store exists
    if app_state.history_store is not None and app_state.loaded_model is not None:
        try:
            app_state.history_store.add_to_history(
                result=result,
                model_id=app_state.loaded_model.metadata.model_id,
                model_name=app_state.loaded_model.metadata.model_name,
                slider_positions=positions,
                n_components=n_active,
                preset_name="custom",
            )
        except Exception as exc:
            logger.warning("Failed to add to history: %s", exc)

    # Build quality badge
    badge = _quality_badge_markdown(result.quality)

    # Return audio as (sample_rate, ndarray) tuple for gr.Audio
    return (
        (result.sample_rate, result.audio),
        badge,
        gr.update(visible=True),
    )


def _export_audio(sample_rate_str: str, bit_depth: str, filename: str):
    """Export the last generated audio as a WAV file."""
    from small_dataset_audio.inference.export import export_wav

    last_result = app_state.metrics_buffer.get("last_result")
    if last_result is None:
        return "No audio to export. Generate audio first."

    # Parse sample rate
    sr = int(sample_rate_str)

    # Build output path
    output_dir = app_state.generated_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename and filename.strip():
        fname = filename.strip()
        if not fname.endswith(".wav"):
            fname = fname + ".wav"
    else:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"gen_{timestamp}_seed{last_result.seed_used}.wav"

    output_path = output_dir / fname

    try:
        export_wav(
            audio=last_result.audio,
            path=output_path,
            sample_rate=sr,
            bit_depth=bit_depth,
        )
        return f"Exported to `{output_path}`"
    except Exception as exc:
        logger.exception("Export failed")
        return f"**Export failed:** {exc}"


def _save_preset(preset_name: str, *slider_values):
    """Save current slider positions as a preset."""
    if app_state.preset_manager is None:
        return gr.update(), "No model loaded."

    if not preset_name or not preset_name.strip():
        return gr.update(), "Please enter a preset name."

    analysis = app_state.loaded_model.analysis if app_state.loaded_model else None
    if analysis is None:
        return gr.update(), "No model analysis available."

    n_active = analysis.n_active_components
    positions = [int(slider_values[i]) for i in range(n_active)]

    app_state.preset_manager.save_preset(
        name=preset_name.strip(),
        slider_positions=positions,
        n_components=n_active,
    )

    # Refresh preset dropdown
    presets = app_state.preset_manager.list_presets()
    choices = ["Custom"] + [p.name for p in presets]
    return gr.update(choices=choices, value=preset_name.strip()), ""


def _load_preset(preset_name: str):
    """Load slider values from a preset.

    Returns updates for all MAX_SLIDERS sliders.
    """
    if preset_name == "Custom" or app_state.preset_manager is None:
        return [gr.update() for _ in range(MAX_SLIDERS)]

    # Find preset by name
    presets = app_state.preset_manager.list_presets()
    target = None
    for p in presets:
        if p.name == preset_name:
            target = p
            break

    if target is None:
        return [gr.update() for _ in range(MAX_SLIDERS)]

    # Load preset positions
    slider_state, _seed = app_state.preset_manager.load_preset(target.preset_id)

    updates = []
    for i in range(MAX_SLIDERS):
        if i < len(slider_state.positions):
            updates.append(gr.update(value=slider_state.positions[i]))
        else:
            updates.append(gr.update(value=0))

    return updates


def _delete_preset(preset_name: str):
    """Delete a preset by name."""
    if preset_name == "Custom" or app_state.preset_manager is None:
        return gr.update()

    # Find preset by name
    presets = app_state.preset_manager.list_presets()
    for p in presets:
        if p.name == preset_name:
            app_state.preset_manager.delete_preset(p.preset_id)
            break

    # Refresh dropdown
    presets = app_state.preset_manager.list_presets()
    choices = ["Custom"] + [p.name for p in presets]
    return gr.update(choices=choices, value="Custom")


def _randomize_seed():
    """Return a random integer seed."""
    return random.randint(0, 2**31)


def _toggle_stereo_width(stereo_mode: str):
    """Show/hide stereo width slider based on stereo mode."""
    return gr.update(visible=(stereo_mode != "mono"))


def _toggle_preset_name():
    """Show preset name textbox when saving."""
    return gr.update(visible=True)


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_generate_tab() -> None:
    """Build the Generate tab UI within the current Blocks context.

    Layout:
    - Empty state message (shown when no model loaded)
    - Controls section (hidden until model loaded):
      - Sliders in 3 columns (timbral / temporal / spatial)
      - Generation config (duration, stereo mode, stereo width)
      - Seed row
      - Generate button
      - Audio player with quality badge
      - Export controls
      - Preset management
    """
    # Empty state message
    empty_msg = gr.Markdown(
        value=get_empty_state_message("generate"),
        visible=True,
    )

    # Controls section -- hidden until model loaded
    with gr.Column(visible=False) as controls_section:
        gr.Markdown("## Generate")

        # ----- Slider section: 3 columns -----
        gr.Markdown("### Parameters")
        sliders: list[gr.Slider] = []

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Timbral**", elem_classes=["slider-category"])
                for i in range(4):  # Sliders 0-3
                    s = gr.Slider(
                        minimum=-10,
                        maximum=10,
                        value=0,
                        step=1,
                        label=f"Axis {i + 1}",
                        visible=False,
                        interactive=True,
                    )
                    sliders.append(s)
            with gr.Column():
                gr.Markdown("**Temporal**", elem_classes=["slider-category"])
                for i in range(4, 8):  # Sliders 4-7
                    s = gr.Slider(
                        minimum=-10,
                        maximum=10,
                        value=0,
                        step=1,
                        label=f"Axis {i + 1}",
                        visible=False,
                        interactive=True,
                    )
                    sliders.append(s)
            with gr.Column():
                gr.Markdown("**Spatial**", elem_classes=["slider-category"])
                for i in range(8, MAX_SLIDERS):  # Sliders 8-11
                    s = gr.Slider(
                        minimum=-10,
                        maximum=10,
                        value=0,
                        step=1,
                        label=f"Axis {i + 1}",
                        visible=False,
                        interactive=True,
                    )
                    sliders.append(s)

        # ----- Generation config row -----
        gr.Markdown("### Generation Config")
        with gr.Row():
            duration_input = gr.Number(
                label="Duration (sec)",
                value=1.0,
                minimum=0.1,
                maximum=60.0,
                precision=1,
            )
            stereo_mode_dd = gr.Dropdown(
                choices=["mono", "mid_side", "dual_seed"],
                value="mono",
                label="Stereo Mode",
            )
            stereo_width_slider = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                step=0.1,
                value=0.7,
                label="Stereo Width",
                visible=False,
            )

        # ----- Seed row -----
        with gr.Row():
            seed_input = gr.Number(
                label="Seed",
                value=None,
                precision=0,
            )
            randomize_btn = gr.Button("Randomize", size="sm")

        # ----- Generate button -----
        generate_btn = gr.Button("Generate", variant="primary", size="lg")

        # ----- Audio output section -----
        with gr.Row():
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    visible=False,
                    type="numpy",
                    interactive=False,
                    autoplay=False,
                )
                quality_badge = gr.Markdown(value="")
            with gr.Column(scale=1):
                gr.Markdown("### Export")
                export_sr_dd = gr.Dropdown(
                    choices=["44100", "48000", "96000"],
                    value="48000",
                    label="Sample Rate",
                )
                export_bd_dd = gr.Dropdown(
                    choices=["16-bit", "24-bit", "32-bit float"],
                    value="24-bit",
                    label="Bit Depth",
                )
                export_filename = gr.Textbox(
                    label="Filename",
                    placeholder="auto-generated",
                )
                export_btn = gr.Button("Export WAV", variant="secondary")
                export_status = gr.Markdown(value="")

        # ----- Preset section -----
        gr.Markdown("### Presets")
        with gr.Row():
            preset_dd = gr.Dropdown(
                label="Preset",
                choices=["Custom"],
                value="Custom",
                filterable=True,
                allow_custom_value=False,
            )
            preset_name_input = gr.Textbox(
                label="Preset Name",
                placeholder="My Preset",
                visible=False,
            )
            save_preset_btn = gr.Button("Save Preset", size="sm")
            delete_preset_btn = gr.Button("Delete Preset", size="sm", variant="stop")

    # ----- Wire event handlers -----

    # Stereo width visibility toggle
    stereo_mode_dd.change(
        fn=_toggle_stereo_width,
        inputs=[stereo_mode_dd],
        outputs=[stereo_width_slider],
    )

    # Randomize seed
    randomize_btn.click(
        fn=_randomize_seed,
        inputs=None,
        outputs=[seed_input],
    )

    # Generate audio
    generate_btn.click(
        fn=_generate_audio,
        inputs=sliders + [duration_input, stereo_mode_dd, stereo_width_slider, seed_input],
        outputs=[audio_output, quality_badge, audio_output],
    )

    # Export audio
    export_btn.click(
        fn=_export_audio,
        inputs=[export_sr_dd, export_bd_dd, export_filename],
        outputs=[export_status],
    )

    # Show preset name input when save is clicked
    save_preset_btn.click(
        fn=_toggle_preset_name,
        inputs=None,
        outputs=[preset_name_input],
    ).then(
        fn=_save_preset,
        inputs=[preset_name_input] + sliders,
        outputs=[preset_dd, export_status],
    )

    # Load preset
    preset_dd.change(
        fn=_load_preset,
        inputs=[preset_dd],
        outputs=sliders,
    )

    # Delete preset
    delete_preset_btn.click(
        fn=_delete_preset,
        inputs=[preset_dd],
        outputs=[preset_dd],
    )

    # Return components that need external wiring (model loading)
    # Store references for update_sliders_for_model
    return {
        "sliders": sliders,
        "preset_dd": preset_dd,
        "controls_section": controls_section,
        "empty_msg": empty_msg,
    }
