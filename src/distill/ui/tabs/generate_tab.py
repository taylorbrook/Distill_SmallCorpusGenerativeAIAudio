"""Generate tab: slider-controlled audio generation, playback, multi-format
export, spatial audio controls, multi-model blending, presets, history
browsing, and A/B comparison.

Provides the core user interaction surface -- the generate-listen-export
workflow.  Slider controls map to the PCA-derived latent space from Phase 5.
Pre-creates ``MAX_SLIDERS`` Gradio sliders distributed across 3 columns
(timbral / temporal / spatial); visibility updates dynamically when a model
is loaded.

Phase 10 additions:
- Output mode selector (mono/stereo/binaural) with spatial width+depth sliders
- Export format dropdown (WAV/MP3/FLAC/OGG) with editable metadata fields
- Multi-model blend accordion with up to 4 model slots and weight sliders

Uses ``app_state`` singleton for backend access (GenerationPipeline,
PresetManager, GenerationHistory, ABComparison, BlendEngine, etc.).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from distill.ui.components.guided_nav import get_empty_state_message
from distill.ui.state import app_state

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum blend model slots
MAX_BLEND_SLOTS = 4

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
    from distill.controls.mapping import get_slider_info
    from distill.presets.manager import PresetManager
    from distill.history.store import GenerationHistory

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
        slider_0 ... slider_{MAX_SLIDERS-1}, duration, output_mode,
        spatial_width, spatial_depth, seed

    Returns:
        Tuple of (audio, badge, audio_visible, history_gallery, ab_choices_a,
        ab_choices_b).
    """
    from distill.controls.mapping import SliderState, sliders_to_latent
    from distill.inference.generation import GenerationConfig
    from distill.inference.spatial import SpatialConfig, SpatialMode

    # Unpack arguments
    slider_values = list(args[:MAX_SLIDERS])
    duration = args[MAX_SLIDERS]
    output_mode = args[MAX_SLIDERS + 1]
    spatial_width = args[MAX_SLIDERS + 2]
    spatial_depth = args[MAX_SLIDERS + 3]
    seed_val = args[MAX_SLIDERS + 4]
    evolution = args[MAX_SLIDERS + 5]

    # Validate model loaded
    if app_state.loaded_model is None or app_state.pipeline is None:
        return (
            None,
            "**No model loaded.** Go to the Library tab to load a model.",
            gr.update(visible=False),
            gr.update(),  # history gallery
            gr.update(),  # ab dropdown a
            gr.update(),  # ab dropdown b
        )

    analysis = app_state.loaded_model.analysis
    if analysis is None:
        return (
            None,
            "**Model has no latent space analysis.** Re-analyze the model.",
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    n_active = min(analysis.n_active_components, MAX_SLIDERS)

    # Build SliderState from slider values (only first N active)
    positions = [int(slider_values[i]) for i in range(n_active)]
    slider_state = SliderState(positions=positions, n_components=n_active)

    # Convert to latent vector
    latent_vector = sliders_to_latent(slider_state, analysis)

    # Parse seed
    seed = int(seed_val) if seed_val is not None and seed_val != "" else None

    # Build SpatialConfig from output mode, width, depth
    spatial_mode = SpatialMode(output_mode)
    spatial_config = SpatialConfig(
        mode=spatial_mode,
        width=float(spatial_width),
        depth=float(spatial_depth),
    )

    # Build GenerationConfig with new spatial config
    config = GenerationConfig(
        latent_vector=latent_vector,
        duration_s=float(duration),
        spatial=spatial_config,
        seed=seed,
        evolution_amount=float(evolution),
    )

    # Check if blend engine has active models
    if (
        app_state.blend_engine is not None
        and len(app_state.blend_engine.get_active_slots()) > 0
    ):
        try:
            result = app_state.blend_engine.blend_generate(positions, config)
        except Exception as exc:
            logger.exception("Blend generation failed")
            return (
                None,
                f"**Blend generation failed:** {exc}",
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(),
            )
    else:
        # Single model generation
        try:
            result = app_state.pipeline.generate(config)
        except Exception as exc:
            logger.exception("Generation failed")
            return (
                None,
                f"**Generation failed:** {exc}",
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(),
            )

    # Store result in app_state for export
    app_state.metrics_buffer["last_result"] = result

    # Add to history if history store exists
    if app_state.history_store is not None and app_state.loaded_model is not None:
        try:
            app_state.history_store.add_to_history(
                result=result,
                model_id=app_state.loaded_model.metadata.model_id,
                model_name=app_state.loaded_model.metadata.name,
                slider_positions=positions,
                n_components=n_active,
                preset_name="custom",
            )
        except Exception as exc:
            logger.warning("Failed to add to history: %s", exc)

    # Build quality badge
    badge = _quality_badge_markdown(result.quality)

    # Auto-refresh history gallery and A/B dropdown choices after generation
    gallery_items = _build_history_gallery()
    ab_choices = _build_ab_choices()

    # Return audio as (sample_rate, ndarray) tuple for gr.Audio
    return (
        (result.sample_rate, result.audio),
        badge,
        gr.update(visible=True),
        gallery_items,
        gr.update(choices=ab_choices, value=None),
        gr.update(choices=ab_choices, value=None),
    )


def _generate_prior_audio(
    temperature, top_k, top_p, duration, overlap_ms, seed_val,
    progress=gr.Progress(),
):
    """Generate audio from a trained VQ-VAE prior.

    Returns:
        Tuple of (audio, status_markdown, audio_visibility).
    """
    # Validate model loaded
    if (
        app_state.loaded_vq_model is None
        or app_state.loaded_vq_model.prior is None
    ):
        return (
            None,
            "**No VQ-VAE model with prior loaded.** Train a prior first.",
            gr.update(visible=False),
        )

    # Parse seed
    seed = (
        int(seed_val)
        if seed_val is not None and seed_val != ""
        else None
    )

    # Compute overlap_samples from overlap_ms (48 samples per ms at 48kHz)
    overlap_samples = int(float(overlap_ms) * 48)

    # Progress callback wrapping gr.Progress
    def _progress_cb(fraction, desc):
        progress(fraction, desc=desc)

    try:
        from distill.inference.generation import generate_audio_from_prior

        audio, seed_used = generate_audio_from_prior(
            loaded=app_state.loaded_vq_model,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            duration_s=float(duration),
            overlap_samples=overlap_samples,
            seed=seed,
            progress_callback=_progress_cb,
        )

        # Store result info for potential export
        app_state.metrics_buffer["last_prior_result"] = {
            "audio": audio,
            "seed_used": seed_used,
            "duration_s": len(audio) / 48000,
            "sample_rate": 48000,
        }

        return (
            (48000, audio),
            f"**Generated** -- seed: {seed_used}, duration: {len(audio)/48000:.1f}s",
            gr.update(visible=True),
        )
    except Exception as exc:
        logger.exception("Prior generation failed")
        return (
            None,
            f"**Generation failed:** {exc}",
            gr.update(visible=False),
        )


def _export_audio(
    sample_rate_str: str,
    bit_depth: str,
    filename: str,
    export_format: str,
    meta_artist: str,
    meta_album: str,
    meta_title: str,
):
    """Export the last generated audio in the selected format with metadata."""
    from distill.inference.export import ExportFormat, FORMAT_EXTENSIONS
    from distill.audio.metadata import build_export_metadata

    last_result = app_state.metrics_buffer.get("last_result")
    if last_result is None:
        return "No audio to export. Generate audio first."

    # Resolve export format
    fmt = ExportFormat(export_format)
    extension = FORMAT_EXTENSIONS[fmt]

    # Build output path
    output_dir = app_state.generated_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename and filename.strip():
        fname = filename.strip()
        # Strip any existing extension and apply the correct one
        for ext in (".wav", ".mp3", ".flac", ".ogg"):
            if fname.lower().endswith(ext):
                fname = fname[:-len(ext)]
                break
        fname = fname + extension
    else:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"gen_{timestamp}_seed{last_result.seed_used}{extension}"

    output_path = output_dir / fname

    # Build metadata for tag embedding
    model_name = "unknown"
    if app_state.loaded_model is not None:
        model_name = app_state.loaded_model.metadata.name

    overrides = {}
    if meta_artist and meta_artist.strip():
        overrides["artist"] = meta_artist.strip()
    if meta_album and meta_album.strip():
        overrides["album"] = meta_album.strip()
    if meta_title and meta_title.strip():
        overrides["title"] = meta_title.strip()

    metadata = build_export_metadata(
        model_name=model_name,
        seed=last_result.seed_used,
        overrides=overrides if overrides else None,
    )

    try:
        if app_state.pipeline is not None:
            audio_path, json_path = app_state.pipeline.export(
                result=last_result,
                output_dir=output_dir,
                filename=fname,
                export_format=fmt,
                metadata=metadata,
            )
        else:
            # Fallback: direct export without pipeline
            from distill.inference.export import export_audio

            export_audio(
                audio=last_result.audio,
                path=output_path,
                sample_rate=int(sample_rate_str),
                format=fmt,
                bit_depth=bit_depth,
                metadata=metadata,
            )
            audio_path = output_path

        return f"Exported {export_format.upper()} to `{audio_path}`"
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

    n_active = min(analysis.n_active_components, MAX_SLIDERS)
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


def _toggle_spatial_controls(output_mode: str):
    """Show/hide spatial width and depth sliders based on output mode."""
    show = output_mode != "mono"
    return gr.update(visible=show), gr.update(visible=show)


def _toggle_bit_depth(export_format: str):
    """Show bit depth dropdown only for WAV format."""
    return gr.update(visible=(export_format == "wav"))


def _toggle_preset_name():
    """Show preset name textbox when saving."""
    return gr.update(visible=True)


# ---------------------------------------------------------------------------
# Blend panel handlers
# ---------------------------------------------------------------------------

# Track how many blend rows are visible (module-level counter)
_blend_visible_count = 1


def _add_blend_row():
    """Reveal the next hidden blend model row.

    Returns status markdown + visibility updates for all MAX_BLEND_SLOTS rows.
    """
    global _blend_visible_count  # noqa: WPS420
    if _blend_visible_count < MAX_BLEND_SLOTS:
        _blend_visible_count += 1
    status = f"{_blend_visible_count} model slot(s) active."
    updates = [status]
    for i in range(MAX_BLEND_SLOTS):
        updates.append(gr.update(visible=(i < _blend_visible_count)))
    return updates


def _remove_blend_row():
    """Hide the last visible blend model row.

    Returns status markdown + visibility updates for all MAX_BLEND_SLOTS rows.
    """
    global _blend_visible_count  # noqa: WPS420
    if _blend_visible_count > 1:
        _blend_visible_count -= 1
    status = f"{_blend_visible_count} model slot(s) active."
    if _blend_visible_count == 1:
        status = "Single model mode (blend inactive)."
    updates = [status]
    for i in range(MAX_BLEND_SLOTS):
        updates.append(gr.update(visible=(i < _blend_visible_count)))
    return updates


def _refresh_blend_model_choices():
    """Refresh all blend model dropdown choices from ModelLibrary.

    Returns gr.update for each of MAX_BLEND_SLOTS dropdowns.
    Called via cross-tab wiring when library content changes.
    """
    choices: list[str] = []
    if app_state.model_library is not None:
        try:
            entries = app_state.model_library.list_models()
            choices = [e.name for e in entries]
        except Exception:
            logger.warning("Failed to list models for blend dropdowns")

    updates = []
    for _ in range(MAX_BLEND_SLOTS):
        updates.append(gr.update(choices=choices))
    return updates


# ---------------------------------------------------------------------------
# History handlers
# ---------------------------------------------------------------------------


def _build_history_gallery() -> list[tuple[str, str]]:
    """Build gallery items from history entries.

    Returns a list of (thumbnail_path, caption) tuples for gr.Gallery.
    """
    if app_state.history_store is None:
        return []

    entries = app_state.history_store.list_entries(limit=20)
    items: list[tuple[str, str]] = []
    for entry in entries:
        thumb_path = app_state.history_store.history_dir / entry.thumbnail_file
        if thumb_path.exists():
            # Caption: seed + timestamp (compact)
            ts_short = entry.timestamp[:16].replace("T", " ")
            caption = f"seed:{entry.seed} | {ts_short}"
            items.append((str(thumb_path), caption))
    return items


def _refresh_history():
    """Refresh the history gallery and A/B dropdown choices.

    Returns:
        Tuple of (gallery_items, ab_choices_a, ab_choices_b).
    """
    gallery_items = _build_history_gallery()
    ab_choices = _build_ab_choices()
    return (
        gallery_items,
        gr.update(choices=ab_choices, value=None),
        gr.update(choices=ab_choices, value=None),
    )


def _play_history_entry(evt: gr.SelectData):
    """Play the audio file for a selected history gallery entry.

    Returns the audio file path for the History Playback audio component.
    """
    if app_state.history_store is None:
        return gr.update(visible=False, value=None)

    entries = app_state.history_store.list_entries(limit=20)
    idx = evt.index
    if 0 <= idx < len(entries):
        entry = entries[idx]
        audio_path = app_state.history_store.history_dir / entry.audio_file
        if audio_path.exists():
            return gr.update(visible=True, value=str(audio_path))

    return gr.update(visible=False, value=None)


# ---------------------------------------------------------------------------
# A/B Comparison handlers
# ---------------------------------------------------------------------------


def _build_ab_choices() -> list[str]:
    """Build dropdown choices for A/B comparison from history entries.

    Each choice is formatted as "seed:{seed} | {timestamp_short}" to match
    the gallery caption, with the entry_id prefix for lookup.
    """
    if app_state.history_store is None:
        return []

    entries = app_state.history_store.list_entries(limit=20)
    choices: list[str] = []
    for entry in entries:
        ts_short = entry.timestamp[:16].replace("T", " ")
        label = f"{entry.entry_id[:8]} | seed:{entry.seed} | {ts_short}"
        choices.append(label)
    return choices


def _find_entry_id_from_choice(choice: str) -> str | None:
    """Extract the entry_id prefix from an A/B dropdown choice string."""
    if not choice:
        return None
    # Format: "{entry_id[:8]} | seed:... | ..."
    parts = choice.split(" | ")
    if not parts:
        return None
    prefix = parts[0].strip()

    # Find matching entry by prefix
    if app_state.history_store is None:
        return None
    entries = app_state.history_store.list_entries(limit=20)
    for entry in entries:
        if entry.entry_id.startswith(prefix):
            return entry.entry_id
    return None


def _start_comparison(entry_a_choice: str, entry_b_choice: str):
    """Start A/B comparison between two selected history entries.

    Returns:
        Tuple of (audio_a, audio_b, status_msg).
    """
    from distill.history.comparison import ABComparison

    if not entry_a_choice or not entry_b_choice:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "Select two entries to compare.",
        )

    entry_a_id = _find_entry_id_from_choice(entry_a_choice)
    entry_b_id = _find_entry_id_from_choice(entry_b_choice)

    if entry_a_id is None or entry_b_id is None:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "Could not find selected entries.",
        )

    if entry_a_id == entry_b_id:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            "Select two different entries to compare.",
        )

    try:
        comparison = ABComparison.from_two_entries(entry_a_id, entry_b_id)
        app_state.ab_comparison = comparison
        path_a, path_b = comparison.get_audio_paths(app_state.history_store)

        return (
            gr.update(visible=True, value=str(path_a) if path_a else None),
            gr.update(visible=True, value=str(path_b) if path_b else None),
            "Comparing A vs B. Listen and pick a winner!",
        )
    except Exception as exc:
        logger.warning("A/B comparison failed: %s", exc)
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            f"Comparison failed: {exc}",
        )


def _keep_winner(side: str):
    """Save the winning side's parameters as a preset.

    Args:
        side: "a" or "b".

    Returns:
        Status message string.
    """
    if app_state.ab_comparison is None:
        return "No active comparison. Start a comparison first."

    if app_state.history_store is None or app_state.preset_manager is None:
        return "History or preset manager not available."

    try:
        # Generate a preset name from the winner
        winner_entry_id = (
            app_state.ab_comparison.entry_a_id
            if side == "a"
            else app_state.ab_comparison.entry_b_id
        )
        if winner_entry_id is None:
            return "Winner is a live generation -- cannot save as preset."

        entry = app_state.history_store.get(winner_entry_id)
        if entry is None:
            return "Winner entry not found in history."

        preset_name = f"AB Winner (seed:{entry.seed})"

        app_state.ab_comparison.keep_winner(
            winner=side,
            preset_name=preset_name,
            history=app_state.history_store,
            preset_manager=app_state.preset_manager,
        )

        return f"Saved winner ({side.upper()}) as preset: {preset_name}"

    except Exception as exc:
        logger.warning("Keep winner failed: %s", exc)
        return f"Failed to save winner: {exc}"


# ---------------------------------------------------------------------------
# Model-type UI visibility helper
# ---------------------------------------------------------------------------


def _update_generate_tab_for_model():
    """Return visibility updates for the generate tab based on loaded model type.

    Returns a list of gr.update dicts for:
        [empty_msg, controls_section, prior_controls_section]

    - VQ-VAE model with prior loaded: show prior_controls, hide v1.0 controls, hide empty_msg
    - v1.0 model loaded: show v1.0 controls, hide prior_controls, hide empty_msg
    - Neither loaded: show empty_msg, hide both control sections
    """
    if (
        app_state.loaded_vq_model is not None
        and app_state.loaded_vq_model.prior is not None
    ):
        # VQ-VAE model with prior -- show prior controls
        return [
            gr.update(visible=False),   # empty_msg
            gr.update(visible=False),   # controls_section (v1.0)
            gr.update(visible=True),    # prior_controls_section
        ]
    elif app_state.loaded_model is not None:
        # v1.0 model -- show v1.0 controls
        return [
            gr.update(visible=False),   # empty_msg
            gr.update(visible=True),    # controls_section (v1.0)
            gr.update(visible=False),   # prior_controls_section
        ]
    else:
        # Nothing loaded -- show empty state
        return [
            gr.update(visible=True),    # empty_msg
            gr.update(visible=False),   # controls_section (v1.0)
            gr.update(visible=False),   # prior_controls_section
        ]


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------


def build_generate_tab() -> dict:
    """Build the Generate tab UI within the current Blocks context.

    Layout:
    - Empty state message (shown when no model loaded)
    - Prior controls section (visible when VQ-VAE model with prior loaded)
    - Controls section (hidden until v1.0 model loaded):
      - Multi-model blend accordion (collapsible)
      - Sliders in 3 columns (timbral / temporal / spatial)
      - Generation config (duration, output mode, spatial width/depth)
      - Seed row
      - Generate button
      - Audio player with quality badge
      - Export controls (format selector, metadata, sample rate, bit depth)
      - Preset management
      - History accordion (collapsible)
      - A/B Comparison accordion (collapsible)

    Returns:
        Component reference dict for cross-tab wiring.
    """
    # Empty state message
    empty_msg = gr.Markdown(
        value=get_empty_state_message("generate"),
        visible=True,
    )

    # ----- Prior-based controls section (v1.1) -----
    # Visible when VQ-VAE model with prior loaded
    with gr.Column(visible=False) as prior_controls_section:
        gr.Markdown("## Generate from Prior")

        # Sampling controls row
        with gr.Row():
            prior_temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.05,
                label="Temperature",
                info="Higher = more diverse, lower = more focused",
            )
            prior_top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p (Nucleus)",
                info="0 = disabled. Lower = more focused.",
            )
            prior_top_k = gr.Slider(
                minimum=0,
                maximum=512,
                value=0,
                step=1,
                label="Top-k",
                info="0 = disabled. Lower = fewer choices per step.",
            )

        # Duration and overlap row
        with gr.Row():
            prior_duration = gr.Slider(
                minimum=1,
                maximum=30,
                value=10,
                step=1,
                label="Duration (seconds)",
            )
            prior_overlap = gr.Slider(
                minimum=0,
                maximum=200,
                value=50,
                step=10,
                label="Crossfade Overlap (ms)",
                info="Overlap between chunks. 50ms default.",
            )

        # Seed row
        with gr.Row():
            prior_seed = gr.Number(
                label="Seed",
                value=None,
                precision=0,
            )
            prior_randomize_btn = gr.Button("Randomize", size="sm")

        # Generate button
        prior_generate_btn = gr.Button(
            "Generate", variant="primary", size="lg",
        )

        # Audio output section
        prior_audio_output = gr.Audio(
            label="Generated Audio",
            visible=False,
            type="numpy",
            interactive=False,
            autoplay=False,
        )
        prior_status_md = gr.Markdown(value="")

        # Simplified export section for prior-generated audio
        with gr.Accordion("Export", open=False):
            prior_export_format_dd = gr.Dropdown(
                choices=["wav", "mp3", "flac", "ogg"],
                value="wav",
                label="Format",
            )
            prior_export_btn = gr.Button("Export", variant="secondary")
            prior_export_status = gr.Markdown(value="")

    # Controls section -- hidden until v1.0 model loaded
    with gr.Column(visible=False) as controls_section:
        gr.Markdown("## Generate")

        # ----- Multi-model blend accordion (collapsible) -----
        with gr.Accordion("Multi-Model Blend", open=False):
            blend_mode_radio = gr.Radio(
                choices=["latent", "audio"],
                value="latent",
                label="Blend Mode",
            )
            # Pre-create up to 4 model rows (hidden by default)
            blend_model_dds: list[gr.Dropdown] = []
            blend_weight_sliders: list[gr.Slider] = []
            blend_rows: list[gr.Row] = []
            for slot_i in range(MAX_BLEND_SLOTS):
                with gr.Row(visible=(slot_i == 0)) as blend_row:
                    blend_dd = gr.Dropdown(
                        label=f"Model {slot_i + 1}",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    blend_w = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=25,
                        label=f"Weight {slot_i + 1}",
                        interactive=True,
                    )
                    blend_model_dds.append(blend_dd)
                    blend_weight_sliders.append(blend_w)
                    blend_rows.append(blend_row)
            with gr.Row():
                add_blend_model_btn = gr.Button("Add Model", size="sm")
                remove_blend_model_btn = gr.Button("Remove Last", size="sm")
            blend_status = gr.Markdown(
                value="Single model mode (blend inactive)."
            )

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
            output_mode_dd = gr.Dropdown(
                choices=["mono", "stereo", "binaural"],
                value="mono",
                label="Output Mode",
            )
            spatial_width_slider = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                step=0.1,
                value=0.7,
                label="Spatial Width",
                visible=False,
            )
            spatial_depth_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.5,
                label="Spatial Depth",
                visible=False,
            )
        with gr.Row():
            evolution_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.5,
                label="Evolution (latent drift for multi-second audio)",
                info="0 = static, 1 = maximum variation across chunks",
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
                export_format_dd = gr.Dropdown(
                    choices=["wav", "mp3", "flac", "ogg"],
                    value="wav",
                    label="Format",
                )
                export_sr_dd = gr.Dropdown(
                    choices=["44100", "48000", "96000"],
                    value="48000",
                    label="Sample Rate",
                )
                export_bd_dd = gr.Dropdown(
                    choices=["16-bit", "24-bit", "32-bit float"],
                    value="24-bit",
                    label="Bit Depth",
                    visible=True,
                )
                export_filename = gr.Textbox(
                    label="Filename",
                    placeholder="auto-generated",
                )
                with gr.Accordion("Export Metadata", open=False):
                    meta_artist = gr.Textbox(
                        label="Artist",
                        value="Distill Generator",
                    )
                    meta_album = gr.Textbox(
                        label="Album",
                        placeholder="model name",
                    )
                    meta_title = gr.Textbox(
                        label="Title",
                        placeholder="auto-generated",
                    )
                export_btn = gr.Button("Export", variant="secondary")
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

        # ----- History accordion (collapsible) -----
        with gr.Accordion("Generation History", open=False):
            history_gallery = gr.Gallery(
                label="History",
                columns=4,
                height=300,
                visible=True,
            )
            refresh_history_btn = gr.Button("Refresh History", size="sm")
            history_playback = gr.Audio(
                label="History Playback",
                visible=False,
                interactive=False,
            )

        # ----- A/B Comparison accordion (collapsible) -----
        with gr.Accordion("A/B Comparison", open=False):
            gr.Markdown("Select two generations from history to compare.")
            with gr.Row():
                ab_dropdown_a = gr.Dropdown(
                    label="Entry A",
                    choices=[],
                    value=None,
                )
                ab_dropdown_b = gr.Dropdown(
                    label="Entry B",
                    choices=[],
                    value=None,
                )
            compare_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**A**")
                    ab_audio_a = gr.Audio(
                        label="Generation A",
                        visible=False,
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("**B**")
                    ab_audio_b = gr.Audio(
                        label="Generation B",
                        visible=False,
                        interactive=False,
                    )
            with gr.Row():
                keep_a_btn = gr.Button("Keep A as Preset")
                keep_b_btn = gr.Button("Keep B as Preset")
            ab_status = gr.Markdown(value="")

    # ----- Wire event handlers -----

    # ----- Prior controls event wiring (v1.1) -----

    # Randomize seed (prior section)
    prior_randomize_btn.click(
        fn=_randomize_seed,
        inputs=None,
        outputs=[prior_seed],
    )

    # Generate from prior
    prior_generate_btn.click(
        fn=_generate_prior_audio,
        inputs=[
            prior_temperature,
            prior_top_k,
            prior_top_p,
            prior_duration,
            prior_overlap,
            prior_seed,
        ],
        outputs=[prior_audio_output, prior_status_md, prior_audio_output],
    )

    # Prior export (simplified -- exports last prior result as WAV)
    def _export_prior_audio(export_format: str):
        """Export the last prior-generated audio."""
        last = app_state.metrics_buffer.get("last_prior_result")
        if last is None:
            return "No prior audio to export. Generate audio first."
        try:
            from distill.inference.export import ExportFormat, export_audio
            from datetime import datetime, timezone

            fmt = ExportFormat(export_format)
            output_dir = app_state.generated_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            from distill.inference.export import FORMAT_EXTENSIONS

            ext = FORMAT_EXTENSIONS[fmt]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            fname = f"prior_{timestamp}_seed{last['seed_used']}{ext}"
            output_path = output_dir / fname
            export_audio(
                audio=last["audio"],
                path=output_path,
                sample_rate=48000,
                format=fmt,
                bit_depth="24-bit",
                metadata=None,
            )
            return f"Exported to `{output_path}`"
        except Exception as exc:
            logger.exception("Prior export failed")
            return f"**Export failed:** {exc}"

    prior_export_btn.click(
        fn=_export_prior_audio,
        inputs=[prior_export_format_dd],
        outputs=[prior_export_status],
    )

    # ----- v1.0 controls event wiring -----

    # Spatial controls visibility toggle
    output_mode_dd.change(
        fn=_toggle_spatial_controls,
        inputs=[output_mode_dd],
        outputs=[spatial_width_slider, spatial_depth_slider],
    )

    # Bit depth visibility based on export format
    export_format_dd.change(
        fn=_toggle_bit_depth,
        inputs=[export_format_dd],
        outputs=[export_bd_dd],
    )

    # Randomize seed
    randomize_btn.click(
        fn=_randomize_seed,
        inputs=None,
        outputs=[seed_input],
    )

    # Generate audio (now uses output_mode, spatial_width, spatial_depth, evolution)
    generate_btn.click(
        fn=_generate_audio,
        inputs=sliders + [
            duration_input,
            output_mode_dd,
            spatial_width_slider,
            spatial_depth_slider,
            seed_input,
            evolution_slider,
        ],
        outputs=[audio_output, quality_badge, audio_output,
                 history_gallery, ab_dropdown_a, ab_dropdown_b],
    )

    # Export audio (now includes format and metadata)
    export_btn.click(
        fn=_export_audio,
        inputs=[
            export_sr_dd,
            export_bd_dd,
            export_filename,
            export_format_dd,
            meta_artist,
            meta_album,
            meta_title,
        ],
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

    # ----- Blend panel event wiring -----

    # Add/remove blend model rows
    add_blend_model_btn.click(
        fn=_add_blend_row,
        inputs=None,
        outputs=[blend_status] + blend_rows,
    )

    remove_blend_model_btn.click(
        fn=_remove_blend_row,
        inputs=None,
        outputs=[blend_status] + blend_rows,
    )

    # ----- History event wiring -----

    # Refresh history button
    refresh_history_btn.click(
        fn=_refresh_history,
        inputs=None,
        outputs=[history_gallery, ab_dropdown_a, ab_dropdown_b],
    )

    # Click on gallery thumbnail plays the audio
    history_gallery.select(
        fn=_play_history_entry,
        inputs=None,
        outputs=[history_playback],
    )

    # ----- A/B Comparison event wiring -----

    # Compare button
    compare_btn.click(
        fn=_start_comparison,
        inputs=[ab_dropdown_a, ab_dropdown_b],
        outputs=[ab_audio_a, ab_audio_b, ab_status],
    )

    # Keep A / Keep B buttons
    keep_a_btn.click(
        fn=lambda: _keep_winner("a"),
        inputs=None,
        outputs=[ab_status],
    )

    keep_b_btn.click(
        fn=lambda: _keep_winner("b"),
        inputs=None,
        outputs=[ab_status],
    )

    # Return components that need external wiring (model loading)
    return {
        "sliders": sliders,
        "preset_dd": preset_dd,
        "controls_section": controls_section,
        "prior_controls_section": prior_controls_section,
        "empty_msg": empty_msg,
        "blend_model_dds": blend_model_dds,
    }
