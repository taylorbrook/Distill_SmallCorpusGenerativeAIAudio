"""``distill generate`` command -- generate audio from trained models.

Provides batch audio generation with model resolution (by name, ID, or
``.distillgan`` file path), preset support, and configurable output options.
Supports multi-format export (WAV/MP3/FLAC/OGG), spatial audio
(mono/stereo/binaural), multi-model blending, and metadata embedding.

Design notes:
- ``Console(stderr=True)`` for all Rich output (progress/status to stderr).
- ``print()`` for machine-readable output to stdout (file paths or JSON).
- All heavy imports lazy inside function bodies (project pattern).
- stdout gets file paths (or JSON). Enables piping: ``distill generate model -n 5 | xargs ls -la``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer()

# Valid format choices
_VALID_FORMATS = ("wav", "mp3", "flac", "ogg")


# ---------------------------------------------------------------------------
# Model resolution helper
# ---------------------------------------------------------------------------


def resolve_model(
    model_ref: str,
    models_dir: Path,
    device: str,
) -> "LoadedModel":
    """Resolve a model reference to a loaded model.

    Resolution order:
    1. If ``model_ref`` ends in ``.distillgan`` and the path exists, load directly.
    2. Try UUID lookup via ``ModelLibrary.get()``.
    3. Try name search via ``ModelLibrary.search()``.

    Parameters
    ----------
    model_ref : str
        Model name, ID, or ``.distillgan`` file path.
    models_dir : Path
        Directory containing saved models.
    device : str
        Device string for model loading.

    Returns
    -------
    LoadedModel
        Loaded model ready for generation.

    Raises
    ------
    typer.BadParameter
        If model is not found or reference is ambiguous.
    """
    from distill.models.persistence import load_model

    # Reject v1 .distill format with clear error
    if model_ref.endswith(".distill"):
        raise typer.BadParameter(
            "This model was saved in v1 format (.distill) which is no longer "
            "supported. Please retrain your model."
        )

    # 1. Direct .distillgan file path
    if model_ref.endswith(".distillgan"):
        sda_path = Path(model_ref)
        if sda_path.exists():
            return load_model(sda_path, device=device)
        raise typer.BadParameter(f"Model file not found: {model_ref}")

    # 2. UUID lookup
    from distill.library.catalog import ModelLibrary

    library = ModelLibrary(models_dir)
    entry = library.get(model_ref)
    if entry is not None:
        return load_model(models_dir / entry.file_path, device=device)

    # 3. Name search
    results = library.search(query=model_ref)
    if len(results) == 1:
        return load_model(models_dir / results[0].file_path, device=device)
    if len(results) > 1:
        names = ", ".join(f"'{r.name}'" for r in results)
        raise typer.BadParameter(
            f"Ambiguous model reference '{model_ref}'. "
            f"Matches: {names}. Use the model ID to be specific."
        )

    raise typer.BadParameter(f"Model not found: {model_ref}")


# ---------------------------------------------------------------------------
# Blend argument parser
# ---------------------------------------------------------------------------


def _parse_blend_arg(blend_str: str) -> tuple[str, float]:
    """Parse a ``MODEL:WEIGHT`` blend argument.

    Splits on the last colon so model references containing colons
    (unlikely but defensive) still work.

    Returns
    -------
    tuple[str, float]
        ``(model_ref, weight)`` pair.

    Raises
    ------
    typer.BadParameter
        If the format is invalid.
    """
    if ":" not in blend_str:
        raise typer.BadParameter(
            f"Invalid blend argument '{blend_str}'. "
            "Expected format: MODEL:WEIGHT (e.g., 'my_model:60')"
        )
    last_colon = blend_str.rfind(":")
    model_ref = blend_str[:last_colon].strip()
    weight_str = blend_str[last_colon + 1:].strip()
    try:
        weight = float(weight_str)
    except ValueError:
        raise typer.BadParameter(
            f"Invalid weight '{weight_str}' in blend argument '{blend_str}'. "
            "Weight must be a number (e.g., 60)."
        )
    if not model_ref:
        raise typer.BadParameter(
            f"Empty model reference in blend argument '{blend_str}'."
        )
    return model_ref, weight


# ---------------------------------------------------------------------------
# Generate command
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def generate(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Model name, ID, or .distillgan file path"),
    duration: float = typer.Option(
        1.0, "--duration", "-d", help="Duration in seconds (max 60)"
    ),
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", "-s", help="Random seed (auto-increments for batch)"),
    ] = None,
    count: int = typer.Option(1, "--count", "-n", help="Number of files to generate"),
    preset: Annotated[
        Optional[str],
        typer.Option("--preset", "-p", help="Preset name to load (model-scoped)"),
    ] = None,
    slider: Annotated[
        Optional[list[str]],
        typer.Option(
            "--slider",
            help="Set slider position as INDEX:VALUE (e.g., --slider '0:5' --slider '1:-3'). Range: -10 to 10.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir", "-o", help="Output directory (default: from config)"
        ),
    ] = None,
    # --- Format options (Phase 10) ---
    format_: str = typer.Option(
        "wav", "--format", "-f", help="Export format: wav, mp3, flac, ogg"
    ),
    # --- Spatial audio options (Phase 10) ---
    spatial_mode: str = typer.Option(
        "mono", "--spatial-mode", help="Spatial mode: mono, stereo, binaural"
    ),
    spatial_width: float = typer.Option(
        0.7, "--spatial-width", help="Spatial width (0.0-1.5)"
    ),
    spatial_depth: float = typer.Option(
        0.5, "--spatial-depth", help="Spatial depth (0.0-1.0)"
    ),
    # --- Blend options (Phase 10) ---
    blend: Annotated[
        Optional[list[str]],
        typer.Option(
            "--blend", "-b",
            help="Blend models as MODEL:WEIGHT pairs (e.g., --blend 'model_a:60' --blend 'model_b:40')",
        ),
    ] = None,
    # --- Metadata override options (Phase 10) ---
    meta_artist: Annotated[
        Optional[str],
        typer.Option("--artist", help="Override artist metadata tag"),
    ] = None,
    meta_album: Annotated[
        Optional[str],
        typer.Option("--album", help="Override album metadata tag"),
    ] = None,
    meta_title: Annotated[
        Optional[str],
        typer.Option("--title", help="Override title metadata tag"),
    ] = None,
    # --- Legacy options ---
    stereo: Annotated[
        Optional[str],
        typer.Option(
            "--stereo",
            help="[deprecated] Use --spatial-mode instead. Legacy stereo mode: mono, mid_side, dual_seed",
        ),
    ] = None,
    sample_rate: int = typer.Option(
        44100, "--sample-rate", help="Output sample rate: 44100, 48000, 96000"
    ),
    bit_depth: str = typer.Option(
        "24-bit", "--bit-depth", help="Bit depth: 16-bit, 24-bit, 32-bit-float"
    ),
    device: str = typer.Option(
        "auto", "--device", help="Compute device: auto, mps, cuda, cpu"
    ),
    config: Annotated[
        Optional[Path], typer.Option("--config", help="Config file path")
    ] = None,
    json_output: bool = typer.Option(
        False, "--json", help="Output results as JSON to stdout"
    ),
) -> None:
    """Generate audio from a trained model.

    Supports multi-format export, spatial audio, model blending, and
    metadata embedding.
    """
    from rich.console import Console

    from distill.audio.metadata import build_export_metadata
    from distill.cli import bootstrap
    from distill.config.settings import resolve_path
    from distill.inference.export import ExportFormat, FORMAT_EXTENSIONS, export_audio
    from distill.inference.generation import (
        GenerationConfig,
        GenerationPipeline,
    )
    from distill.inference.spatial import (
        SpatialConfig,
        SpatialMode,
        migrate_stereo_config,
    )

    console = Console(stderr=True)

    # ---- Validate format ----
    fmt_lower = format_.lower().strip()
    if fmt_lower not in _VALID_FORMATS:
        raise typer.BadParameter(
            f"Invalid format '{format_}'. Must be one of: {', '.join(_VALID_FORMATS)}"
        )
    export_format = ExportFormat(fmt_lower)
    file_ext = FORMAT_EXTENSIONS[export_format].lstrip(".")

    # ---- Handle deprecated --stereo flag ----
    if stereo is not None:
        console.print(
            "[yellow]Warning:[/yellow] --stereo is deprecated. "
            "Use --spatial-mode instead."
        )
        # Only use stereo flag if spatial_mode was not explicitly set
        if spatial_mode == "mono":
            migrated = migrate_stereo_config(stereo, spatial_width)
            spatial_config = migrated
        else:
            # User provided both --stereo and --spatial-mode; prefer --spatial-mode
            spatial_config = SpatialConfig(
                mode=SpatialMode(spatial_mode),
                width=spatial_width,
                depth=spatial_depth,
            )
    else:
        spatial_config = SpatialConfig(
            mode=SpatialMode(spatial_mode),
            width=spatial_width,
            depth=spatial_depth,
        )

    spatial_config.validate()

    # Bootstrap config and device
    app_config, torch_device, config_path = bootstrap(config, device)

    # Resolve output directory
    if output_dir is None:
        output_dir = resolve_path(
            app_config["paths"]["generated"], base_dir=config_path.parent
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve models directory
    models_dir = resolve_path(
        app_config["paths"]["models"], base_dir=config_path.parent
    )

    # ---- Build metadata overrides ----
    meta_overrides: dict[str, str] = {}
    if meta_artist is not None:
        meta_overrides["artist"] = meta_artist
    if meta_album is not None:
        meta_overrides["album"] = meta_album
    if meta_title is not None:
        meta_overrides["title"] = meta_title

    # ---- Handle blend mode ----
    if blend:
        if slider is not None:
            console.print(
                "[yellow]Warning:[/yellow] --slider is not supported with --blend. "
                "Using neutral slider positions for blend."
            )

        from distill.inference.blending import BlendEngine

        engine = BlendEngine()

        for blend_arg in blend:
            ref, weight = _parse_blend_arg(blend_arg)
            console.print(f"[dim]Loading blend model: {ref} (weight={weight})[/dim]")
            blend_loaded = resolve_model(ref, models_dir, str(torch_device))
            engine.add_model(
                model=blend_loaded.model,
                spectrogram=blend_loaded.spectrogram,
                analysis=blend_loaded.analysis,
                metadata=blend_loaded.metadata,
                device=torch_device,
                weight=weight,
            )
            console.print(
                f"[green]Blend loaded:[/green] {blend_loaded.metadata.name} "
                f"(weight={weight})"
            )

        # Also load the primary model argument into the blend engine
        console.print(f"[dim]Loading primary model: {model}[/dim]")
        loaded = resolve_model(model, models_dir, str(torch_device))
        # Primary model gets remaining weight (or 50 if not specified)
        primary_weight = 50.0
        engine.add_model(
            model=loaded.model,
            spectrogram=loaded.spectrogram,
            analysis=loaded.analysis,
            metadata=loaded.metadata,
            device=torch_device,
            weight=primary_weight,
        )
        console.print(
            f"[green]Primary loaded:[/green] {loaded.metadata.name} "
            f"(weight={primary_weight}, device: {torch_device.type})"
        )

        model_name_for_meta = " + ".join(
            s.metadata.name for s in engine.get_active_slots()
        )

        # Build generation config for blending (uses old stereo fields for
        # internal pipeline compatibility, spatial applied post-generation)
        gen_config = GenerationConfig(
            duration_s=duration,
            seed=seed,
            stereo_mode="mono",  # spatial applied separately
            sample_rate=sample_rate,
            bit_depth=bit_depth,
        )

        # Generate batch via blend engine
        results = []
        for i in range(count):
            if seed is not None:
                gen_config.seed = seed + i

            result = engine.blend_generate(
                slider_positions=[0] * 12,  # neutral positions
                config=gen_config,
            )

            # Apply spatial processing to blended result
            audio = _apply_spatial_post(result.audio, spatial_config, result.sample_rate)

            # Build metadata
            metadata = build_export_metadata(
                model_name=model_name_for_meta,
                seed=result.seed_used,
                preset_name=preset,
                overrides=meta_overrides if meta_overrides else None,
            )

            # Export with format and metadata
            export_path = _export_result(
                audio=audio,
                result=result,
                output_dir=output_dir,
                export_format=export_format,
                file_ext=file_ext,
                metadata=metadata,
            )

            results.append({
                "file": str(export_path),
                "format": fmt_lower,
                "seed": result.seed_used,
            })
            console.print(
                f"[green]Generated:[/green] {export_path} ({fmt_lower})"
            )

    else:
        # ---- Single model mode ----
        console.print(f"[dim]Loading model: {model}[/dim]")
        loaded = resolve_model(model, models_dir, str(torch_device))
        console.print(
            f"[green]Loaded:[/green] {loaded.metadata.name} "
            f"(device: {torch_device.type})"
        )

        # Build generation config
        latent_vector = None

        # Handle preset loading (requires model context)
        if preset is not None:
            if loaded.analysis is None:
                console.print(
                    "[yellow]Warning:[/yellow] Model has no latent analysis. "
                    "Preset ignored."
                )
            else:
                from distill.controls.mapping import (
                    SliderState,
                    sliders_to_latent,
                )
                from distill.presets.manager import PresetManager

                # Resolve presets directory
                presets_path_str = app_config.get("paths", {}).get("presets")
                if presets_path_str:
                    presets_dir = resolve_path(
                        presets_path_str, base_dir=config_path.parent
                    )
                else:
                    presets_dir = config_path.parent / "data" / "presets"

                pm = PresetManager(presets_dir, loaded.metadata.model_id)
                all_presets = pm.list_presets()

                # Find preset by name (case-insensitive)
                matched = [
                    p for p in all_presets if p.name.lower() == preset.lower()
                ]
                if not matched:
                    raise typer.BadParameter(
                        f"Preset '{preset}' not found for model "
                        f"'{loaded.metadata.name}'"
                    )

                preset_entry = matched[0]
                slider_state = SliderState(
                    positions=list(preset_entry.slider_positions),
                    n_components=len(preset_entry.slider_positions),
                )
                latent_vector = sliders_to_latent(slider_state, loaded.analysis)
                console.print(
                    f"[green]Preset loaded:[/green] {preset_entry.name}"
                )

        # Handle --slider option (direct slider position control)
        if slider is not None and latent_vector is None:
            if loaded.analysis is None:
                console.print(
                    "[yellow]Warning:[/yellow] Model has no latent analysis. "
                    "--slider ignored, using random latent vectors."
                )
            else:
                from distill.controls.mapping import (
                    SliderState,
                    sliders_to_latent,
                )

                n_active = loaded.analysis.n_active_components
                positions = [0] * n_active  # default to center (neutral)

                for s in slider:
                    if ":" not in s:
                        raise typer.BadParameter(
                            f"Invalid slider format '{s}'. Expected INDEX:VALUE (e.g., '0:5')"
                        )
                    idx_str, val_str = s.split(":", 1)
                    try:
                        idx = int(idx_str)
                        val = int(val_str)
                    except ValueError:
                        raise typer.BadParameter(
                            f"Invalid slider values in '{s}'. INDEX and VALUE must be integers."
                        )
                    if idx < 0 or idx >= n_active:
                        raise typer.BadParameter(
                            f"Slider index {idx} out of range. Model has {n_active} active components (0-{n_active - 1})."
                        )
                    if val < -10 or val > 10:
                        raise typer.BadParameter(
                            f"Slider value {val} out of range. Must be between -10 and 10."
                        )
                    positions[idx] = val

                slider_state = SliderState(positions=positions, n_components=n_active)
                latent_vector = sliders_to_latent(slider_state, loaded.analysis)
                console.print(
                    f"[green]Sliders applied:[/green] {len(slider)} position(s) set"
                )

        gen_config = GenerationConfig(
            duration_s=duration,
            seed=seed,
            stereo_mode="mono",  # spatial applied post-generation
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            latent_vector=latent_vector,
        )

        # Create pipeline with vocoder
        from distill.vocoder import get_vocoder

        vocoder = get_vocoder("bigvgan", device=str(torch_device))
        pipeline = GenerationPipeline(
            loaded.model, loaded.spectrogram, torch_device, vocoder=vocoder
        )
        pipeline.model_name = loaded.metadata.name

        # Generate batch
        results = []
        for i in range(count):
            if seed is not None:
                gen_config.seed = seed + i

            result = pipeline.generate(gen_config)

            # Apply spatial processing
            audio = _apply_spatial_post(
                result.audio, spatial_config, result.sample_rate
            )

            # Build metadata
            metadata = build_export_metadata(
                model_name=loaded.metadata.name,
                seed=result.seed_used,
                preset_name=preset,
                overrides=meta_overrides if meta_overrides else None,
            )

            # Export with format and metadata
            export_path = _export_result(
                audio=audio,
                result=result,
                output_dir=output_dir,
                export_format=export_format,
                file_ext=file_ext,
                metadata=metadata,
            )

            results.append({
                "file": str(export_path),
                "format": fmt_lower,
                "seed": result.seed_used,
            })
            console.print(
                f"[green]Generated:[/green] {export_path} ({fmt_lower})"
            )

    # Output results
    if json_output:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(r["file"])


# ---------------------------------------------------------------------------
# Spatial post-processing helper
# ---------------------------------------------------------------------------


def _apply_spatial_post(
    audio: "np.ndarray",
    spatial_config: "SpatialConfig",
    sample_rate: int,
) -> "np.ndarray":
    """Apply spatial audio processing to generated mono audio.

    If the audio is already stereo (from stereo generation mode) or
    spatial mode is MONO, returns audio unchanged. Otherwise applies
    the configured spatial mode (stereo mid-side or binaural HRTF).

    Parameters
    ----------
    audio : np.ndarray
        Audio data (1-D mono or 2-D stereo).
    spatial_config : SpatialConfig
        Spatial audio configuration.
    sample_rate : int
        Audio sample rate.

    Returns
    -------
    np.ndarray
        Processed audio.
    """
    from distill.inference.spatial import SpatialMode, apply_spatial

    if spatial_config.mode == SpatialMode.MONO:
        return audio

    # Only apply spatial to mono audio
    import numpy as np  # noqa: WPS433

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return apply_spatial(audio, spatial_config, sample_rate)

    # Already stereo -- return as-is
    return audio


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------


def _export_result(
    audio: "np.ndarray",
    result: "GenerationResult",
    output_dir: Path,
    export_format: "ExportFormat",
    file_ext: str,
    metadata: dict,
) -> Path:
    """Export generated audio in the requested format with metadata.

    Writes sidecar JSON first (research pitfall #6), then the audio file
    in the requested format, then embeds metadata tags.

    Parameters
    ----------
    audio : np.ndarray
        Audio data to export.
    result : GenerationResult
        Generation result (for seed, config, quality metadata).
    output_dir : Path
        Output directory.
    export_format : ExportFormat
        Target export format.
    file_ext : str
        File extension without dot (e.g., "mp3").
    metadata : dict
        Metadata dict for tag embedding.

    Returns
    -------
    Path
        Path to the exported audio file.
    """
    from dataclasses import asdict
    from datetime import datetime, timezone

    from distill.inference.export import (
        ExportFormat,
        export_audio,
        write_sidecar_json,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-generate filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"gen_{timestamp}_seed{result.seed_used}.{file_ext}"
    export_path = output_dir / filename

    # Build config dict for sidecar JSON
    config_dict = asdict(result.config)
    if config_dict.get("latent_vector") is not None:
        config_dict["latent_vector"] = config_dict["latent_vector"].tolist()

    # Write sidecar JSON first (research pitfall #6)
    import numpy as np  # noqa: WPS433

    audio_arr = np.asarray(audio, dtype=np.float32)
    channels = 1 if audio_arr.ndim == 1 else audio_arr.shape[0]

    write_sidecar_json(
        wav_path=export_path,
        model_name=metadata.get("model_name", "unknown"),
        generation_config=config_dict,
        seed=result.seed_used,
        quality_metrics=result.quality,
        duration_s=result.duration_s,
        sample_rate=result.sample_rate,
        bit_depth=result.config.bit_depth,
        channels=channels,
    )

    # Export audio in requested format with metadata embedding
    export_audio(
        audio=audio_arr,
        path=export_path,
        sample_rate=result.sample_rate,
        format=export_format,
        bit_depth=result.config.bit_depth,
        metadata=metadata,
    )

    return export_path
