"""``sda generate`` command -- generate audio from trained models.

Provides batch audio generation with model resolution (by name, ID, or
``.sda`` file path), preset support, and configurable output options.

Design notes:
- ``Console(stderr=True)`` for all Rich output (progress/status to stderr).
- ``print()`` for machine-readable output to stdout (file paths or JSON).
- All heavy imports lazy inside function bodies (project pattern).
- stdout gets file paths (or JSON). Enables piping: ``sda generate model -n 5 | xargs ls -la``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer()


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
    1. If ``model_ref`` ends in ``.sda`` and the path exists, load directly.
    2. Try UUID lookup via ``ModelLibrary.get()``.
    3. Try name search via ``ModelLibrary.search()``.

    Parameters
    ----------
    model_ref : str
        Model name, ID, or ``.sda`` file path.
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
    from small_dataset_audio.models.persistence import load_model

    # 1. Direct .sda file path
    if model_ref.endswith(".sda"):
        sda_path = Path(model_ref)
        if sda_path.exists():
            return load_model(sda_path, device=device)
        raise typer.BadParameter(f"Model file not found: {model_ref}")

    # 2. UUID lookup
    from small_dataset_audio.library.catalog import ModelLibrary

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
# Generate command
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def generate(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Model name, ID, or .sda file path"),
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
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir", "-o", help="Output directory (default: from config)"
        ),
    ] = None,
    stereo_mode: str = typer.Option(
        "mono", "--stereo", help="Stereo mode: mono, mid_side, dual_seed"
    ),
    sample_rate: int = typer.Option(
        48000, "--sample-rate", help="Output sample rate: 44100, 48000, 96000"
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
    """Generate audio from a trained model."""
    from rich.console import Console

    from small_dataset_audio.cli import bootstrap
    from small_dataset_audio.config.settings import resolve_path
    from small_dataset_audio.inference.generation import (
        GenerationConfig,
        GenerationPipeline,
    )

    console = Console(stderr=True)

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

    # Load model
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
            from small_dataset_audio.controls.mapping import (
                SliderState,
                sliders_to_latent,
            )
            from small_dataset_audio.presets.manager import PresetManager

            # Resolve presets directory
            presets_path_str = app_config.get("paths", {}).get("presets")
            if presets_path_str:
                presets_dir = resolve_path(presets_path_str, base_dir=config_path.parent)
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

    gen_config = GenerationConfig(
        duration_s=duration,
        seed=seed,
        stereo_mode=stereo_mode,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        latent_vector=latent_vector,
    )

    # Create pipeline
    pipeline = GenerationPipeline(loaded.model, loaded.spectrogram, torch_device)
    pipeline.model_name = loaded.metadata.name

    # Generate batch
    results = []
    for i in range(count):
        if seed is not None:
            gen_config.seed = seed + i

        result = pipeline.generate(gen_config)
        wav_path, json_path = pipeline.export(result, output_dir)
        results.append({
            "wav": str(wav_path),
            "json": str(json_path),
            "seed": result.seed_used,
        })
        console.print(f"[green]Generated:[/green] {wav_path}")

    # Output results
    if json_output:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(r["wav"])
