# Phase 9: CLI Interface - Research

**Researched:** 2026-02-14
**Domain:** Python CLI frameworks, subcommand architecture, progress display, batch generation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Operations exposed: generate, train, model management (list/info/delete)
- Presets are loadable from CLI (read-only -- preset creation/management stays in GUI)
- Include `sda ui` (or equivalent) subcommand to launch Gradio GUI from the CLI entry point
- Training progress: tqdm-style progress bar with epoch/loss/ETA
- Output defaults to project's configured output directory (from config), with --output-dir override
- No dry-run mode -- generation is fast enough that it's unnecessary

### Claude's Discretion
- CLI framework choice (argparse, click, typer, etc.)
- Command naming conventions
- Default verbosity level
- JSON output support
- Stderr/stdout routing
- Batch specification syntax
- Parameter sweep support
- Model resolution strategy
- Path override mechanism (flags vs env vars vs both)
- Training config flag design
- Exit codes and error message formatting
- Help text style and depth
- Subcommand vs flat style (recommend subcommand-based given 3 operation domains)
- Entry point command name (recommend short, ergonomic)
- --json flag for machine-readable output

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Summary

Phase 9 wraps existing Phase 1-8 functionality in a CLI for batch generation, training, and model management. The project already has a `sda` entry point defined in `pyproject.toml` (`[project.scripts] sda = "small_dataset_audio.app:main"`) and uses `argparse` for basic --device/--verbose/--benchmark/--config flags. The current `main()` function in `app.py` defaults to launching the Gradio UI after startup validation.

The recommended approach is to adopt **Typer** (v0.23.1) as the CLI framework, which already depends on both Click and Rich (both already in the project's dependency tree). Typer's decorator-based subcommand system with type hints provides clean code, automatic help generation, and native Rich integration for progress bars. The existing `app.py` entry point must be refactored: the current flat argparse approach becomes a Typer app with subcommands (`sda generate`, `sda train`, `sda model`, `sda ui`), where the default behavior (no subcommand) launches the GUI for backward compatibility.

The CLI module should live in a new `src/small_dataset_audio/cli/` package, with the `app.py` entry point updated to dispatch to the Typer app. All heavy work (generation, training, model loading) is done by existing modules -- the CLI is purely a thin orchestration layer that constructs configs, calls existing APIs, and displays progress/results.

**Primary recommendation:** Use Typer 0.23.1 with Rich progress bars, subcommand architecture (`generate`, `train`, `model`, `ui`), and keep CLI code as a thin wrapper over existing Phase 1-8 APIs.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| typer | >=0.23,<1.0 | CLI framework with subcommands | Type-hint-based, auto help, built on Click, depends on Rich (already in project) |
| rich | >=14.0 (already installed) | Progress bars, formatted output, tables | Already a project dependency, Typer depends on it |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| click | (transitive via typer) | Underlying CLI engine | Not imported directly; Typer wraps it |
| shellingham | (transitive via typer) | Shell detection for completions | Automatic via Typer |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Typer | argparse (current) | Already in project, but verbose for subcommands, no auto Rich integration, manual help text |
| Typer | Click | More manual wiring, no type-hint inference, but more control |

**Installation:**
```bash
uv add "typer>=0.23,<1.0"
```

Note: `rich` is already a dependency. Typer also depends on Rich, so versions will be compatible.

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── cli/                    # NEW: CLI command modules
│   ├── __init__.py         # Typer app creation, subcommand registration
│   ├── generate.py         # sda generate command
│   ├── train.py            # sda train command
│   ├── model.py            # sda model list/info/delete commands
│   └── ui.py               # sda ui command (launches Gradio)
├── app.py                  # MODIFIED: dispatch to CLI or legacy behavior
└── ... (existing modules unchanged)
```

### Pattern 1: Typer App with Sub-Typers
**What:** Main app registers sub-typer instances for each command domain
**When to use:** Always -- this is the primary architecture
**Example:**
```python
# src/small_dataset_audio/cli/__init__.py
import typer

app = typer.Typer(
    name="sda",
    help="Small Dataset Audio - Generative audio from small personal datasets",
    no_args_is_help=False,  # Allow default behavior (launch UI)
)

# Import and register sub-typers
from small_dataset_audio.cli.generate import app as generate_app
from small_dataset_audio.cli.model import app as model_app
from small_dataset_audio.cli.train import app as train_app

app.add_typer(generate_app, name="generate", help="Generate audio from trained models")
app.add_typer(train_app, name="train", help="Train models on audio datasets")
app.add_typer(model_app, name="model", help="Manage saved models")

@app.command()
def ui(
    device: str = typer.Option("auto", help="Compute device"),
    config: Path = typer.Option(None, help="Config file path"),
):
    """Launch the Gradio web UI."""
    ...
```

### Pattern 2: Shared Bootstrap (Config + Device)
**What:** Factor out config loading and device selection shared across all commands
**When to use:** Every command needs config and device
**Example:**
```python
# src/small_dataset_audio/cli/__init__.py
from pathlib import Path
from small_dataset_audio.config.settings import get_config_path, load_config
from small_dataset_audio.hardware.device import select_device

def bootstrap(config_path: Path | None = None, device: str = "auto"):
    """Load config and select device. Shared by all commands."""
    path = config_path or get_config_path()
    config = load_config(path)
    torch_device = select_device(
        config.get("hardware", {}).get("device", "auto")
        if device == "auto" else device
    )
    return config, torch_device, path
```

### Pattern 3: Rich Progress for Training
**What:** Use Rich's Progress with custom columns to show epoch/loss/ETA
**When to use:** Training command
**Example:**
```python
# Source: Rich docs - https://rich.readthedocs.io/en/stable/progress.html
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

def make_training_progress():
    return Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]}/{task.fields[total]}"),
        BarColumn(),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TextColumn("val={task.fields[val_loss]:.4f}"),
        TimeRemainingColumn(),
    )

# In training callback:
def cli_training_callback(event):
    if isinstance(event, EpochMetrics):
        progress.update(
            task_id,
            advance=1,
            epoch=event.epoch + 1,
            total=event.total_epochs,
            loss=event.train_loss,
            val_loss=event.val_loss,
        )
```

### Pattern 4: Model Resolution (library name OR file path)
**What:** Accept either a model name/ID from the library or a direct .sda file path
**When to use:** Any command needing a model (generate, model info)
**Example:**
```python
def resolve_model(model_ref: str, models_dir: Path, device: str) -> LoadedModel:
    """Resolve a model reference to a LoadedModel.

    Accepts:
    - A file path ending in .sda
    - A model name (searched in library)
    - A model ID (UUID lookup)
    """
    from small_dataset_audio.models.persistence import load_model
    from small_dataset_audio.library.catalog import ModelLibrary

    model_path = Path(model_ref)
    if model_path.suffix == ".sda" and model_path.exists():
        return load_model(model_path, device=device)

    # Search library by name or ID
    library = ModelLibrary(models_dir)
    entry = library.get(model_ref)  # Try as ID
    if entry is None:
        results = library.search(query=model_ref)
        if len(results) == 1:
            entry = results[0]
        elif len(results) > 1:
            raise typer.BadParameter(
                f"Ambiguous model name '{model_ref}'. "
                f"Matches: {', '.join(r.name for r in results)}"
            )
        else:
            raise typer.BadParameter(f"Model not found: {model_ref}")

    return load_model(models_dir / entry.file_path, device=device)
```

### Anti-Patterns to Avoid
- **Importing torch at CLI parse time:** Always lazy-import torch inside command functions, not at module level. The CLI should parse and validate arguments before triggering heavy imports. The project already follows this pattern everywhere.
- **Duplicating business logic in CLI:** The CLI should construct config objects and call existing APIs (GenerationPipeline, TrainingRunner, ModelLibrary), never re-implement generation/training logic.
- **Blocking the main thread for training:** For CLI training, use `train()` directly (not `TrainingRunner` with background thread) since the CLI IS the main process. The background thread pattern is for the GUI. The training loop already accepts a `callback` and `cancel_event` directly.
- **Hardcoding paths:** Always resolve paths through `config.settings.resolve_path()` and `get_config_path()`. Default output directory comes from config `paths.generated`, not cwd.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Argument parsing | Custom parser | Typer decorators | Type validation, help gen, shell completion |
| Progress bars | Print-based progress | Rich Progress | Flicker-free, custom columns, stderr-safe |
| Colored output | ANSI escape codes | Rich Console | Cross-platform, auto-detects terminal |
| Config loading | New config system | Existing `config.settings` module | Already handles TOML, defaults, path resolution |
| Model loading | New model loader | Existing `models.persistence.load_model` | Already handles format validation, device placement, analysis restoration |
| Generation pipeline | Direct model calls | Existing `GenerationPipeline` | Handles chunking, stereo, export, quality |
| Training orchestration | Custom training loop | Existing `training.loop.train()` | Already has callback system, checkpointing, cancellation |

**Key insight:** Phase 9 is a thin wrapper. Every line of CLI code that duplicates existing module logic is a maintenance burden and a divergence risk. The CLI constructs config objects, calls existing APIs, and formats results for the terminal.

## Common Pitfalls

### Pitfall 1: Breaking the Existing `sda` Entry Point
**What goes wrong:** Replacing `app.py:main` with a Typer app breaks `sda` for users who expect it to launch the GUI without arguments.
**Why it happens:** Typer's default `no_args_is_help=True` shows help text instead of doing something useful.
**How to avoid:** Set `no_args_is_help=False` on the main Typer app. Add a callback that detects no subcommand and launches the GUI (same as current behavior). Or use Typer's `invoke_without_command=True` on the app callback.
**Warning signs:** Running bare `sda` shows help text instead of launching the GUI.

### Pitfall 2: Progress Bars on stdout Corrupting Pipeable Output
**What goes wrong:** Progress bars written to stdout mix with generated file paths or JSON output, breaking pipe workflows (`sda generate ... | xargs ...`).
**Why it happens:** Rich defaults to stdout for display.
**How to avoid:** Create Rich Console with `stderr=True` for all progress/status output. Only write machine-readable results (file paths, JSON) to stdout.
**Warning signs:** `sda generate --json ... 2>/dev/null | jq .` fails to parse.

### Pitfall 3: Training in CLI Uses Wrong Pattern (Background Thread)
**What goes wrong:** Using `TrainingRunner` (which spawns a daemon thread) instead of calling `train()` directly, then needing to handle thread lifecycle in CLI.
**Why it happens:** Copying the GUI's training integration pattern.
**How to avoid:** Call `training.loop.train()` directly with a CLI-specific callback. The training loop already supports `callback` and `cancel_event` parameters. Use `signal.signal(SIGINT, handler)` to set cancel_event on Ctrl+C.
**Warning signs:** Training silently exits without saving checkpoint on Ctrl+C.

### Pitfall 4: Lazy Import Timing for Fast CLI Startup
**What goes wrong:** Importing torch/torchaudio at module level makes `sda --help` take 5-10 seconds.
**Why it happens:** Python imports are eager; torch is heavy.
**How to avoid:** Keep all torch/model imports inside command functions (inside the `@app.command()` decorated functions), never at module top. The existing codebase already follows this pattern -- maintain it in CLI modules.
**Warning signs:** `time sda --help` takes more than 0.5 seconds.

### Pitfall 5: Preset Loading Without Model Context
**What goes wrong:** User specifies `--preset mypreset` but presets are model-scoped. Without knowing which model, can't find the preset.
**Why it happens:** PresetManager requires model_id in constructor.
**How to avoid:** Require model specification before preset: `sda generate --model mymodel --preset mypreset`. Resolve model first, then use model_id to instantiate PresetManager.
**Warning signs:** "Preset not found" errors when the preset exists but for a different model.

### Pitfall 6: Seed Reproducibility Across CLI Invocations
**What goes wrong:** Same seed produces different output across runs.
**Why it happens:** Not setting torch manual seed deterministically, or GPU nondeterminism.
**How to avoid:** Document that seed reproducibility is best-effort (same as GenerationPipeline behavior). Pass seed through GenerationConfig which handles torch.manual_seed internally.
**Warning signs:** Users report seed X gives different output each time.

## Code Examples

Verified patterns from the existing codebase:

### Generate Command Structure
```python
# src/small_dataset_audio/cli/generate.py
from pathlib import Path
from typing import Optional
import typer

app = typer.Typer()

@app.command()
def generate(
    model: str = typer.Argument(..., help="Model name, ID, or .sda file path"),
    duration: float = typer.Option(1.0, "--duration", "-d", help="Duration in seconds"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    count: int = typer.Option(1, "--count", "-n", help="Number of files to generate"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Preset name to load"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    stereo_mode: str = typer.Option("mono", "--stereo", help="Stereo mode: mono, mid_side, dual_seed"),
    sample_rate: int = typer.Option(48000, "--sample-rate", help="Output sample rate"),
    bit_depth: str = typer.Option("24-bit", "--bit-depth", help="Output bit depth"),
    device: str = typer.Option("auto", "--device", help="Compute device"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
):
    """Generate audio from a trained model."""
    from rich.console import Console

    from small_dataset_audio.cli import bootstrap
    from small_dataset_audio.config.settings import resolve_path
    from small_dataset_audio.inference.generation import GenerationConfig, GenerationPipeline
    from small_dataset_audio.models.persistence import load_model

    console = Console(stderr=True)  # Progress/status to stderr
    app_config, torch_device, config_path = bootstrap(config, device)

    # Resolve output directory
    if output_dir is None:
        output_dir = resolve_path(
            app_config["paths"]["generated"],
            base_dir=config_path.parent,
        )

    # Resolve model
    models_dir = resolve_path(app_config["paths"]["models"], base_dir=config_path.parent)
    loaded = resolve_model(model, models_dir, str(torch_device))

    # Build generation config
    gen_config = GenerationConfig(
        duration_s=duration,
        seed=seed,
        stereo_mode=stereo_mode,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
    )

    # Apply preset if specified
    if preset:
        presets_dir = resolve_path(app_config["paths"]["presets"], base_dir=config_path.parent)
        # ... load preset, apply slider positions to gen_config ...

    # Create pipeline and generate
    pipeline = GenerationPipeline(loaded.model, loaded.spectrogram, torch_device)
    pipeline.model_name = loaded.metadata.name

    results = []
    for i in range(count):
        if seed is not None:
            gen_config.seed = seed + i  # Increment seed for batch
        result = pipeline.generate(gen_config)
        wav_path, json_path = pipeline.export(result, output_dir)
        results.append({"wav": str(wav_path), "json": str(json_path), "seed": result.seed_used})
        console.print(f"[green]Generated:[/green] {wav_path}")

    # Machine-readable output to stdout
    if json_output:
        import json as json_mod
        print(json_mod.dumps(results, indent=2))
    else:
        for r in results:
            print(r["wav"])  # stdout: file paths only
```

### Train Command with Rich Progress
```python
# src/small_dataset_audio/cli/train.py
import signal
import threading
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, SpinnerColumn

app = typer.Typer()

@app.command()
def train(
    dataset_dir: Path = typer.Argument(..., help="Path to dataset directory"),
    preset: str = typer.Option("auto", help="Training preset: auto, conservative, balanced, aggressive"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Override max epochs"),
    learning_rate: Optional[float] = typer.Option(None, "--lr", help="Override learning rate"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override batch size"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Training output dir"),
    device: str = typer.Option("auto", "--device", help="Compute device"),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
):
    """Train a model on an audio dataset."""
    console = Console(stderr=True)

    # ... bootstrap, collect files, build TrainingConfig ...

    # Setup Rich progress bar
    cancel_event = threading.Event()

    def handle_sigint(signum, frame):
        console.print("\n[yellow]Cancelling training (saving checkpoint)...[/yellow]")
        cancel_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Epoch {task.fields[epoch]}/{task.fields[total]}"),
        BarColumn(),
        TextColumn("train={task.fields[train_loss]:.4f}"),
        TextColumn("val={task.fields[val_loss]:.4f}"),
        TimeRemainingColumn(),
        console=console,  # stderr
    ) as progress:
        task = progress.add_task(
            "Training", total=training_config.max_epochs,
            epoch=0, total=training_config.max_epochs,
            train_loss=0.0, val_loss=0.0,
        )

        def cli_callback(event):
            from small_dataset_audio.training.metrics import EpochMetrics, TrainingCompleteEvent
            if isinstance(event, EpochMetrics):
                progress.update(
                    task, completed=event.epoch + 1,
                    epoch=event.epoch + 1,
                    total=event.total_epochs,
                    train_loss=event.train_loss,
                    val_loss=event.val_loss,
                )

        # Call train() directly (NOT TrainingRunner)
        from small_dataset_audio.training.loop import train as run_training
        result = run_training(
            config=training_config,
            file_paths=file_paths,
            output_dir=output_dir,
            device=torch_device,
            callback=cli_callback,
            cancel_event=cancel_event,
        )
```

### Model Management Commands
```python
# src/small_dataset_audio/cli/model.py
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()

@app.command("list")
def list_models(
    config: Path = typer.Option(None, "--config", help="Config file path"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
):
    """List all models in the library."""
    console = Console(stderr=True)
    # ... bootstrap, load ModelLibrary ...
    library = ModelLibrary(models_dir)
    entries = library.list_all()

    if json_output:
        import json
        print(json.dumps([asdict(e) for e in entries], indent=2))
        return

    table = Table(title="Model Library")
    table.add_column("Name", style="bold")
    table.add_column("Dataset")
    table.add_column("Epochs", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Saved")
    table.add_column("ID", style="dim")

    for entry in entries:
        table.add_row(
            entry.name, entry.dataset_name,
            str(entry.training_epochs),
            f"{entry.final_val_loss:.4f}",
            entry.save_date[:10],
            entry.model_id[:8],
        )
    console.print(table)

@app.command("info")
def model_info(model: str = typer.Argument(..., help="Model name or ID")):
    """Show detailed info about a model."""
    ...

@app.command("delete")
def delete_model_cmd(
    model: str = typer.Argument(..., help="Model name or ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a model from the library."""
    ...
```

## Discretion Recommendations

Based on research, here are specific recommendations for all Claude's Discretion items:

### CLI Framework: Typer 0.23.x
**Rationale:** Typer depends on both Click and Rich (already in project). Type-hint-based arguments eliminate boilerplate. Auto-generated help. The project already uses Rich for console output. Typer is the modern standard for Python CLIs.

### Command Structure: Subcommand-based
**Rationale:** Three distinct operation domains (generate, train, model) plus `ui`. Subcommands keep help text focused and allow domain-specific options.

### Command Naming: `sda generate`, `sda train`, `sda model`, `sda ui`
**Rationale:** Short verbs matching user mental model. `model` groups list/info/delete. `sda` entry point already exists in pyproject.toml.

### Default Verbosity: Normal (show progress, show results)
**Rationale:** CLI users expect to see what's happening. Add `--quiet`/`-q` to suppress progress. Add `--verbose`/`-v` for debug info.

### JSON Output: Yes, via `--json` flag
**Rationale:** Essential for scripting. When `--json` is set, print structured JSON to stdout. Progress/status always goes to stderr so JSON is clean.

### Stderr/Stdout Routing
**Rationale:** Standard Unix convention:
- **stdout:** Machine-readable data (file paths, JSON results)
- **stderr:** Human-readable progress, status messages, errors, Rich output
This enables piping: `sda generate --model x -n 5 | xargs ffprobe`

### Batch Specification: `--count N` plus `--seed-list` file
**Rationale:** `--count N` with optional `--seed S` (auto-increments: S, S+1, ..., S+N-1) covers 90% of use cases. For advanced: `--seed-list seeds.txt` reads one seed per line. Simple, composable.

### Parameter Sweep: Not in v1
**Rationale:** Parameter sweeps are complex (cartesian products, parallel execution) and can be accomplished with shell scripting: `for s in 42 43 44; do sda generate --seed $s ...; done`. Defer to a future phase if demand emerges.

### Model Resolution: Name, ID, or file path
**Rationale:** Accept any of: (1) `.sda` file path, (2) model UUID, (3) model name substring (error if ambiguous). Covers both library users and direct-file users.

### Path Override: CLI flags (primary) + `--config` for project-level override
**Rationale:** `--output-dir`, `--config` flags for explicit control. `--config` overrides the entire project config. No env vars in v1 (adds complexity without clear demand -- can add SDA_CONFIG_PATH later).

### Training Config: Named presets + individual overrides
**Rationale:** `--preset conservative|balanced|aggressive|auto` selects base config. Individual flags (`--epochs`, `--lr`, `--batch-size`) override specific values. Matches the existing `TrainingConfig` + `OverfittingPreset` system exactly.

### Exit Codes
| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (invalid args, runtime error) |
| 2 | Model not found / file not found |
| 3 | Training cancelled (checkpoint saved) |

### Help Text: Concise with examples
**Rationale:** Typer auto-generates help from docstrings and type hints. Add `[dim]Example:[/dim]` in command docstrings for common usage patterns.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| argparse with manual subparsers | Typer with type hints | Typer 0.9+ (2023) | Less boilerplate, auto help, Rich integration |
| tqdm for progress bars | Rich Progress with custom columns | Rich 12+ (2022) | More features, better formatting, built-in to Typer's dependency tree |
| Print-based status output | Rich Console with stderr | Rich 10+ (2021) | Clean stdout/stderr separation, styled output |

**Deprecated/outdated:**
- `optparse`: Removed in favor of argparse since Python 3.x. Not relevant.
- `tqdm` as a separate dependency: Rich's progress bars are superior and already in the dependency tree via Typer.

## Existing Code to Modify

### `app.py` Entry Point Refactoring
The current `app.py:main()` function does config loading, first-run setup, device selection, and Gradio UI launch. For Phase 9:
1. Extract the bootstrap logic (config + device + validation) into `cli/__init__.py:bootstrap()`
2. Keep `first_run_setup()` in `app.py` (interactive, only needed on first launch)
3. Update `main()` to instantiate the Typer app and dispatch
4. The `sda ui` command reuses the existing `launch_ui()` from `ui/__init__.py`

### `pyproject.toml` Entry Point
Currently: `sda = "small_dataset_audio.app:main"`
Update to: `sda = "small_dataset_audio.cli:main"` (where `cli/__init__.py` defines `main = app` or wraps the Typer app)

### Dependency Addition
Add to `[project]` dependencies: `"typer>=0.23,<1.0"`

## Open Questions

1. **Default behavior for bare `sda` (no subcommand)**
   - What we know: Currently launches GUI. Users expect this to continue.
   - What's unclear: Typer's `invoke_without_command=True` callback pattern may need testing to ensure backward compatibility.
   - Recommendation: Use Typer's callback with `invoke_without_command=True`. If no subcommand is given, run the GUI launch flow (config + validation + launch_ui). Test this thoroughly.

2. **Preset resolution by name vs ID**
   - What we know: Presets are model-scoped, stored in `data/presets/{model_id}/presets.json`. PresetManager requires model_id.
   - What's unclear: Whether to accept preset name or preset ID on CLI. Names may not be unique across folders.
   - Recommendation: Accept preset name (case-insensitive search). If ambiguous within the model's presets, show matches and error. Preset ID as fallback for scripting.

3. **Signal handling for training cancellation on Windows**
   - What we know: SIGINT works on Unix. Windows CTRL+C is different.
   - What's unclear: Whether `signal.signal(signal.SIGINT, handler)` works correctly on Windows with Rich progress bars.
   - Recommendation: Test on target platforms. The existing `threading.Event` pattern should work cross-platform. Fall back to KeyboardInterrupt exception handler if signal approach fails.

## Sources

### Primary (HIGH confidence)
- Existing codebase inspection: `app.py`, `inference/generation.py`, `training/runner.py`, `training/loop.py`, `library/catalog.py`, `presets/manager.py`, `models/persistence.py`, `config/settings.py`, `training/config.py`, `training/metrics.py`
- [Typer PyPI](https://pypi.org/project/typer/) - Version 0.23.1, dependencies confirmed (Click, Rich, Shellingham)
- [Typer subcommands tutorial](https://typer.tiangolo.com/tutorial/subcommands/add-typer/) - Sub-typer pattern verified
- [Rich Progress Display docs](https://rich.readthedocs.io/en/stable/progress.html) - Custom columns, task.fields, stderr support

### Secondary (MEDIUM confidence)
- [Typer progress bar tutorial](https://typer.tiangolo.com/tutorial/progressbar/) - Rich integration recommendation
- [Typer alternatives page](https://typer.tiangolo.com/alternatives/) - Click/argparse comparison
- [CLI stdout/stderr best practices](https://julienharbulot.com/python-cli-streams.html) - Standard Unix routing pattern

### Tertiary (LOW confidence)
- None -- all findings verified against official docs or codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Typer version confirmed on PyPI, Rich already in project, dependencies verified
- Architecture: HIGH - All wrapped APIs exist and have been read/verified in codebase
- Pitfalls: HIGH - Identified from direct codebase analysis (entry point, import patterns, threading model)

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (stable domain, no fast-moving dependencies)
