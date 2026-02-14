# Phase 8: Gradio UI - Research

**Researched:** 2026-02-13
**Domain:** Gradio web UI framework wrapping existing Python backend (Phases 1-7)
**Confidence:** HIGH

## Summary

Phase 8 creates a complete Gradio-based GUI that surfaces all backend functionality from Phases 1-7. The application already has a well-structured Python backend with clean public APIs for dataset import, training (threaded runner with callbacks), generation (pipeline with slider-to-latent mapping), model persistence, presets, history, and A/B comparison. Gradio 5/6 provides all the components needed: `gr.Tab` for the 4-tab layout, `gr.Slider` for musical controls, `gr.Audio` with built-in waveform display for playback, `gr.Plot` with `gr.Timer` for live training loss curves, `gr.File`/`gr.UploadButton` for dataset import, `gr.Gallery` for thumbnail grids, `gr.Dataframe` for sortable table views, and `gr.State` for session state management.

The primary technical challenge is the training tab: the `TrainingRunner` already runs in a background thread with a `MetricsCallback`, but Gradio needs a polling mechanism (`gr.Timer`) to read metrics from shared state and update the loss chart, stats panel, and preview audio players. The generator/yield pattern works for generation progress but NOT for training (which is truly backgrounded). The solution is: store metrics in a thread-safe shared structure, poll with `gr.Timer` every 1-2 seconds to update the training dashboard.

The secondary challenge is `gr.State` deepcopy limitation -- the backend objects (`TrainingRunner`, `GenerationPipeline`, `ModelLibrary`, etc.) cannot be deep-copied. Use the global-dict-keyed-by-session-hash pattern documented in Gradio's state guide, or simply use module-level singletons since this is a single-user desktop application.

**Primary recommendation:** Use Gradio 5+ (current PyPI: 6.5.1) with `gr.Blocks` layout, `gr.Timer` for training dashboard polling, module-level application state (single-user app), and matplotlib for loss curves via `gr.Plot`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- 4 tabs: Data, Train, Generate, Library
- Guided navigation with overrides -- tabs highlight the suggested next step, but user can click any tab; unready tabs show what's needed (e.g., "Import a dataset to start training")
- Generate tab includes history and A/B comparison -- Claude's discretion on whether inline or collapsible sections, based on Gradio's component model
- Sliders grouped by category (timbral, temporal, spatial) in 2-3 columns -- more compact layout
- Explicit Generate button with spinner/progress; inline audio player appears when done (no auto-play)
- Live loss chart (train + val curves) with stats panel (epoch, learning rate, ETA)
- Inline audio players for each preview epoch -- user can play any preview to hear model progress over time
- Cancel button during training + explicit "Resume Training" button when checkpoint exists for selected dataset
- Drag-and-drop zone + browse button for file/folder import
- After import: stats panel at top (count, total duration, rate info) + grid of waveform thumbnails below, clickable to play
- Dual view: card grid and table/list with a toggle between them (Library tab)
- Card grid as default view -- each card shows name, dataset info, training date, sample count
- Table view as alternative -- sortable columns, compact and scannable
- Export controls on Generate tab, next to audio player after generation
- Model library should support toggling between card grid and table view
- Training previews should show progression -- inline audio players per epoch, not just the latest one
- Tabs should feel guided (highlight next step) but never locked -- always accessible with helpful empty states

### Claude's Discretion
- Generate tab layout for history/A/B (inline vs collapsible sections)
- Preset recall component choice (dropdown vs button row)
- Seed input prominence (visible field vs advanced toggle)
- Training config UI design (optimized for audio quality outcomes)
- Exact spacing, typography, and visual polish within Gradio's constraints

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gradio | >=5.0,<7.0 | Web UI framework | De facto standard for Python ML GUIs; provides audio components, sliders, tabs, plots, file upload out of the box |
| matplotlib | >=3.9 (already installed) | Training loss curves | Already a project dependency; `gr.Plot` renders matplotlib figures natively |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=1.26 (already installed) | Audio data handling for gr.Audio | Already a project dependency; gr.Audio accepts `(sample_rate, ndarray)` tuples |
| Pillow | (Gradio dependency) | Image handling for thumbnails in Gallery | Pulled in transitively by Gradio; no explicit install needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib for loss curves | plotly | Plotly is interactive (zoom/pan) but adds dependency; matplotlib already installed and produces static figures sufficient for loss monitoring |
| gr.Dataframe for table view | gr.HTML with custom table | Full control but loses built-in sorting; Dataframe is simpler and supports column sorting |

**Installation:**
```bash
uv add "gradio>=5.0,<7.0"
```

Note: Gradio 6.5.1 (latest on PyPI as of 2026-01-29) requires Python >=3.10. This project requires Python >=3.11, so compatibility is assured.

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── ui/
│   ├── __init__.py           # launch_ui() entry point
│   ├── app.py                # gr.Blocks assembly, tab wiring
│   ├── state.py              # AppState singleton, session management
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── data_tab.py       # Data tab: import, validation, thumbnails
│   │   ├── train_tab.py      # Train tab: config, loss chart, previews
│   │   ├── generate_tab.py   # Generate tab: sliders, audio, history, A/B
│   │   └── library_tab.py    # Library tab: card grid, table, model management
│   └── components/
│       ├── __init__.py
│       ├── loss_chart.py     # matplotlib figure builder for training curves
│       ├── model_card.py     # HTML template for model card grid
│       └── guided_nav.py     # Tab highlighting / empty state messages
```

### Pattern 1: Module-Level Application State (Single-User Desktop App)
**What:** Instead of `gr.State` (which requires deepcopy-able objects), use a module-level `AppState` dataclass/class that holds references to the current model, pipeline, training runner, preset manager, history store, etc.
**When to use:** Single-user local application where session isolation is not needed.
**Why:** `TrainingRunner`, `GenerationPipeline`, `ModelLibrary` all contain non-deepcopyable objects (threading locks, PyTorch models). `gr.State` would fail or be extremely expensive. Module-level state is the documented Gradio pattern for this case.
**Example:**
```python
# Source: Gradio state-in-blocks guide + project-specific pattern
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class AppState:
    """Global application state for single-user desktop app."""
    # Paths from config
    datasets_dir: Path = Path("data/datasets")
    models_dir: Path = Path("data/models")
    generated_dir: Path = Path("data/generated")

    # Current loaded model
    loaded_model: Optional["LoadedModel"] = None
    pipeline: Optional["GenerationPipeline"] = None

    # Training state
    training_runner: Optional["TrainingRunner"] = None
    metrics_history: Optional["MetricsHistory"] = None
    training_active: bool = False

    # Current dataset
    current_dataset: Optional["Dataset"] = None
    current_summary: Optional["DatasetSummary"] = None

    # Managers (initialized on model load)
    preset_manager: Optional["PresetManager"] = None
    history_store: Optional["GenerationHistory"] = None
    model_library: Optional["ModelLibrary"] = None

    # A/B comparison state
    ab_comparison: Optional["ABComparison"] = None

# Module-level singleton
app_state = AppState()
```

### Pattern 2: Timer-Based Training Dashboard Polling
**What:** Use `gr.Timer(value=2)` to poll shared training state every 2 seconds and update the loss chart, stats panel, and preview list.
**When to use:** Training tab when `TrainingRunner` is active in its background thread.
**Why:** The `TrainingRunner` runs in a daemon thread and emits metrics via callback. Gradio's request-response model cannot receive push events from background threads. Polling via Timer is the documented pattern.
**Example:**
```python
# Source: Gradio Timer docs + project TrainingRunner.callback pattern
import gradio as gr
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (already used in project)
import matplotlib.pyplot as plt

# Shared metrics buffer (written by training callback, read by Timer)
_metrics_buffer = {"epoch_metrics": [], "previews": [], "complete": False}

def training_callback(event):
    """MetricsCallback that stores events for the Timer to read."""
    if isinstance(event, EpochMetrics):
        _metrics_buffer["epoch_metrics"].append(event)
    elif isinstance(event, PreviewEvent):
        _metrics_buffer["previews"].append(event)
    elif isinstance(event, TrainingCompleteEvent):
        _metrics_buffer["complete"] = True

def build_loss_chart():
    """Build matplotlib figure from accumulated metrics."""
    metrics = _metrics_buffer["epoch_metrics"]
    if not metrics:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = [m.epoch for m in metrics]
    ax.plot(epochs, [m.train_loss for m in metrics], label="Train")
    ax.plot(epochs, [m.val_loss for m in metrics], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training Progress")
    plt.tight_layout()
    return fig

def poll_training():
    """Called by gr.Timer -- returns updated components."""
    chart = build_loss_chart()
    metrics = _metrics_buffer["epoch_metrics"]
    if metrics:
        latest = metrics[-1]
        stats = f"Epoch {latest.epoch}/{latest.total_epochs} | LR: {latest.learning_rate:.2e} | ETA: {latest.eta_seconds:.0f}s"
    else:
        stats = "Waiting for first epoch..."
    return chart, stats

# In Blocks:
timer = gr.Timer(value=2, active=False)  # activated when training starts
timer.tick(fn=poll_training, outputs=[loss_plot, stats_text])
```

### Pattern 3: Generator-Based Generation with Progress
**What:** Use a generator function that yields progress updates during audio generation.
**When to use:** Generate tab -- generation takes a few seconds, user needs feedback.
**Example:**
```python
def generate_audio(slider_values, seed, duration, ...):
    """Generator that yields progress, then final audio."""
    yield gr.update(visible=True), "Generating...", None  # show spinner

    # Build latent vector from sliders
    slider_state = SliderState(positions=slider_values, n_components=len(slider_values))
    latent = sliders_to_latent(slider_state, app_state.loaded_model.analysis)

    # Configure and generate
    config = GenerationConfig(
        duration_s=duration,
        seed=seed,
        latent_vector=latent,
        ...
    )
    result = app_state.pipeline.generate(config)

    # Return audio as numpy tuple for gr.Audio
    audio_tuple = (result.sample_rate, result.audio)
    yield gr.update(visible=False), "Done!", audio_tuple
```

### Pattern 4: Card Grid via gr.HTML + Table via gr.Dataframe Toggle
**What:** Model library dual-view using `gr.HTML` for card grid layout and `gr.Dataframe` for sortable table, toggled with radio buttons.
**When to use:** Library tab -- user decision requires both card grid and table views.
**Why:** `gr.Gallery` is image-only. Cards need text metadata (name, date, sample count). Custom HTML cards give full control over layout. `gr.Dataframe` gives sortable columns for table view.
**Example:**
```python
def render_model_cards(models):
    """Generate HTML card grid from model entries."""
    cards_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px;">'
    for m in models:
        cards_html += f'''
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px;">
            <h3>{m.name}</h3>
            <p>Dataset: {m.dataset_name} ({m.dataset_file_count} files)</p>
            <p>Trained: {m.training_date[:10]} | {m.training_epochs} epochs</p>
            <p>Components: {m.n_active_components}</p>
        </div>
        '''
    cards_html += '</div>'
    return cards_html
```

### Pattern 5: Guided Navigation with Empty States
**What:** Each tab checks whether prerequisites are met and displays helpful messages when they are not.
**When to use:** All tabs -- locked decision says guided but never locked.
**Example:**
```python
def get_train_tab_content():
    if app_state.current_dataset is None:
        return gr.update(visible=True), gr.update(visible=False)  # show empty state, hide controls
    return gr.update(visible=False), gr.update(visible=True)  # hide empty state, show controls

# Empty state component:
empty_msg = gr.Markdown(
    "## Import a dataset to start training\n\n"
    "Go to the **Data** tab to import audio files first.",
    visible=True,
)
```

### Anti-Patterns to Avoid
- **gr.State for PyTorch models:** Models, training runners, and pipeline objects are NOT deepcopy-able. Use module-level state instead.
- **Pushing updates from background threads:** Gradio has no push mechanism. Use `gr.Timer` polling, not attempts to call `gr.update()` from training threads.
- **matplotlib GUI backend:** Always use `matplotlib.use("Agg")` -- the project already does this. Starting matplotlib GUI outside the main thread will fail.
- **Blocking the Gradio event loop with training:** Training MUST run in the existing `TrainingRunner` background thread, not in a Gradio event handler. Gradio handlers should only start/stop training and poll for status.
- **Building multipage apps for tabs:** Multipage apps in Gradio do NOT support cross-page component interactions. Use `gr.Tabs` with `gr.Tab` instead -- all components are in the same Blocks scope and can interact.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio playback with waveform | Custom HTML5 audio + canvas | `gr.Audio` with `WaveformOptions` | Built-in waveform via Wavesurfer.js, transport controls, download button |
| File upload with drag-and-drop | Custom dropzone HTML | `gr.File(file_count="multiple")` + `gr.UploadButton(file_count="directory")` | Built-in drag-and-drop zone; combine File for drag-drop + UploadButton for directory browsing |
| Sortable data table | Custom HTML table + JavaScript | `gr.Dataframe` | Built-in column sorting, styling API, scrollable |
| Periodic UI updates | Custom JavaScript polling | `gr.Timer` | Built-in tick event with configurable interval, can be activated/deactivated |
| Progress indication | Custom progress div | `gr.Progress` for generation; `gr.Timer` + plot for training | Built-in progress bar integration with tqdm support |
| Image thumbnail grid | Custom HTML image grid | `gr.Gallery` | Built-in grid layout, preview mode, selection events, scrollable |

**Key insight:** Gradio provides audio-specific components (gr.Audio with waveform), timer-based polling (gr.Timer), and layout primitives (Tabs, Row, Column, Accordion) that cover every UI need for this application. The only custom HTML needed is model cards in the library grid view, where text metadata display requires more flexibility than gr.Gallery (which is image-only).

## Common Pitfalls

### Pitfall 1: gr.State Deepcopy Failure
**What goes wrong:** Storing PyTorch models, threading objects, or file handles in `gr.State` causes deepcopy errors or extreme memory usage.
**Why it happens:** Gradio deep-copies State values for session isolation. PyTorch tensors, CUDA contexts, and threading locks cannot be deep-copied.
**How to avoid:** Use module-level `AppState` singleton for all non-trivial objects. Only use `gr.State` for simple JSON-serializable data (current tab index, UI toggle states).
**Warning signs:** `TypeError: cannot pickle` errors, `RuntimeError: CUDA error` on state access, or sudden memory doubling.

### Pitfall 2: Blocking Gradio Event Handlers
**What goes wrong:** Running training directly in a Gradio event handler freezes the entire UI until training completes.
**Why it happens:** Gradio event handlers run synchronously in the server's event loop. A multi-minute training run blocks ALL UI interaction.
**How to avoid:** Event handlers should only call `TrainingRunner.start()` (which spawns a daemon thread) and return immediately. Use `gr.Timer` to poll training progress. The handler for the "Train" button takes < 100ms.
**Warning signs:** UI becomes unresponsive when training starts, browser shows "page not responding."

### Pitfall 3: Matplotlib Thread Safety
**What goes wrong:** Creating matplotlib figures from the training callback thread causes crashes or garbled output.
**Why it happens:** matplotlib's Agg backend is not fully thread-safe. Creating figures concurrently can corrupt global state.
**How to avoid:** Build loss chart figures ONLY in the Timer tick handler (which runs in Gradio's event loop, not the training thread). The training callback should only store data in a thread-safe structure (list.append is thread-safe in CPython due to GIL). The Timer handler reads data and creates the figure.
**Warning signs:** Corrupted plot images, random crashes during training, matplotlib warnings about thread safety.

### Pitfall 4: Timer Active During Non-Training
**What goes wrong:** `gr.Timer` keeps polling when training is not active, wasting resources and potentially causing errors.
**Why it happens:** Timer runs continuously once started. If the user navigates away from training or training completes, the timer keeps firing.
**How to avoid:** Set `timer = gr.Timer(value=2, active=False)`. Activate it when training starts (return `gr.Timer(active=True)` from the start handler). Deactivate when training completes or is cancelled (detect in the tick handler and return `gr.Timer(active=False)`).
**Warning signs:** Unnecessary server load, error logs from polling non-existent training state.

### Pitfall 5: Large File Handling in gr.File
**What goes wrong:** Uploading many large audio files via `gr.File` causes timeouts or memory issues because Gradio copies files to a temp directory.
**Why it happens:** Gradio copies uploaded files to its temp directory, which means large datasets get duplicated on disk.
**How to avoid:** After receiving uploaded files from `gr.File`, copy them to the project's datasets directory and create the `Dataset` object from the final location. Clear the upload component after import. Consider showing import progress for large batches.
**Warning signs:** Disk space doubling during import, slow uploads, timeout errors.

### Pitfall 6: Audio Component Format Mismatch
**What goes wrong:** Passing audio data in the wrong format to `gr.Audio` produces silence or errors.
**Why it happens:** `gr.Audio` with `type="numpy"` expects `(sample_rate, ndarray)` tuples. The project's `GenerationResult.audio` is a raw ndarray without sample rate packaging.
**How to avoid:** Always wrap audio for `gr.Audio` output as `(result.sample_rate, result.audio)`. Ensure audio is float32 (Gradio auto-normalizes to prevent clipping). For file-based playback (history entries), pass the file path string directly.
**Warning signs:** Silent audio output, "Invalid audio data" errors, clicking/distortion.

### Pitfall 7: Dynamic Slider Count
**What goes wrong:** Different models have different numbers of active PCA components, requiring a variable number of sliders.
**Why it happens:** The `AnalysisResult.n_active_components` varies by model. A model trained on 10 files might have 3 components; one trained on 200 might have 8.
**How to avoid:** Create a maximum number of slider components (e.g., 12) and show/hide them based on the loaded model's component count. Use `gr.update(visible=True/False)` to toggle slider visibility. Store slider metadata from `get_slider_info()` to set labels and ranges dynamically.
**Warning signs:** Sliders for non-existent components, index-out-of-range errors, blank slider labels.

## Code Examples

Verified patterns from official sources and project backend:

### Starting the Gradio App
```python
# Source: Gradio docs + project structure
import gradio as gr

def create_app() -> gr.Blocks:
    """Build the complete Gradio Blocks application."""
    with gr.Blocks(
        title="Small Dataset Audio",
        fill_width=True,
    ) as app:
        gr.Markdown("# Small Dataset Audio")

        with gr.Tabs() as tabs:
            with gr.Tab("Data", id="data"):
                build_data_tab()
            with gr.Tab("Train", id="train"):
                build_train_tab()
            with gr.Tab("Generate", id="generate"):
                build_generate_tab()
            with gr.Tab("Library", id="library"):
                build_library_tab()

    return app

def launch_ui():
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
```

### Dataset Import Handler
```python
# Source: Project's Dataset.from_files() + Gradio File upload
def handle_file_upload(files):
    """Process uploaded audio files into a Dataset."""
    if not files:
        return "No files uploaded.", None, None

    file_paths = [Path(f.name) for f in files]
    dataset = Dataset.from_files(
        files=file_paths,
        name="Uploaded Dataset",
        base_dir=app_state.datasets_dir,
    )
    app_state.current_dataset = dataset
    summary = compute_summary(dataset, generate_thumbnails=True)
    app_state.current_summary = summary

    # Build stats text
    stats = (
        f"**{summary.valid_file_count}** files | "
        f"**{summary.total_duration_seconds:.1f}s** total | "
        f"**{summary.dominant_sample_rate} Hz**"
    )

    # Build thumbnail gallery
    thumbnails = list(summary.thumbnail_paths.values())

    return stats, thumbnails, gr.update(visible=True)
```

### Musical Parameter Sliders
```python
# Source: Project's get_slider_info() + Gradio Slider docs
def build_sliders(max_components=12):
    """Create slider components for musical parameters."""
    sliders = []
    for i in range(max_components):
        s = gr.Slider(
            minimum=-10,
            maximum=10,
            value=0,
            step=1,
            precision=0,
            label=f"Axis {i + 1}",
            visible=False,
            interactive=True,
        )
        sliders.append(s)
    return sliders

def update_sliders_for_model(loaded_model):
    """Configure sliders based on loaded model's analysis."""
    if loaded_model.analysis is None:
        return [gr.update(visible=False)] * MAX_SLIDERS

    infos = get_slider_info(loaded_model.analysis)
    updates = []
    for i in range(MAX_SLIDERS):
        if i < len(infos):
            info = infos[i]
            updates.append(gr.update(
                visible=True,
                label=info["suggested_label"] or info["label"],
                minimum=info["min_step"],
                maximum=info["max_step"],
                value=0,
            ))
        else:
            updates.append(gr.update(visible=False))
    return updates
```

### Training Start/Cancel/Resume
```python
# Source: Project's TrainingRunner API + Gradio Timer pattern
def start_training(dataset_path, preset_name, max_epochs, learning_rate):
    """Start training and activate the polling timer."""
    from small_dataset_audio.training.config import get_adaptive_config

    dataset = app_state.current_dataset
    config = get_adaptive_config(dataset.file_count)
    config.max_epochs = max_epochs
    config.learning_rate = learning_rate

    # Clear metrics buffer
    _metrics_buffer.clear()
    _metrics_buffer.update({"epoch_metrics": [], "previews": [], "complete": False})

    runner = TrainingRunner()
    app_state.training_runner = runner
    runner.start(
        config=config,
        file_paths=dataset.valid_files,
        output_dir=app_state.datasets_dir / dataset.name,
        device=device,
        callback=training_callback,
    )

    # Return: activate timer, disable start button, enable cancel button
    return gr.Timer(active=True), gr.update(interactive=False), gr.update(interactive=True)

def cancel_training():
    """Cancel training and deactivate timer."""
    if app_state.training_runner:
        app_state.training_runner.cancel()
    return gr.Timer(active=False), gr.update(interactive=True), gr.update(interactive=False)
```

### History and A/B Comparison
```python
# Source: Project's GenerationHistory + ABComparison APIs
def load_history_entries(model_id=None, limit=20):
    """Load history entries for display."""
    entries = app_state.history_store.list_entries(model_id=model_id, limit=limit)
    thumbnails = []
    for entry in entries:
        thumb_path = app_state.history_store.history_dir / entry.thumbnail_file
        if thumb_path.exists():
            thumbnails.append((str(thumb_path), f"Seed: {entry.seed}"))
    return thumbnails

def start_ab_comparison(entry_a_id, entry_b_id):
    """Initialize A/B comparison between two history entries."""
    app_state.ab_comparison = ABComparison.from_two_entries(entry_a_id, entry_b_id)
    paths = app_state.ab_comparison.get_audio_paths(app_state.history_store)
    return str(paths[0]), str(paths[1]), "A"  # audio_a, audio_b, active_label
```

## Discretion Recommendations

### Generate Tab Layout: Collapsible Sections (RECOMMENDED)
**Recommendation:** Use `gr.Accordion` for History and A/B Comparison sections, defaulting to collapsed.
**Rationale:** The Generate tab already has sliders (2-3 columns), generation controls, and audio output. Adding history gallery and A/B controls inline would make the page very long. `gr.Accordion` with `open=False` keeps the primary generate-listen-export workflow compact while making history and A/B easily accessible with one click. History expands to show a `gr.Gallery` of waveform thumbnails; A/B expands to show two audio players with a toggle button.

### Preset Recall: Dropdown (RECOMMENDED)
**Recommendation:** Use `gr.Dropdown` for preset selection.
**Rationale:** Presets can grow to dozens per model. A dropdown with `filterable=True` scales better than a button row. Buttons work for 3-5 items but become unwieldy at 10+. The dropdown also supports dynamic choice updates when the model changes. Include a "Custom" option that represents the current unsaved slider state.

### Seed Input: Visible Field with Sensible Default (RECOMMENDED)
**Recommendation:** Show seed as a visible `gr.Number` field with a "Random" button next to it.
**Rationale:** Seed is central to the reproducibility workflow (core project value: "controllable exploration"). Hiding it in an advanced toggle adds friction to a common action. A compact `gr.Number` field with a `gr.Button("Randomize")` beside it is small and always accessible. Default to empty/None (random).

### Training Config: Preset Selector + Smart Defaults + Advanced Accordion (RECOMMENDED)
**Recommendation:** Show a preset dropdown (Conservative/Balanced/Aggressive) that auto-selects based on dataset size. Display key parameters (epochs, learning rate) as visible fields. Put advanced regularization options in a collapsed `gr.Accordion("Advanced Training Settings", open=False)`.
**Rationale:** The backend's `get_adaptive_config()` already selects optimal defaults. Most users should just click "Train." Power users can expand advanced settings to tweak dropout, weight decay, etc. This matches the project's layered control philosophy (automatic defaults -> presets -> advanced toggles).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `gr.Interface` simple layouts | `gr.Blocks` composable layouts | Gradio 3.0 (2022) | Blocks required for multi-tab, complex state apps |
| `every=N` parameter on components | `gr.Timer` component | Gradio 5.0 (2024) | Dedicated timer component, cleaner API for periodic updates |
| `gr.update()` function | Return component constructors | Gradio 5.0 (2024) | `gr.Slider(visible=True)` instead of `gr.update(visible=True)` -- both still work |
| `gr.make_waveform()` helper | Native waveform in `gr.Audio` | Gradio 5.0 (2024) | Audio component has built-in Wavesurfer.js waveform display |
| No SSR | Server-side rendering | Gradio 5.0 (2024) | Instant page loads, better UX |
| `gr.Tab` only | `gr.Tab` + `gr.Sidebar` + `gr.Accordion` | Gradio 5.0+ | More layout options for complex apps |

**Deprecated/outdated:**
- `gr.make_waveform()`: Removed in Gradio 5. Audio component has native waveform support.
- `gr.Interface` for complex apps: Still works but `gr.Blocks` is required for cross-component interactions and custom layouts.
- `every=float` on components: Replaced by `gr.Timer` (more explicit, can be activated/deactivated).

## Open Questions

1. **gr.File directory upload + drag-and-drop interaction**
   - What we know: `gr.UploadButton(file_count="directory")` supports folder upload. `gr.File(file_count="multiple")` supports drag-and-drop. There are known drag-and-drop bugs with specific file types.
   - What's unclear: Whether combining both (File for drag-drop + UploadButton for directory browse) works cleanly in the same tab, or if they conflict.
   - Recommendation: Implement both side-by-side. If drag-and-drop has issues with audio file extensions, use broader `file_types=["audio"]` type filter rather than specific extensions. Test during implementation.

2. **Model card grid click selection**
   - What we know: `gr.HTML` renders custom card HTML. `gr.Dataframe` supports row selection.
   - What's unclear: How to make HTML cards clickable to select/load a model. `gr.HTML` does not emit click events with row data.
   - Recommendation: Use `gr.Gallery` for card thumbnails (one image per model, auto-generated or placeholder) with captions for metadata, OR use `gr.Dataframe` for both views with custom styling. If HTML cards are used, pair them with explicit "Load" buttons per card. The table view (`gr.Dataframe`) handles selection natively.

3. **Training preview audio player count**
   - What we know: Training generates preview audio every N epochs. User wants inline players for each preview epoch.
   - What's unclear: Maximum number of preview audio components to pre-create (Gradio requires components to exist at build time).
   - Recommendation: Pre-create 10-20 `gr.Audio` components (hidden by default). As previews arrive, show them with `gr.update(visible=True, value=audio_path)`. 20 should cover most training runs (200 epochs / 5 preview interval = 40 previews, but early ones can scroll). Alternatively, use a single `gr.Gallery`-like list of audio file links.

## Sources

### Primary (HIGH confidence)
- [Gradio PyPI](https://pypi.org/project/gradio/) - Version 6.5.1, Python >=3.10 support confirmed
- [Gradio Layout Guide](https://www.gradio.app/guides/controlling-layout) - Tabs, Row, Column, Accordion, Sidebar documentation
- [Gradio Timer Docs](https://www.gradio.app/docs/gradio/timer) - Timer API, tick event, active parameter
- [Gradio Audio Docs](https://www.gradio.app/docs/gradio/audio) - Full component API, WaveformOptions, events
- [Gradio Slider Docs](https://www.gradio.app/docs/gradio/slider) - precision=0 for integer, step, min/max, events
- [Gradio Gallery Docs](https://www.gradio.app/docs/gradio/gallery) - Grid layout, columns, preview, selection
- [Gradio Dropdown Docs](https://www.gradio.app/docs/gradio/dropdown) - filterable, multiselect, dynamic choices
- [Gradio State Guide](https://www.gradio.app/guides/state-in-blocks) - Session state, deepcopy limitation, global dict pattern
- [Gradio Streaming Outputs](https://www.gradio.app/guides/streaming-outputs) - Generator/yield pattern for iterative updates
- [Gradio Progress Bars](https://www.gradio.app/guides/progress-bars) - gr.Progress, tqdm integration
- [Gradio Running Background Tasks](https://www.gradio.app/guides/running-background-tasks) - APScheduler pattern, background operations
- [Gradio Resource Cleanup](https://www.gradio.app/guides/resource-cleanup) - Blocks.unload(), session cleanup
- [Gradio Multipage Apps](https://www.gradio.app/guides/multipage-apps) - Confirmed tabs needed (multipage lacks cross-page interaction)

### Secondary (MEDIUM confidence)
- [Gradio Plot Docs](https://www.gradio.app/docs/gradio/plot) - matplotlib, plotly, Timer integration for live updates
- [Gradio Dataframe Guide](https://www.gradio.app/guides/styling-the-gradio-dataframe) - Sortable columns, styling, pandas Styler
- [Gradio UploadButton Docs](https://www.gradio.app/docs/gradio/uploadbutton) - directory upload, file_count parameter
- [Gradio Blocks Guide](https://www.gradio.app/guides/blocks-and-event-listeners) - Event listeners, component updates, cancel events

### Tertiary (LOW confidence)
- Known drag-and-drop bugs with specific file types (GitHub issues #6888, #7094, #10325) - may be fixed in Gradio 6.x
- `gr.HTML` click event limitations for card grid selection - needs implementation-time validation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Gradio is the locked framework choice; version and API verified via PyPI and official docs
- Architecture: HIGH - Patterns verified against Gradio documentation; backend APIs thoroughly reviewed from project source code
- Pitfalls: HIGH - State deepcopy, thread safety, Timer polling all documented in official guides; matplotlib threading from project decisions
- Discretion areas: MEDIUM - Recommendations based on Gradio component capabilities but exact UX needs implementation validation

**Research date:** 2026-02-13
**Valid until:** 2026-03-15 (Gradio releases frequently but core patterns are stable)
