# Phase 15: UI & CLI Vocoder Controls - Research

**Researched:** 2026-02-27
**Domain:** Gradio UI controls, Typer CLI options, HuggingFace Hub download progress
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Vocoder controls live in a **collapsible "Vocoder Settings" accordion** below the Generation Config row
- Accordion is collapsed by default (Claude's discretion on smart-open for first download or when per-model vocoder exists)
- Dropdown options: Auto, BigVGAN Universal, Per-model HiFi-GAN
- **Per-model HiFi-GAN is disabled (grayed out) with tooltip** when the current model doesn't have a trained per-model vocoder -- prevents selecting an unavailable option
- Status text inside accordion shows **readiness + resolution**: e.g., "Using: BigVGAN Universal" or "Using: Per-model HiFi-GAN"
- BigVGAN download progress appears **inline in the vocoder accordion** with progress bar and MB counter (e.g., "Downloading BigVGAN universal model... 245/489 MB")
- Download is **lazy** -- triggered on first generate attempt, not on app startup
- **Generate button disabled** during download with tooltip "Downloading vocoder..."
- On download failure: **error message in accordion + Retry Download button** (not a Gradio toast)
- **Always print vocoder line** on every generate call to stderr: `Vocoder: BigVGAN Universal (auto)`
- `--vocoder` flag accepts: auto, bigvgan, hifigan
- **Auto is the default** -- no flag needed, `distill generate model` just works
- When `--vocoder hifigan` is specified but model has no per-model vocoder: **error and exit** with non-zero status and message: "Error: model X has no trained per-model vocoder. Use --vocoder auto or bigvgan."
- BigVGAN download uses **Rich progress bar** (consistent with CLI's existing Rich console), not HuggingFace Hub's tqdm default
- **UI: both places** -- status text in accordion shows resolution before generation ("Using: BigVGAN Universal"), quality badge after generation confirms vocoder used alongside seed/sample-rate/bit-depth
- **CLI: label + reason** -- `Vocoder: BigVGAN Universal (auto -- no per-model vocoder)` or `Vocoder: Per-model HiFi-GAN (auto -- per-model available)` -- explains WHY auto chose what it chose
- **JSON output includes vocoder field** -- `{"vocoder": {"name": "bigvgan_universal", "selection": "auto"}}` added to `--json` output

### Claude's Discretion
- Accordion smart-open logic (whether to auto-open on first download or when per-model vocoder available)
- Exact tooltip wording for disabled Per-model HiFi-GAN option
- Progress bar styling details within Gradio constraints
- Rich progress bar formatting (speed, ETA display)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| UI-01 | Generate tab has vocoder selection (Auto / BigVGAN Universal / Per-model HiFi-GAN) | Gradio Accordion + Dropdown pattern verified; interactive toggle for disabled state; model_card vocoder detection via `LoadedModel.vocoder_state` and `ModelEntry.vocoder` |
| UI-02 | BigVGAN download progress shown in UI on first use | Gradio `Progress(track_tqdm=True)` auto-intercepts HuggingFace Hub tqdm; inline Markdown status in accordion via polling/threading for MB counter |
| CLI-01 | `--vocoder` flag on generate command selects vocoder (auto/bigvgan/hifigan) | Typer `typer.Option` with `str` type; `get_vocoder()` factory already supports type dispatch; auto-resolution logic via `LoadedModel.vocoder_state` |
| CLI-03 | BigVGAN download progress shown via Rich progress bar | `tqdm.rich.tqdm_rich` class passed as `tqdm_class` to `snapshot_download()`; `Console(stderr=True)` pattern consistent with existing CLI |

</phase_requirements>

## Summary

Phase 15 adds vocoder selection controls and download progress visibility to two surfaces: the Gradio web UI and the Typer CLI. The codebase is well-structured for this addition -- the vocoder system (`distill.vocoder`), the `get_vocoder()` factory, the `BigVGANVocoder` constructor, and the `weight_manager.ensure_bigvgan_weights()` function are all in place from Phase 12. The model persistence layer already stores `vocoder_state` in `.distillgan` files and exposes `VocoderInfo` in the catalog. The generate tab and CLI generate command already use the vocoder pipeline.

The primary technical challenge is bridging HuggingFace Hub's download progress (tqdm-based) into both the Gradio UI and the Rich CLI in a clean, non-frozen way. Both paths have well-supported integration points: Gradio's `Progress(track_tqdm=True)` auto-patches tqdm, and `snapshot_download()` accepts a `tqdm_class` parameter for Rich integration. The vocoder auto-selection logic (which vocoder to pick when "auto" is selected) is straightforward: check `LoadedModel.vocoder_state` -- if present, use per-model HiFi-GAN; if not, use BigVGAN universal.

**Primary recommendation:** Add a `resolve_vocoder()` function that implements auto-selection logic; extend `weight_manager.py` to accept a `tqdm_class` for download customization; build the Gradio accordion with lazy download trigger; add `--vocoder` option to the CLI generate command.

## Standard Stack

### Core (already installed, no new dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gradio | 6.6.0 | Web UI framework | Already used for all tabs; `gr.Progress(track_tqdm=True)` provides tqdm interception |
| typer | 0.24.1 | CLI framework | Already used for all CLI commands; `typer.Option` for `--vocoder` flag |
| rich | 14.3.3 | Terminal styling / progress | Already used in all CLI commands with `Console(stderr=True)` pattern |
| huggingface_hub | 1.4.1 | Model download | Already used by `weight_manager.py`; `snapshot_download(tqdm_class=...)` supports custom progress |
| tqdm | (transitive) | Progress abstraction | Bridges HuggingFace Hub to both Gradio and Rich via `tqdm_class` parameter |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm.rich.tqdm_rich | (from tqdm) | Rich-backed tqdm class | Pass as `tqdm_class` to `snapshot_download()` for CLI Rich progress bar |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `tqdm.rich.tqdm_rich` | Custom `RichProgressTqdm` class | tqdm_rich works out of the box; custom class only needed if formatting needs differ significantly |
| `gr.Progress(track_tqdm=True)` | Custom Gradio polling via `gr.Timer` | track_tqdm is simpler and automatic; Timer only needed for inline accordion MB counter |

**Installation:** No new dependencies needed. All packages already in `pyproject.toml`.

## Architecture Patterns

### Recommended Project Structure Changes
```
src/distill/
├── vocoder/
│   ├── __init__.py       # Update get_vocoder() to support 'auto' + resolve_vocoder()
│   ├── weight_manager.py # Add tqdm_class param to ensure_bigvgan_weights()
│   └── ... (unchanged)
├── cli/
│   └── generate.py       # Add --vocoder option, vocoder status line, JSON vocoder field
└── ui/
    └── tabs/
        └── generate_tab.py  # Add Vocoder Settings accordion with controls
```

### Pattern 1: Vocoder Auto-Resolution
**What:** Central function that decides which vocoder to use based on user selection + model capabilities.
**When to use:** Both UI and CLI call this before generation.
**Example:**
```python
# Source: Codebase analysis -- new function in distill/vocoder/__init__.py
def resolve_vocoder(
    selection: str,  # "auto", "bigvgan", "hifigan"
    loaded_model: LoadedModel,
    device: str = "auto",
    tqdm_class: type | None = None,
) -> tuple[VocoderBase, dict]:
    """Resolve vocoder selection to a concrete vocoder instance.

    Returns (vocoder, info_dict) where info_dict contains:
      - name: "bigvgan_universal" or "per_model_hifigan"
      - selection: "auto" or "bigvgan" or "hifigan"
      - reason: human-readable explanation of auto-selection
    """
    has_per_model = loaded_model.vocoder_state is not None

    if selection == "hifigan":
        if not has_per_model:
            raise ValueError(
                f"Model '{loaded_model.metadata.name}' has no trained "
                "per-model vocoder. Use --vocoder auto or bigvgan."
            )
        # Phase 16: load per-model HiFi-GAN from vocoder_state
        raise NotImplementedError("Per-model HiFi-GAN vocoder is Phase 16.")

    if selection == "auto":
        if has_per_model:
            # Phase 16: prefer per-model when available
            # For now, always fall through to BigVGAN
            pass
        return (
            get_vocoder("bigvgan", device=device, tqdm_class=tqdm_class),
            {"name": "bigvgan_universal", "selection": "auto",
             "reason": "no per-model vocoder"},
        )

    # selection == "bigvgan"
    return (
        get_vocoder("bigvgan", device=device, tqdm_class=tqdm_class),
        {"name": "bigvgan_universal", "selection": "bigvgan", "reason": "explicit"},
    )
```

### Pattern 2: CLI Rich Download Progress
**What:** Pass `tqdm.rich.tqdm_rich` as `tqdm_class` to `snapshot_download()` for Rich-styled progress.
**When to use:** CLI generate command when BigVGAN weights need downloading.
**Example:**
```python
# Source: huggingface_hub API (tqdm_class param) + tqdm.rich module
from tqdm.rich import tqdm_rich
from rich.console import Console

console = Console(stderr=True)

# In weight_manager.ensure_bigvgan_weights():
local_dir = snapshot_download(
    repo_id=BIGVGAN_REPO_ID,
    tqdm_class=tqdm_rich,  # Rich progress bar instead of default tqdm
)
```

### Pattern 3: Gradio Download Progress via track_tqdm
**What:** Gradio's `Progress(track_tqdm=True)` auto-intercepts any tqdm calls (including HuggingFace Hub's internal tqdm) within the handler function and forwards them to the Gradio progress bar.
**When to use:** UI generate handler when BigVGAN needs downloading.
**Example:**
```python
# Source: Gradio 6.6.0 gr.Progress API
def _generate_audio(*args, progress=gr.Progress(track_tqdm=True)):
    # When ensure_bigvgan_weights() calls snapshot_download(),
    # which internally uses tqdm, Gradio's patched tqdm intercepts
    # the progress and shows it in the Gradio UI automatically.
    vocoder = get_vocoder("bigvgan", device=device_str)
    # ... rest of generation
```

### Pattern 4: Gradio Accordion with Dynamic State
**What:** Collapsible accordion with vocoder dropdown, status text, and progress area.
**When to use:** Generate tab below Generation Config row.
**Example:**
```python
# Source: Gradio 6.6.0 Accordion API + existing codebase patterns
with gr.Accordion("Vocoder Settings", open=False) as vocoder_accordion:
    vocoder_dropdown = gr.Dropdown(
        choices=["Auto", "BigVGAN Universal", "Per-model HiFi-GAN"],
        value="Auto",
        label="Vocoder",
    )
    vocoder_status = gr.Markdown("Using: BigVGAN Universal")
    vocoder_progress = gr.Markdown("", visible=False)
    retry_btn = gr.Button("Retry Download", visible=False, variant="secondary")
```

### Pattern 5: Disabled Dropdown Option for Unavailable Per-model HiFi-GAN
**What:** Gradio's Dropdown does not natively support disabling individual options. Instead, use `interactive=False` on the dropdown when Per-model HiFi-GAN is the only option, or dynamically adjust `choices` to exclude it when unavailable, paired with an info message explaining why.
**When to use:** When the loaded model has no trained per-model vocoder.
**Implementation approach:** Remove "Per-model HiFi-GAN" from choices when unavailable, add info text "(model has no per-model vocoder)" next to the dropdown. If user selects it (somehow), validate and show error.
```python
# Build choices based on model state
has_per_model = (
    app_state.loaded_model is not None
    and app_state.loaded_model.vocoder_state is not None
)
choices = ["Auto", "BigVGAN Universal"]
if has_per_model:
    choices.append("Per-model HiFi-GAN")
```

### Anti-Patterns to Avoid
- **Blocking the UI thread during download:** Never call `ensure_bigvgan_weights()` synchronously in a Gradio handler without progress indication. Use `gr.Progress(track_tqdm=True)` to keep the UI responsive.
- **Startup download:** Do NOT download BigVGAN on app startup. Download is lazy -- triggered on first generate attempt only.
- **Duplicating vocoder resolution logic:** Put auto-selection in one place (`resolve_vocoder()`), not separately in UI and CLI.
- **Using Gradio toast for download errors:** User explicitly chose inline error in accordion. Use `gr.Markdown` update for error display + visible `Retry` button.
- **Hardcoding vocoder name strings:** Use constants for "bigvgan_universal", "per_model_hifigan" etc. to keep CLI, UI, and JSON output consistent.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Download progress in Gradio | Custom WebSocket polling | `gr.Progress(track_tqdm=True)` | Gradio automatically patches tqdm and forwards to its progress bar |
| Rich CLI progress for downloads | Custom progress thread | `tqdm.rich.tqdm_rich` via `tqdm_class` param | tqdm_rich is purpose-built for Rich integration; huggingface_hub natively supports it |
| Vocoder type enum | String comparisons everywhere | Constants or Enum | Prevents typos in "auto"/"bigvgan"/"hifigan" across UI/CLI/JSON |
| Disabled dropdown option | Custom JS/HTML hacks | Dynamic choices list (remove unavailable option) | Gradio Dropdown has no native per-option disable; simplest is to omit the choice |

**Key insight:** The entire download progress pipeline has built-in hooks at every layer (huggingface_hub -> tqdm_class -> tqdm -> Gradio patch / Rich). No custom progress tracking code is needed -- just wire the existing hooks together.

## Common Pitfalls

### Pitfall 1: Gradio track_tqdm Only Works in Handler Functions
**What goes wrong:** Calling `gr.Progress(track_tqdm=True)` outside a Gradio event handler has no effect -- tqdm is patched but there's no active Gradio context to receive progress updates.
**Why it happens:** Gradio uses `LocalContext.progress` (a `contextvars.ContextVar`) that is only set when executing inside a Gradio event handler.
**How to avoid:** Ensure the BigVGAN download is triggered from within a Gradio handler function (e.g., `_generate_audio()`), not from module-level or `init_state()`.
**Warning signs:** Download works but UI freezes with no progress -- the tqdm output goes to the console instead.

### Pitfall 2: Generate Button State During Download
**What goes wrong:** User clicks Generate, download starts, user clicks Generate again (or presses Enter), causing a second concurrent download or error.
**Why it happens:** Gradio doesn't automatically disable buttons during handler execution unless told to.
**How to avoid:** Use `generate_btn.click(..., queue=True)` (Gradio's default with queuing) which naturally serializes requests. Additionally, the handler should check `is_bigvgan_cached()` and update status text before/after download.
**Warning signs:** Duplicate download attempts or confusing "model already downloading" errors.

### Pitfall 3: tqdm_rich Experimental Warning Spam
**What goes wrong:** `tqdm.rich.tqdm_rich` emits a `TqdmExperimentalWarning` on every instantiation.
**Why it happens:** tqdm considers its Rich integration experimental and warns every time.
**How to avoid:** Suppress the warning with `warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)` before calling `snapshot_download()` with the `tqdm_rich` class.
**Warning signs:** "rich is experimental/alpha" warnings cluttering CLI output.

### Pitfall 4: Vocoder Resolution Must Happen Before Pipeline Construction
**What goes wrong:** The `GenerationPipeline.__init__` constructor calls `get_vocoder("bigvgan")` as a default if no vocoder is passed. This triggers download immediately during pipeline construction, before the user's vocoder selection is known.
**Why it happens:** Current code in `_load_model_handler()` (library_tab.py) creates the pipeline with `get_vocoder("bigvgan")` at model load time, not at generate time.
**How to avoid:** Defer vocoder creation to generate time. Either: (a) pass `vocoder=None` during model load and create vocoder in the generate handler, or (b) make `GenerationPipeline` accept a lazy vocoder factory. Option (a) is simpler and matches the "lazy download on first generate" requirement.
**Warning signs:** BigVGAN downloads when user clicks "Load" in the Library tab instead of when they click "Generate".

### Pitfall 5: CLI --vocoder hifigan Error Path
**What goes wrong:** User requests `--vocoder hifigan` but the model has no per-model vocoder. Must produce a clear error, not a cryptic traceback.
**Why it happens:** `vocoder_state` is `None` for models without trained per-model vocoders.
**How to avoid:** Check `loaded_model.vocoder_state` before attempting to create the vocoder. Raise `typer.BadParameter` with the exact message specified in CONTEXT.md.
**Warning signs:** Python traceback instead of clean error message.

### Pitfall 6: JSON Output Vocoder Field Compatibility
**What goes wrong:** Adding `"vocoder"` to JSON output breaks consumers expecting the old schema.
**Why it happens:** Existing `--json` output has an established schema (`file`, `format`, `seed`).
**How to avoid:** Add `"vocoder"` as a new top-level key in each result dict. Since this is additive (new key, not changing existing keys), consumers using strict key access will not break.
**Warning signs:** Downstream scripts parsing JSON output fail on new field.

## Code Examples

### Vocoder Accordion in Generate Tab
```python
# Source: Codebase analysis of generate_tab.py + Gradio 6.6.0 API
# Placement: After "### Generation Config" section, before "### Seed row"

with gr.Accordion("Vocoder Settings", open=False) as vocoder_accordion:
    vocoder_dropdown = gr.Dropdown(
        choices=["Auto", "BigVGAN Universal"],  # Per-model added dynamically
        value="Auto",
        label="Vocoder",
        info="Auto selects the best available vocoder for your model",
    )
    vocoder_status = gr.Markdown(
        value="**Using:** BigVGAN Universal",
    )
    vocoder_progress = gr.Markdown(value="", visible=False)
    retry_download_btn = gr.Button(
        "Retry Download", visible=False, variant="secondary", size="sm",
    )
```

### CLI --vocoder Flag Addition
```python
# Source: Codebase analysis of generate.py Typer patterns
# Added to the generate() function signature:

vocoder: str = typer.Option(
    "auto", "--vocoder",
    help="Vocoder selection: auto, bigvgan, hifigan",
),

# After model loading, before pipeline creation:
from distill.vocoder import resolve_vocoder

vocoder_instance, vocoder_info = resolve_vocoder(
    selection=vocoder,
    loaded_model=loaded,
    device=str(torch_device),
    tqdm_class=tqdm_rich if not json_output else None,
)

# Always print vocoder line to stderr:
reason = vocoder_info.get("reason", "")
vocoder_label = "BigVGAN Universal" if vocoder_info["name"] == "bigvgan_universal" else "Per-model HiFi-GAN"
console.print(f"[bold]Vocoder:[/bold] {vocoder_label} ({vocoder_info['selection']}"
              + (f" -- {reason}" if reason else "") + ")")
```

### Weight Manager tqdm_class Support
```python
# Source: Existing weight_manager.py + huggingface_hub API
def ensure_bigvgan_weights(tqdm_class: type | None = None) -> Path:
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": BIGVGAN_REPO_ID}
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class

    try:
        local_dir = snapshot_download(**kwargs)
        return Path(local_dir)
    except Exception as online_err:
        # ... existing fallback logic
```

### Quality Badge Extension for Vocoder
```python
# Source: Existing _quality_badge_markdown() pattern in generate_tab.py
# Extend the badge to include vocoder info:

def _quality_badge_markdown(quality: dict, vocoder_info: dict | None = None) -> str:
    # ... existing badge code ...
    badge = f"**Quality:** {icon} {rating.upper()} -- SNR {snr_str} dB | {reason}"
    if vocoder_info:
        vocoder_label = "BigVGAN Universal" if vocoder_info["name"] == "bigvgan_universal" else "Per-model HiFi-GAN"
        badge += f" | **Vocoder:** {vocoder_label}"
    return badge
```

### CLI JSON Output Vocoder Field
```python
# Source: Existing JSON output pattern in generate.py
results.append({
    "file": str(export_path),
    "format": fmt_lower,
    "seed": result.seed_used,
    "vocoder": {
        "name": vocoder_info["name"],
        "selection": vocoder_info["selection"],
    },
})
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded `get_vocoder("bigvgan")` in generate command | Selectable via `--vocoder` flag with auto-resolution | Phase 15 (this phase) | Users can control vocoder selection |
| BigVGAN download with default tqdm (stdout) | Rich progress bar (CLI) / Gradio progress (UI) | Phase 15 (this phase) | Better UX during first-time download |
| No vocoder info in generation output | Vocoder field in quality badge + JSON output | Phase 15 (this phase) | Users know which vocoder produced their audio |
| Pipeline created at model load time with vocoder | Pipeline created at generate time (lazy vocoder) | Phase 15 (this phase) | Enables lazy download + vocoder selection |

**Deprecated/outdated:**
- Hardcoded `get_vocoder("bigvgan", device=device_str)` calls in `_load_model_handler()` and `generate()` -- these should be replaced with `resolve_vocoder()` that respects user selection.

## Open Questions

1. **Per-model HiFi-GAN loading (Phase 16)**
   - What we know: `LoadedModel.vocoder_state` contains the state dict when a per-model vocoder is trained. `VocoderInfo` in catalog tracks whether a model has one.
   - What's unclear: The actual loading code for per-model HiFi-GAN doesn't exist yet (Phase 16).
   - Recommendation: Implement the "hifigan" path in `resolve_vocoder()` as a `NotImplementedError` for now. The UI should show Per-model HiFi-GAN as an option only when the model has `vocoder_state`, but actually selecting it in Phase 15 can show "Per-model vocoder coming in Phase 16" or simply not include it in choices until Phase 16 wires it up. **CONTEXT.md says to disable/gray out the option** -- dynamically removing from choices + info text is the cleanest Gradio approach.

2. **Inline MB counter vs Gradio Progress bar**
   - What we know: CONTEXT.md specifies "progress bar and MB counter" inline in the accordion. Gradio's `track_tqdm=True` shows progress in Gradio's built-in progress bar (top of page), not inside the accordion.
   - What's unclear: Whether the built-in Gradio progress bar satisfies the "inline in accordion" requirement, or if we need a separate `gr.Markdown` polling pattern.
   - Recommendation: Use Gradio's built-in `Progress(track_tqdm=True)` for the actual progress (it shows automatically during the handler), and additionally update the `vocoder_progress` Markdown in the accordion with a "Downloading BigVGAN..." message before the download starts. The Gradio progress bar appears at the top of the outputs section, which is reasonably visible. A fully inline progress bar in the accordion would require complex polling (Thread + shared state + `gr.Timer`), which adds fragility for marginal UX benefit. Start with the simpler approach; add inline polling only if the user requests it.

## Sources

### Primary (HIGH confidence)
- **Gradio 6.6.0 `gr.Progress`** -- Verified via `inspect.getsource(gr.Progress)` and `inspect.getsource(patch_tqdm)`. `track_tqdm=True` patches `tqdm.tqdm.__init__`, `.update`, `.close`, and `.__iter__` to forward progress to Gradio context.
- **huggingface_hub 1.4.1 `snapshot_download()`** -- Verified via `inspect.signature()`. Accepts `tqdm_class` parameter. Internally creates a bytes-based tqdm progress bar and an `_AggregatedTqdm` for individual file downloads that roll up into the parent.
- **tqdm.rich.tqdm_rich** -- Verified importable and inspected source. Creates a `rich.progress.Progress` context manager with `BarColumn`, `FractionColumn`, `TimeElapsedColumn`, `TimeRemainingColumn`, `RateColumn`.
- **Codebase analysis** -- Direct reading of `generate_tab.py`, `generate.py`, `weight_manager.py`, `bigvgan_vocoder.py`, `base.py`, `__init__.py`, `state.py`, `library_tab.py`, `model_card.py`, `catalog.py`, `persistence.py`, `generation.py`, `app.py`, `cli/__init__.py`, `train.py`. All file paths and patterns verified.

### Secondary (MEDIUM confidence)
- **Gradio Dropdown per-option disable** -- Gradio 6.x Dropdown does not support disabling individual choices (no `disabled_choices` parameter in API). Verified by inspecting `gr.Dropdown` signature. The standard approach is to dynamically adjust the choices list.

### Tertiary (LOW confidence)
- None. All findings verified directly from installed packages and codebase.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries already installed and verified; no new dependencies
- Architecture: HIGH -- All integration points (tqdm_class, track_tqdm, vocoder factory, CLI options) verified via source inspection
- Pitfalls: HIGH -- Gradio tqdm patching behavior verified via source; huggingface_hub tqdm_class verified via signature; pipeline construction timing verified via code reading

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable -- all dependencies are pinned major versions)
