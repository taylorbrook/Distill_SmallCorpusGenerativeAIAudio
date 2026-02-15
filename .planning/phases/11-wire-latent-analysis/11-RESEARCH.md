# Phase 11: Wire Latent Space Analysis - Research

**Researched:** 2026-02-14
**Domain:** Integration wiring -- connecting existing modules across training, persistence, UI, and CLI
**Confidence:** HIGH

## Summary

Phase 11 is a pure integration/wiring phase. All the building blocks already exist in the codebase and are individually complete: `LatentSpaceAnalyzer` (Phase 5), `analysis_to_dict`/`analysis_from_dict` serialization (Phase 5), `save_model`/`load_model` with `latent_analysis` dict support (Phase 6), `GenerationConfig.latent_vector` (Phase 5), slider UI with `_update_sliders_for_model` (Phase 8), and CLI `--preset` support (Phase 9). The gap is that **none of the trigger points call the analyzer** and **the CLI train command doesn't auto-save a model after training**. The training loop (`training/loop.py`) finishes without running analysis; the train CLI (`cli/train.py`) finishes without saving to the model library; and the library save UI handler (`library_tab.py`) saves from checkpoint but never runs analysis first.

This research identified exactly five integration gaps that need closure, all of which involve inserting calls to existing APIs at existing lifecycle points. No new algorithms, no new data structures, no new libraries. The only subtlety is threading (analysis needs training data access, which is only available during/immediately-after training), the sklearn dependency (PCA is used inside the analyzer -- already in the codebase), and a pre-existing naming bug (`metadata.model_name` vs `metadata.name` in generate_tab.py).

**Primary recommendation:** Wire analysis as a post-training step (after final checkpoint, before `TrainingCompleteEvent`), with the result stored in the checkpoint dict and propagated into any model save operation. All integration points are well-defined and the existing APIs are compatible. This is a low-risk, high-clarity wiring phase.

## Standard Stack

### Core

No new libraries needed. All dependencies are already in the project.

| Library | Purpose | Already Present | Notes |
|---------|---------|-----------------|-------|
| `sklearn.decomposition.PCA` | PCA fitting in LatentSpaceAnalyzer | Yes (controls/analyzer.py) | Lazy-imported inside analyze() |
| `scipy.stats.pearsonr` | Feature correlation | Yes (controls/analyzer.py) | Lazy-imported inside analyze() |
| `numpy` | Array operations | Yes (throughout) | Project-wide lazy import pattern |
| `torch` | Model inference, DataLoader | Yes (throughout) | Project-wide lazy import pattern |

### Supporting

No additional supporting libraries.

### Alternatives Considered

Not applicable -- this is integration of existing code, not introducing new technology.

## Architecture Patterns

### Current Architecture (Relevant Modules)

```
src/small_dataset_audio/
  controls/
    analyzer.py        # LatentSpaceAnalyzer.analyze() -- THE core analysis engine
    mapping.py         # sliders_to_latent(), get_slider_info() -- slider conversion
    serialization.py   # analysis_to_dict(), analysis_from_dict() -- checkpoint persistence
    features.py        # compute_audio_features() -- correlation target features
  training/
    loop.py            # train() orchestrator -- GAP: does NOT call analyzer after training
    runner.py          # TrainingRunner -- daemon thread wrapper around train()
    dataset.py         # create_data_loaders() -- needed to provide dataloader to analyzer
    checkpoint.py      # save_checkpoint() -- GAP: no latent_analysis field in checkpoint dict
  models/
    persistence.py     # save_model(), load_model() -- ALREADY supports analysis param
  cli/
    train.py           # train_cmd() -- GAP: no auto-save-model or auto-analyze
    generate.py        # generate() -- ALREADY supports --preset (uses sliders_to_latent)
  ui/
    state.py           # AppState singleton -- ALREADY has loaded_model with analysis
    tabs/
      train_tab.py     # _start_training() -- GAP: no post-training analysis trigger
      library_tab.py   # _save_model_handler() -- GAP: saves from checkpoint, no analysis
      generate_tab.py  # _update_sliders_for_model() -- ALREADY reads analysis from loaded model
    app.py             # Cross-tab wiring -- ALREADY chains load_btn -> update_sliders
```

### Gap 1: Analysis Not Run After Training

**What exists:** `train()` in `loop.py` returns `{model, metrics_history, output_dir, best_checkpoint_path}`. The `TrainingRunner` stores this in `self._result`. Neither the loop nor the runner calls `LatentSpaceAnalyzer.analyze()`.

**What's needed:** After training completes (before the `TrainingCompleteEvent` is emitted, or immediately after in the runner), call `LatentSpaceAnalyzer.analyze()` using the trained model and training dataloader. Store the `AnalysisResult` in the result dict as `analysis`.

**Integration point:** Best place is in `train()` after the final checkpoint save, before the completion event. The model, spectrogram, device, and file_paths are all in scope. A new dataloader can be created from `file_paths` (use ALL files, not split, for maximum PCA coverage). The analysis result should be stored in the result dict.

**Key considerations:**
- The analyzer needs a DataLoader of raw waveforms (the same format the training DataLoader provides -- `[B, 1, samples]` tensors). Can reuse `AudioTrainingDataset` with no augmentation.
- The analyzer also needs an `AudioSpectrogram` instance. One already exists in `train()` scope as `spectrogram`.
- Analysis takes time (sweep points per component, feature extraction). A progress callback or log message is sufficient for v1.
- On analysis failure, log warning and continue (model is still usable without analysis). The analysis field should be `None` in this case.

### Gap 2: Analysis Not Saved in Checkpoints

**What exists:** `save_checkpoint()` in `checkpoint.py` saves `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `training_config`, `spectrogram_config`, `metrics_history`. No `latent_analysis` field.

**What's needed:** When analysis exists, serialize it via `analysis_to_dict()` and include as `latent_analysis` key in the checkpoint dict. When loading checkpoints for resume (`load_checkpoint()`), the analysis dict is ignored (it lives in the raw checkpoint dict, accessible by the caller).

**Integration point:** Add optional `latent_analysis` parameter to `save_checkpoint()`. The final checkpoint save in `train()` passes the analysis dict. The `save_model_from_checkpoint()` in `persistence.py` already reads `checkpoint.get("latent_analysis")` -- so it will just work once the checkpoint has the data.

### Gap 3: CLI Train Does Not Auto-Save Model

**What exists:** `cli/train.py` calls `train()` directly, prints results, and exits. It does not save the model to the library. The user would need to manually run a hypothetical `sda model save` command (which doesn't exist either).

**What's needed:** After training completes in the CLI, automatically save the best checkpoint as a `.sda` model file using `save_model_from_checkpoint()`. The analysis result (from Gap 1) should be included. The CLI should print the model path and ID.

**Integration point:** After the `result = run_training(...)` call in `train_cmd()`, call `save_model_from_checkpoint()` with the `best_checkpoint_path` from the result, auto-generating metadata from dataset info and training results. If analysis is in the result dict, it propagates through `save_model_from_checkpoint()` which already handles `checkpoint.get("latent_analysis")`.

**Alternative approach:** Instead of modifying `save_model_from_checkpoint`, pass the analysis through the checkpoint. Since `train()` saves a final checkpoint, and we add analysis to that checkpoint (Gap 2), `save_model_from_checkpoint` will pick it up automatically.

### Gap 4: UI Train Tab Does Not Auto-Analyze or Auto-Save

**What exists:** When training completes in the UI, `_poll_training()` in `train_tab.py` detects `is_complete`, updates the dashboard, and re-enables Start. The user must manually go to Library tab, enter a name, and click "Save to Library" -- which calls `_save_model_handler()` using `runner.result.best_checkpoint_path`. That handler uses `save_model_from_checkpoint()` which does NOT trigger analysis.

**What's needed:** Two options:
1. (Minimal) Have the training loop itself run analysis and store in checkpoint (Gap 1 + Gap 2). Then `_save_model_handler()` automatically picks it up from the checkpoint.
2. (Better UX) After training completes in the UI, run analysis and attach to runner result, then auto-prompt user to save.

Option 1 is simpler and sufficient: if analysis runs inside `train()` and gets saved to the final checkpoint, the existing save flow works without modification.

### Gap 5: CLI Generate Has No Direct Slider Support

**What exists:** `sda generate model --preset my_preset` works: it loads the model's analysis, loads the preset's slider positions, calls `sliders_to_latent()`, and passes `latent_vector` to `GenerationConfig`. But there's no `--slider` flag for direct slider position specification from the CLI.

**What's needed (for GEN-02 through GEN-06 via CLI):** A `--slider` option like `--slider 0:5 --slider 1:-3` that sets specific slider positions. This requires the model to have analysis. If no `--preset` and no `--slider`, generation uses random latent vectors (current default).

**Integration point:** Add `--slider` option to `generate()` in `cli/generate.py`. Parse as `INDEX:VALUE` pairs. Build `SliderState` from the values plus zeros for unspecified sliders. Convert to latent vector via `sliders_to_latent()`.

### Pre-existing Bug: `metadata.model_name` vs `metadata.name`

**Location:** `ui/tabs/generate_tab.py` lines 276, 346 reference `app_state.loaded_model.metadata.model_name` but `ModelMetadata` (in `models/persistence.py`) uses field name `name`, not `model_name`. Similarly, `library_tab.py` line 539 uses `e.model_name` but `ModelEntry` uses `name`. This likely causes `AttributeError` in the live UI when generating or exporting from a loaded model's generate tab. This should be fixed as part of this phase's wiring work since it's directly in the code paths being wired.

### Pattern: Post-Training Analysis Pipeline

The recommended integration pattern:

```python
# In training/loop.py, after final checkpoint save, before TrainingCompleteEvent:

# Run latent space analysis
analysis = None
try:
    from small_dataset_audio.controls.analyzer import LatentSpaceAnalyzer
    from small_dataset_audio.training.dataset import AudioTrainingDataset
    from torch.utils.data import DataLoader

    analyzer = LatentSpaceAnalyzer()
    # Use ALL training files (not split) for best PCA coverage
    analysis_dataset = AudioTrainingDataset(
        file_paths=file_paths,
        chunk_samples=int(1.0 * 48_000),
        augmentation_pipeline=None,
    )
    analysis_loader = DataLoader(
        analysis_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    analysis = analyzer.analyze(
        model=model,
        dataloader=analysis_loader,
        spectrogram=spectrogram,
        device=device,
    )
    logger.info("Latent space analysis complete: %d components", analysis.n_active_components)
except Exception:
    logger.warning("Latent space analysis failed", exc_info=True)
    # Continue without analysis -- model is still usable
```

### Pattern: Save Analysis in Checkpoint

```python
# In save_checkpoint, add optional analysis dict:
def save_checkpoint(..., latent_analysis: dict | None = None) -> Path:
    checkpoint = {
        ...existing fields...,
        "latent_analysis": latent_analysis,
    }
```

### Pattern: CLI Auto-Save After Training

```python
# In cli/train.py, after successful training:
from small_dataset_audio.models.persistence import ModelMetadata, save_model_from_checkpoint

metadata = ModelMetadata(
    name=f"trained_{dataset_dir.name}",
    dataset_name=dataset_dir.name,
    dataset_file_count=len(file_paths),
    training_epochs=epochs_completed,
    final_train_loss=final_train_loss,
    final_val_loss=final_val_loss,
)
model_path = save_model_from_checkpoint(
    checkpoint_path=best_checkpoint,
    metadata=metadata,
    models_dir=models_dir,
)
```

### Anti-Patterns to Avoid

- **Running analysis in a separate thread from training:** The model and data are already in scope at training completion. Don't add threading complexity -- run analysis inline.
- **Making analysis mandatory for generation:** Analysis should remain optional. Random latent vectors (current default when no analysis) must continue to work. Guard all analysis-dependent code paths with `if analysis is not None:`.
- **Storing sklearn PCA objects in checkpoints:** Already handled -- the project stores numpy arrays only (Phase 5 decision). Don't regress.
- **Importing controls module at training module load time:** Keep lazy imports inside function bodies (project-wide pattern).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PCA analysis | Custom PCA | `LatentSpaceAnalyzer.analyze()` | Already built in Phase 5, handles all edge cases |
| Slider-to-latent conversion | Manual math | `sliders_to_latent()` | Already built in Phase 5 |
| Analysis serialization | Custom dict building | `analysis_to_dict()`/`analysis_from_dict()` | Already built in Phase 5, handles versioning |
| Model save with analysis | Custom file format | `save_model()` with `analysis` param | Already built in Phase 6, handles catalog updates |
| Model load with analysis | Custom deserialization | `load_model()` | Already built in Phase 6, returns `LoadedModel.analysis` |
| Slider UI update | Custom Gradio wiring | `_update_sliders_for_model()` | Already built in Phase 8, handles visibility/labels |

**Key insight:** Every component already exists. Phase 11 is ONLY about inserting calls to these existing APIs at the right lifecycle points. Zero new algorithms or data structures needed.

## Common Pitfalls

### Pitfall 1: DataLoader Shape Mismatch

**What goes wrong:** The analyzer expects `DataLoader` yielding raw waveform tensors `[B, 1, samples]` or `[B, samples]`. It handles both shapes (line 181-182 of analyzer.py: `if waveforms.dim() == 2: waveforms = waveforms.unsqueeze(1)`). However, the `AudioTrainingDataset.__getitem__` returns `[1, chunk_samples]`, so after batching it's `[B, 1, chunk_samples]`. This is the correct shape.
**How to avoid:** Use `AudioTrainingDataset` directly with no augmentation for the analysis DataLoader. Don't try to reuse the training DataLoader (it has augmentation).

### Pitfall 2: Model Eval Mode Not Restored

**What goes wrong:** Analysis calls `model.eval()` internally and restores the previous training state in a `finally` block. But if analysis runs between the training loop and checkpoint save, the model should be in eval mode for analysis, then the final state doesn't matter (training is done). No issue here.
**How to avoid:** Run analysis after the final checkpoint save (model state is already persisted).

### Pitfall 3: Analysis Failure Crashes Training

**What goes wrong:** If `sklearn` import fails, or the PCA fit encounters numerical issues (e.g., all-zero latent vectors from a badly trained model), an unhandled exception could crash the training completion flow.
**How to avoid:** Wrap analysis in try/except at every call site. Log the warning and continue with `analysis = None`. The downstream code (save_model, UI sliders) already handles `analysis is None` gracefully.

### Pitfall 4: Thread Safety in UI Post-Training Analysis

**What goes wrong:** Training runs in a daemon thread via `TrainingRunner`. If analysis runs inside the training thread, it's fine. If someone tries to run analysis from the Timer poll callback (`_poll_training`), it would run on the Gradio thread while the model might still be in use by the training thread.
**How to avoid:** Run analysis INSIDE the training loop/thread, not from the UI timer. This is the recommended approach (Gap 1).

### Pitfall 5: Checkpoint Format Backward Compatibility

**What goes wrong:** Adding `latent_analysis` to checkpoints could break loading of old checkpoints that don't have this key.
**How to avoid:** Use `.get("latent_analysis")` with `None` default (already done in `save_model_from_checkpoint`). For `save_checkpoint`, the new field is simply absent in old checkpoints -- `checkpoint.get("latent_analysis")` returns `None`, which is the correct behavior.

### Pitfall 6: CLI Train Model Name Collision

**What goes wrong:** Auto-saving a model with a generated name like `trained_my_dataset` could collide with an existing model of the same name.
**How to avoid:** `save_model()` in `persistence.py` already handles duplicate filenames with a counter suffix (lines 208-212). The UUID model_id ensures unique identification regardless of name.

### Pitfall 7: metadata.model_name AttributeError

**What goes wrong:** `generate_tab.py` references `metadata.model_name` but the field is `metadata.name`. This causes `AttributeError` when generating from a loaded model.
**How to avoid:** Fix all references to use `metadata.name`. Search for `model_name` attribute access on `ModelMetadata` and `ModelEntry` objects.

## Code Examples

### Example 1: Post-Training Analysis in loop.py

```python
# Source: Codebase analysis -- integration of existing controls/analyzer.py
# Insert after _save_checkpoint_safe() in the finalize section of train()

analysis_result = None
try:
    from small_dataset_audio.controls.analyzer import LatentSpaceAnalyzer
    from small_dataset_audio.training.dataset import AudioTrainingDataset
    from torch.utils.data import DataLoader as TorchDataLoader

    logger.info("Running latent space analysis...")
    analyzer = LatentSpaceAnalyzer()

    # Create analysis dataloader from ALL files (not split)
    analysis_dataset = AudioTrainingDataset(
        file_paths=file_paths,
        chunk_samples=int(1.0 * 48_000),
        augmentation_pipeline=None,
    )
    analysis_loader = TorchDataLoader(
        analysis_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues at end of training
    )

    analysis_result = analyzer.analyze(
        model=model,
        dataloader=analysis_loader,
        spectrogram=spectrogram,
        device=device,
    )
    logger.info(
        "Analysis complete: %d active components, %.1f%% variance explained",
        analysis_result.n_active_components,
        sum(analysis_result.explained_variance_ratio) * 100,
    )
except Exception:
    logger.warning("Latent space analysis failed -- model saved without analysis", exc_info=True)
```

### Example 2: CLI Slider Support

```python
# Source: Codebase analysis -- extending cli/generate.py
# Add to generate() parameters:
slider: Annotated[
    Optional[list[str]],
    typer.Option("--slider", help="Set slider position as INDEX:VALUE (e.g., --slider '0:5' --slider '1:-3')"),
] = None,

# In the body, after preset handling:
if slider and loaded.analysis is not None:
    from small_dataset_audio.controls.mapping import SliderState, sliders_to_latent

    n_active = loaded.analysis.n_active_components
    positions = [0] * n_active  # defaults to center

    for s in slider:
        idx_str, val_str = s.split(":", 1)
        idx = int(idx_str)
        val = int(val_str)
        if 0 <= idx < n_active:
            positions[idx] = val

    slider_state = SliderState(positions=positions, n_components=n_active)
    latent_vector = sliders_to_latent(slider_state, loaded.analysis)
```

### Example 3: Saving Analysis in Checkpoint

```python
# Source: Codebase analysis -- extending training/checkpoint.py
# Add to save_checkpoint() signature:
def save_checkpoint(
    ...,
    latent_analysis: dict | None = None,
) -> Path:
    checkpoint = {
        ...,
        "latent_analysis": latent_analysis,  # None if not analyzed yet
    }
```

## State of the Art

Not applicable -- this phase is integration of existing project code, not adoption of new technology.

## Open Questions

1. **Auto-save model name in CLI**
   - What we know: The CLI train command will auto-save a model after training. It needs a name.
   - What's unclear: Should the auto-generated name use the dataset directory name, a timestamp, or prompt the user?
   - Recommendation: Use `f"{dataset_dir.name}_{timestamp}"` as a sensible default. Add `--model-name` option for override. The user can rename later via the UI library tab.

2. **Should analysis run on cancellation?**
   - What we know: When training is cancelled, a checkpoint is saved. The model may be partially trained.
   - What's unclear: Should analysis run on a partially trained model?
   - Recommendation: No. Analysis should only run on normal completion (not cancellation). Partially trained models may have collapsed posteriors that produce meaningless PCA components. The user can always re-analyze after resume-to-completion.

3. **Analysis progress reporting in CLI**
   - What we know: Analysis involves encoding all training data + PCA + feature sweeps. Could take 10-60 seconds depending on dataset size.
   - What's unclear: How to show progress in CLI vs UI.
   - Recommendation: CLI: Rich spinner with status text ("Analyzing latent space..."). UI: Since analysis runs inside the training thread, the Timer poll can detect a new metrics_buffer key (e.g., `analysis_running: True`) and show a status message. Keep it simple.

4. **model_name vs name attribute inconsistency**
   - What we know: `generate_tab.py` uses `metadata.model_name` but `ModelMetadata` has `name`. `library_tab.py` line 539 uses `e.model_name` but `ModelEntry` has `name`.
   - What's unclear: Whether `model_name` was intended as a property accessor or is a plain bug.
   - Recommendation: Fix to use `metadata.name` / `entry.name`. This is a straightforward bug. Scan for all occurrences in the codebase and fix.

## Sources

### Primary (HIGH confidence)

All findings are from direct codebase inspection -- no external sources needed for this integration phase.

- `src/small_dataset_audio/controls/analyzer.py` -- LatentSpaceAnalyzer implementation, AnalysisResult dataclass
- `src/small_dataset_audio/controls/mapping.py` -- sliders_to_latent(), get_slider_info(), SliderState
- `src/small_dataset_audio/controls/serialization.py` -- analysis_to_dict(), analysis_from_dict()
- `src/small_dataset_audio/controls/features.py` -- compute_audio_features(), FEATURE_NAMES
- `src/small_dataset_audio/models/persistence.py` -- save_model(), load_model(), ModelMetadata, LoadedModel
- `src/small_dataset_audio/training/loop.py` -- train() orchestrator
- `src/small_dataset_audio/training/runner.py` -- TrainingRunner background thread
- `src/small_dataset_audio/training/checkpoint.py` -- save_checkpoint(), load_checkpoint()
- `src/small_dataset_audio/training/dataset.py` -- AudioTrainingDataset, create_data_loaders()
- `src/small_dataset_audio/cli/train.py` -- train_cmd() CLI command
- `src/small_dataset_audio/cli/generate.py` -- generate() CLI command with --preset support
- `src/small_dataset_audio/ui/tabs/train_tab.py` -- UI training tab
- `src/small_dataset_audio/ui/tabs/generate_tab.py` -- UI generate tab with slider wiring
- `src/small_dataset_audio/ui/tabs/library_tab.py` -- UI library tab with save/load handlers
- `src/small_dataset_audio/ui/app.py` -- Cross-tab wiring (load_btn -> update_sliders)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all code exists
- Architecture: HIGH -- all integration points are in code I directly inspected, APIs are well-documented
- Pitfalls: HIGH -- identified from actual code paths and known project patterns

**Research date:** 2026-02-14
**Valid until:** 2026-03-14 (stable -- no external dependency changes expected)
