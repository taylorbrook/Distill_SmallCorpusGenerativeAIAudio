# Phase 13: VQ-VAE Training Pipeline - Research

**Researched:** 2026-02-21
**Domain:** VQ-VAE training loop integration, codebook health monitoring, v2 model persistence, Gradio/CLI training controls
**Confidence:** HIGH

## Summary

Phase 13 wires the Phase 12 VQ-VAE architecture into the existing training pipeline. The codebase already has a complete v1.0 training system: `training/loop.py` (train/validate epochs + orchestrator), `training/runner.py` (background threading), `training/config.py` (adaptive config), `training/metrics.py` (callback events), `training/checkpoint.py` (save/resume), `ui/tabs/train_tab.py` (Gradio dashboard), and `cli/train.py` (CLI with Rich progress). The v1.0 system uses `ConvVAE` with `vae_loss` (MSE + KL divergence). Phase 13 replaces this with `ConvVQVAE` + `vqvae_loss` (multi-scale spectral + commitment). The VQ-VAE model, loss function, and `VQVAEConfig` already exist from Phase 12 -- this phase is purely integration.

The key additions beyond basic training loop changes are: (1) per-level codebook health metrics displayed in both UI and CLI during training, (2) low utilization warnings at the 30% threshold, (3) v2 model persistence format with VQ-specific metadata, and (4) VQ-VAE-specific controls in the training UI (auto codebook size, RVQ levels slider, commitment weight input).

**Primary recommendation:** Create a parallel VQ-VAE training path (`train_vqvae` function in `loop.py` or a new `vqvae_loop.py`) that mirrors the existing `train()` structure but uses `ConvVQVAE`, `vqvae_loss`, and emits codebook health via the existing callback system. Extend the metrics event dataclasses with VQ-specific fields. Implement v2 persistence as a separate save/load path in `persistence.py` with version=2 and VQ-specific metadata.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Inline per-level rows in the training metrics panel -- each RVQ level gets its own row showing utilization %, perplexity, and dead code count
- CLI training output shows the same per-level detail (not a compact summary)
- Codebook sizing is auto-only in the UI -- no manual override. Size auto-scales from dataset size (64/128/256)
- RVQ levels: slider constrained to 2, 3, or 4 with descriptive labels ("Fewer levels = faster, more levels = finer detail")
- Commitment weight: pre-filled sensible default (e.g., 0.25) with slider/input to adjust -- most users leave it alone
- Same .distill extension for v2 -- differentiated internally by file header/magic bytes (NOTE: context says ".sda" but actual codebase uses ".distill" extension)
- v1 models are left behind -- do not support loading v1 .distill files at all
- v2 metadata includes: training config (codebook size, RVQ levels, commitment weight, codebook dim) + final codebook health snapshot per level + training loss curve history
- CLI: codebook size auto-determined by default, but --codebook-size flag available for power user override
- CLI: --rvq-levels and --commitment-weight flags with sensible defaults
- CLI: --model-name flag for custom naming; auto-generates timestamp-based name if omitted
- CLI output: tqdm-style progress bar per epoch with per-level codebook health stats at each update interval
- CLI end-of-training: full summary printed -- final loss, per-level codebook health, model save path, and any warnings

### Claude's Discretion
- Warning presentation format for low codebook utilization (30% threshold)
- Codebook health update frequency during training
- VQ-VAE controls layout/grouping in Gradio training tab
- Codebook health display behavior on model load
- Exact commitment weight default value
- Progress bar implementation details

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VQVAE-04 | Per-level codebook utilization, perplexity, and dead code count displayed during training | `QuantizerWrapper.get_codebook_utilization()` already computes these metrics; need to emit via callback system and display in UI/CLI |
| VQVAE-07 | Training detects and warns when codebook utilization drops below 30% | Check utilization values from `get_codebook_utilization()` against threshold; emit warning event through callback |
| PERS-01 | VQ-VAE models save as v2 format with codebook state and VQ-specific metadata | New `save_model_v2` / `load_model_v2` in persistence.py with version=2, VQ metadata, codebook health snapshot |
| UI-03 | Training tab updated for VQ-VAE config (codebook size, RVQ levels, commitment weight) | Extend `build_train_tab()` with VQ-VAE controls section; replace v1.0 KL/latent_dim controls |
| CLI-01 | CLI supports VQ-VAE training with configurable codebook parameters | Add --codebook-size, --rvq-levels, --commitment-weight flags to `cli/train.py`; call VQ-VAE training path |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | Model training, optimization, checkpointing | Already in project, drives entire training pipeline |
| vector-quantize-pytorch | (installed) | ResidualVQ layer with EMA updates | Phase 12 dependency, wraps all VQ operations |
| Gradio | (installed) | Training dashboard UI | Phase 6 established all UI patterns |
| Typer + Rich | (installed) | CLI with progress bars | Phase 11 established CLI patterns |
| matplotlib | (installed) | Loss chart rendering | `build_loss_chart()` in `ui/components/loss_chart.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | (installed) | CLI progress bars | Already used via Rich Progress in cli/train.py |

### Alternatives Considered
None -- all libraries already in the project. No new dependencies needed.

## Architecture Patterns

### Recommended Project Structure

All changes fit within existing module structure:

```
src/distill/
├── models/
│   ├── persistence.py    # ADD: save_model_v2, load_model_v2
│   └── vqvae.py          # EXISTING (Phase 12)
├── training/
│   ├── config.py          # EXISTING: VQVAEConfig, get_adaptive_vqvae_config
│   ├── loop.py            # ADD: train_vqvae_epoch, validate_vqvae_epoch, train_vqvae
│   ├── metrics.py         # EXTEND: VQStepMetrics, VQEpochMetrics with codebook health
│   ├── runner.py          # EXTEND: TrainingRunner to accept VQ-VAE mode
│   └── preview.py         # EXTEND: VQ-VAE reconstruction preview (no random sampling)
├── ui/
│   ├── tabs/train_tab.py  # EXTEND: VQ-VAE controls, codebook health display
│   └── components/
│       └── loss_chart.py  # EXTEND: add commitment loss line
└── cli/
    └── train.py           # EXTEND: VQ-VAE flags
```

### Pattern 1: Parallel VQ-VAE Training Loop

**What:** Create `train_vqvae_epoch()`, `validate_vqvae_epoch()`, and `train_vqvae()` functions alongside the existing VAE equivalents. The VQ-VAE versions use `ConvVQVAE` + `vqvae_loss` instead of `ConvVAE` + `vae_loss`, and emit codebook health metrics.

**When to use:** For all VQ-VAE training. The v1.0 `train()` function remains unchanged (no v1.0 regression risk).

**Why parallel, not refactored:**
- The v1.0 and v1.1 training loops differ fundamentally: VQ-VAE has no KL divergence, no KL annealing, no reparameterization, no latent space analysis, no random sampling for previews.
- Trying to parameterize one function for both paradigms creates a fragile branching mess.
- Clean separation matches the Phase 12 pattern (separate `ConvVQVAE` from `ConvVAE`).

**Example shape:**
```python
def train_vqvae_epoch(
    model: ConvVQVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    spectrogram: AudioSpectrogram,
    device: torch.device,
    gradient_clip_norm: float,
    commitment_weight: float = 0.25,
    epoch: int = 0,
    callback: MetricsCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """One VQ-VAE training epoch."""
    model.train()
    for step, batch in enumerate(train_loader):
        mel = spectrogram.waveform_to_mel(batch.to(device))
        recon, indices, commit_loss = model(mel)
        total, recon_loss, weighted_commit = vqvae_loss(
            recon, mel, commit_loss, commitment_weight,
        )
        # NaN check, backward, gradient clip, optimizer step...
        # Compute codebook health periodically
        if step % health_interval == 0:
            health = model.quantizer.get_codebook_utilization(indices)
            # Check for low utilization warnings
            # Emit via callback
```

### Pattern 2: Extended Metrics Events for VQ-VAE

**What:** Add VQ-specific metrics event dataclasses that include codebook health data.

**Why:** The existing `StepMetrics` and `EpochMetrics` have VAE-specific fields (`kl_loss`, `kl_weight`, `kl_divergence`). Rather than making these optional and confusing, create VQ-specific variants.

**Example:**
```python
@dataclass
class VQStepMetrics:
    epoch: int
    step: int
    total_steps: int
    train_loss: float
    recon_loss: float
    commit_loss: float
    commitment_weight: float
    learning_rate: float
    step_time_s: float
    codebook_health: dict[str, dict[str, float | int]] | None = None

@dataclass
class VQEpochMetrics:
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_recon_loss: float
    val_commit_loss: float
    overfitting_gap: float
    learning_rate: float
    eta_seconds: float
    elapsed_seconds: float
    codebook_health: dict[str, dict[str, float | int]] | None = None
    utilization_warnings: list[str] | None = None
```

### Pattern 3: v2 Model Persistence

**What:** Save VQ-VAE models as version 2 `.distill` files with VQ-specific metadata. The format marker remains `"distill_model"` but version bumps to 2.

**Key differences from v1:**
- `model_type: "vqvae"` field to distinguish from v1 `"vae"`
- `model_state_dict` from `ConvVQVAE` (not `ConvVAE`)
- `vqvae_config` dict instead of `latent_dim`
- `codebook_health_snapshot` with per-level utilization/perplexity/dead_codes
- `loss_curve_history` with epoch-level train/val/recon/commit losses
- No `latent_analysis` (PCA analysis is v1.0-only, replaced by codebook health)

**Example save format:**
```python
saved = {
    "format": "distill_model",
    "version": 2,
    "model_type": "vqvae",
    "model_state_dict": model.state_dict(),
    "vqvae_config": {
        "codebook_dim": 128,
        "codebook_size": 256,
        "num_quantizers": 3,
        "decay": 0.95,
        "commitment_weight": 0.25,
        "threshold_ema_dead_code": 2,
        "dropout": 0.2,
    },
    "spectrogram_config": {...},
    "training_config": {...},  # full VQVAEConfig as dict
    "codebook_health_snapshot": {
        "level_0": {"utilization": 0.85, "perplexity": 54.2, "dead_codes": 38},
        "level_1": {"utilization": 0.72, "perplexity": 41.8, "dead_codes": 71},
        "level_2": {"utilization": 0.55, "perplexity": 29.1, "dead_codes": 115},
    },
    "loss_curve_history": {
        "train_losses": [...],
        "val_losses": [...],
        "recon_losses": [...],
        "commit_losses": [...],
    },
    "metadata": {...},  # ModelMetadata dict with VQ-specific fields
}
```

**Loading:** `load_model_v2()` checks `version >= 2` and `model_type == "vqvae"`, then reconstructs `ConvVQVAE` from `vqvae_config`.

### Pattern 4: UI Integration - VQ-VAE Controls in Train Tab

**What:** Replace or augment the v1.0 training controls with VQ-VAE-specific inputs.

**Layout approach:** Since codebook size is auto-only, the primary user controls are:
- RVQ levels slider (2-4, with descriptive labels)
- Commitment weight slider/input (default 0.25)
- The existing max_epochs, learning_rate, preset dropdown remain relevant

**Implementation:**
- The train tab currently has v1.0-specific controls (KL Weight, latent_dim implied by preset)
- Replace/hide the KL Weight slider with commitment weight slider
- Add RVQ levels slider
- Show auto-determined codebook size as a read-only display
- Add a codebook health section to the stats_md panel that updates during training

### Pattern 5: VQ-VAE Reconstruction Previews

**What:** VQ-VAE cannot generate from random N(0,1) samples (unlike continuous VAE). Previews must use reconstruction: encode a real audio chunk, quantize, decode, compare.

**Impact:** `generate_preview()` calls `model.sample()` which doesn't exist on `ConvVQVAE`. Phase 13 needs reconstruction-based previews using `generate_reconstruction_preview()` (already exists) or a VQ-specific variant.

### Anti-Patterns to Avoid
- **Don't modify the existing `train()` function to branch on model type:** This creates fragile code and regression risk. Create `train_vqvae()` as a parallel function.
- **Don't overload StepMetrics/EpochMetrics with optional VQ fields:** The callback consumers (UI timer, CLI callback) would need complex branching. Use separate VQ dataclasses.
- **Don't break v1.0 model loading:** Even though v1 models are "left behind," the `load_model()` function should remain as-is. Add `load_model_v2()` separately.
- **Don't compute codebook health every step:** `get_codebook_utilization()` involves `torch.unique` and `torch.bincount` on indices, which is O(batch_size * seq_len). Every 10-20 steps or once per epoch is sufficient.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Codebook utilization metrics | Custom counting code | `QuantizerWrapper.get_codebook_utilization()` | Already implemented in Phase 12 with correct perplexity formula |
| VQ-VAE loss computation | Manual reconstruction + commitment loss | `vqvae_loss()` | Already implemented in Phase 12 with multi-scale spectral loss |
| Model config from dataset size | Hard-coded parameters | `get_adaptive_vqvae_config(file_count)` | Already implemented in Phase 12 with 3-tier scaling |
| RVQ quantization | Manual codebook lookups | `ConvVQVAE.forward()` which wraps `ResidualVQ` | Already implemented in Phase 12 |
| Training progress bars | Custom terminal output | `rich.progress.Progress` | Existing pattern from cli/train.py |

**Key insight:** Phase 12 built all the model-level components. Phase 13's job is purely integration -- wiring these components into the training pipeline, UI, and CLI.

## Common Pitfalls

### Pitfall 1: Codebook Not Initialized Before First Health Check
**What goes wrong:** `get_codebook_utilization()` called before k-means initialization completes, returning misleading 0% utilization and triggering false warnings.
**Why it happens:** K-means initialization happens on the first forward pass. If health is checked before any training step, all codes appear dead.
**How to avoid:** Skip codebook health checks during step 0 of epoch 0, or wait until after the first forward pass. Gate warnings with `epoch > 0 or step > 0`.
**Warning signs:** 0% utilization on all levels at step 0.

### Pitfall 2: EMA Codebook Updates Require model.train() Mode
**What goes wrong:** Codebook EMA updates only happen during `model.train()`. If the model is accidentally left in eval mode during training, codebooks never update and utilization drops.
**Why it happens:** `ResidualVQ` from vector-quantize-pytorch uses `self.training` flag internally.
**How to avoid:** Ensure `model.train()` is called before each training epoch, and `model.eval()` only during validation. The existing v1.0 pattern does this correctly.
**Warning signs:** Codebook utilization dropping monotonically after epoch 0.

### Pitfall 3: Preview Generation Assumes model.sample() Exists
**What goes wrong:** The v1.0 `generate_preview()` calls `model.sample(num_samples, device)`. `ConvVQVAE` has no `sample()` method -- it needs a learned prior (Phase 14).
**Why it happens:** VQ-VAE decouples encoding from generation. Random latent sampling is a continuous VAE concept.
**How to avoid:** Use reconstruction-based previews for VQ-VAE training. Take a batch from the validation set, run encode-quantize-decode, save original+reconstruction pairs.
**Warning signs:** `AttributeError: 'ConvVQVAE' object has no attribute 'sample'`.

### Pitfall 4: Checkpoint Format Incompatibility
**What goes wrong:** VQ-VAE checkpoints saved with `save_checkpoint()` lack VQ-specific metadata (codebook health, VQ config). Resuming from checkpoint works but loses this information.
**Why it happens:** The existing `save_checkpoint()` stores `kl_weight` and v1.0-specific fields.
**How to avoid:** Either extend `save_checkpoint()` to accept optional VQ fields, or create a parallel checkpoint system for VQ-VAE.
**Warning signs:** Missing codebook health data after training resume.

### Pitfall 5: `.distill` Extension vs. Context's ".sda" Reference
**What goes wrong:** The CONTEXT.md mentions ".sda extension" but the actual codebase uses `.distill` extension (`MODEL_FILE_EXTENSION = ".distill"`).
**Why it happens:** Terminology mismatch between discussion and implementation.
**How to avoid:** Use `.distill` consistently -- that's what the code uses. The CONTEXT.md decision "Same .sda extension for v2" translates to "Same `.distill` extension for v2."
**Warning signs:** File extension inconsistencies causing load failures.

### Pitfall 6: Training Tab Becomes a Branching Mess
**What goes wrong:** Adding VQ-VAE controls alongside v1.0 controls creates a confusing UI with irrelevant controls visible.
**Why it happens:** Trying to support both training paradigms in one tab.
**How to avoid:** Since v1.0 models are "left behind," the train tab should show VQ-VAE controls exclusively. Remove or hide the KL Weight slider and other v1.0-only controls.
**Warning signs:** Users seeing KL Weight slider during VQ-VAE training.

### Pitfall 7: NaN in Commitment Loss with Large Gradients
**What goes wrong:** Commitment loss can produce NaN when gradients explode, especially on MPS.
**Why it happens:** The v1.0 training loop already has NaN detection and gradient clipping, but the VQ-VAE training path needs the same safeguards.
**How to avoid:** Copy the NaN detection pattern from `train_epoch()`: check `total.isnan()`, skip the gradient update, log a warning.
**Warning signs:** NaN loss at early epochs, especially on MPS device.

## Code Examples

Verified patterns from the existing codebase:

### Training Loop Structure (from existing loop.py)
```python
# Source: src/distill/training/loop.py lines 48-195
# Key pattern: iterate batches, convert to mel, forward pass, loss, backward, gradient clip
for step, batch in enumerate(train_loader):
    if cancel_event is not None and cancel_event.is_set():
        break
    batch = batch.to(device)
    mel = spectrogram.waveform_to_mel(batch)
    # ... forward, loss, backward, clip, optimizer step
    # ... NaN detection and step metrics callback
```

### VQ-VAE Forward Pass (from existing vqvae.py)
```python
# Source: src/distill/models/vqvae.py lines 513-547
recon, indices, commit_loss = model(mel)
# recon: [B, 1, n_mels, time]
# indices: [B, H*W, num_quantizers]
# commit_loss: scalar tensor
```

### VQ-VAE Loss (from existing losses.py)
```python
# Source: src/distill/models/losses.py lines 184-232
total_loss, recon_loss, weighted_commit = vqvae_loss(
    recon, mel, commit_loss, commitment_weight=0.25,
)
```

### Codebook Health (from existing vqvae.py)
```python
# Source: src/distill/models/vqvae.py lines 285-326
health = model.quantizer.get_codebook_utilization(indices)
# Returns: {"level_0": {"utilization": 0.85, "perplexity": 54.2, "dead_codes": 38}, ...}
```

### Callback Event Emission (from existing loop.py)
```python
# Source: src/distill/training/loop.py lines 172-179
if callback is not None:
    callback(StepMetrics(
        epoch=epoch, step=step, total_steps=total_steps,
        train_loss=total.item(), recon_loss=recon_loss.item(),
        kl_loss=kl_loss.item(), kl_weight=kl_weight,
        learning_rate=lr, step_time_s=step_time,
    ))
```

### Model Save (from existing persistence.py)
```python
# Source: src/distill/models/persistence.py lines 191-199
saved = {
    "format": MODEL_FORMAT_MARKER,   # "distill_model"
    "version": SAVED_MODEL_VERSION,  # currently 1, will be 2
    "model_state_dict": model.state_dict(),
    "spectrogram_config": spectrogram_config,
    "training_config": training_config,
    "metadata": asdict(metadata),
}
```

### Train Tab Timer Polling (from existing train_tab.py)
```python
# Source: src/distill/ui/tabs/train_tab.py lines 493-593
def _poll_training() -> list:
    buf = app_state.metrics_buffer
    epoch_metrics = buf.get("epoch_metrics", [])
    # ... build chart, stats string, audio updates
    # Returns list matching [loss_plot, stats_md, timer, start_btn, cancel_btn] + preview_audios
```

### CLI Callback (from existing cli/train.py)
```python
# Source: src/distill/cli/train.py lines 191-210
def cli_callback(event: object) -> None:
    if isinstance(event, EpochMetrics):
        progress.update(task_id, completed=event.epoch + 1, ...)
```

### Adaptive VQ-VAE Config (from existing config.py)
```python
# Source: src/distill/training/config.py lines 388-476
config = get_adaptive_vqvae_config(file_count)
# Returns VQVAEConfig with auto-scaled codebook_size, decay, dropout, epochs, etc.
```

## State of the Art

| Old Approach (v1.0) | Current Approach (v1.1 Phase 13) | When Changed | Impact |
|---------------------|----------------------------------|--------------|--------|
| `ConvVAE` + `vae_loss` (KL + MSE) | `ConvVQVAE` + `vqvae_loss` (multi-scale spectral + commitment) | Phase 12-13 | Entire training loop forward pass changes |
| KL annealing schedule | No annealing (commitment_weight is fixed) | Phase 12-13 | Simpler training loop, fewer hyperparameters |
| `model.sample()` for previews | Reconstruction-based previews (encode-quantize-decode) | Phase 13 | No random generation until prior in Phase 14 |
| `latent_dim=64`, PCA analysis | Codebook health metrics (utilization, perplexity, dead codes) | Phase 13 | Different model quality indicators |
| v1 save format (version=1) | v2 save format (version=2, VQ metadata) | Phase 13 | Breaking change, no backward compatibility |
| Single loss curve (train+val) | Loss curve + commitment loss + per-level codebook health | Phase 13 | Richer training dashboard |

**Deprecated/outdated:**
- `get_kl_weight()`: Not used in VQ-VAE training (no KL divergence)
- `compute_kl_divergence()`: Not applicable to VQ-VAE
- `ConvVAE.sample()`: VQ-VAE needs prior for generation
- `latent_dim` parameter: Replaced by `codebook_dim`, `codebook_size`, `num_quantizers`
- v1.0 model persistence functions: Remain for existing models but no new development

## Open Questions

1. **Should the existing `train()` function be kept or replaced?**
   - What we know: User decided "v1 models are left behind." The train tab will show VQ-VAE only.
   - What's unclear: Should we remove v1.0 training entirely or just make VQ-VAE the default?
   - Recommendation: Keep v1.0 `train()` unchanged but route all UI/CLI calls to the new `train_vqvae()`. Remove v1.0 training paths from UI/CLI. The v1.0 code stays in the codebase for reference but is not user-accessible.

2. **Checkpoint format: extend existing or create new?**
   - What we know: The existing `save_checkpoint()` has v1.0-specific fields (`kl_weight`). VQ-VAE needs `commitment_weight`, `codebook_health`, etc.
   - What's unclear: Whether to extend `save_checkpoint()` with optional VQ fields or create a parallel `save_vqvae_checkpoint()`.
   - Recommendation: Extend `save_checkpoint()` with optional VQ-specific fields. The function already accepts arbitrary configs as dicts. Add `codebook_health: dict | None = None` and `commitment_weight: float | None = None` parameters.

3. **How often to compute codebook health during training?**
   - What we know: `get_codebook_utilization()` is not free (involves unique/bincount operations). Users want to see health during training.
   - What's unclear: Exact update frequency.
   - Recommendation: Compute once per epoch (during validation pass when model is in eval mode and processing the full val set). This is both computationally cheap (once per epoch) and gives the most representative snapshot. For the CLI, print at epoch boundaries.

4. **MetricsHistory: extend or create VQ variant?**
   - What we know: `MetricsHistory` serializes to/from dict for checkpoint inclusion. It currently stores `EpochMetrics` with KL-specific fields.
   - What's unclear: Whether to add VQ-specific fields to existing classes or create new ones.
   - Recommendation: Create `VQEpochMetrics` and `VQMetricsHistory` that parallel the existing classes but contain VQ-specific fields. The metrics callback system (`MetricsCallback` type alias) should accept the union of all event types.

## Sources

### Primary (HIGH confidence)
- `src/distill/models/vqvae.py` -- ConvVQVAE, QuantizerWrapper, get_codebook_utilization (Phase 12 output)
- `src/distill/models/losses.py` -- vqvae_loss, multi_scale_mel_loss (Phase 12 output)
- `src/distill/training/config.py` -- VQVAEConfig, get_adaptive_vqvae_config (Phase 12 output)
- `src/distill/training/loop.py` -- existing train(), train_epoch(), validate_epoch() patterns
- `src/distill/training/runner.py` -- TrainingRunner background thread pattern
- `src/distill/training/metrics.py` -- StepMetrics, EpochMetrics, MetricsHistory, MetricsCallback
- `src/distill/training/checkpoint.py` -- save_checkpoint(), load_checkpoint() patterns
- `src/distill/models/persistence.py` -- save_model(), load_model(), ModelMetadata, LoadedModel
- `src/distill/ui/tabs/train_tab.py` -- Gradio train tab layout and timer polling pattern
- `src/distill/cli/train.py` -- CLI training with Rich progress and SIGINT handling
- `src/distill/ui/state.py` -- AppState, metrics_buffer, reset_metrics_buffer
- `src/distill/library/catalog.py` -- ModelEntry, ModelLibrary index management

### Secondary (MEDIUM confidence)
- vector-quantize-pytorch library -- EMA updates only occur in `model.train()` mode (verified from Phase 12 implementation notes)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in the project, no new dependencies
- Architecture: HIGH -- all patterns mirror existing v1.0 code with clear VQ-VAE substitutions
- Pitfalls: HIGH -- derived from direct codebase analysis and VQ-VAE training fundamentals

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (stable -- internal codebase patterns, no external API changes expected)
