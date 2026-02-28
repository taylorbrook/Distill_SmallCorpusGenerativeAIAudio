# Phase 16: Per-Model HiFi-GAN Training & Griffin-Lim Removal - Research

**Researched:** 2026-02-27
**Domain:** GAN-based neural vocoder training, adversarial audio synthesis, mel-to-waveform reconstruction
**Confidence:** HIGH

## Summary

Phase 16 implements per-model HiFi-GAN V2 vocoder training so users can train a small (0.92M parameter) neural vocoder on their specific training audio for maximum fidelity. The system auto-selects the best available vocoder (per-model HiFi-GAN > BigVGAN universal), and Griffin-Lim reconstruction code is fully removed from the codebase.

The HiFi-GAN V2 architecture uses a generator with transposed convolutions (upsample rates [8,8,2,2], initial channels 128) and two discriminators: Multi-Period Discriminator (MPD, periods [2,3,5,7,11]) and Multi-Scale Discriminator (MSD, 3 scales). Training uses adversarial loss + L1 mel spectrogram loss (weight 45) + feature matching loss, with AdamW optimizer (lr=0.0002, betas=[0.8, 0.99]) and ExponentialLR decay (gamma=0.999). The critical adaptation for this project is configuring HiFi-GAN for 128-band mels at the VAE's internal 48kHz sample rate (vs the original 80-band at 22.05kHz), which affects the generator's upsample rates to match the project's hop_length=512 and sample_rate=48000.

The biggest risk is discriminator overfitting on small datasets (5-50 files). Mitigation strategies include: data augmentation during vocoder training (reusing the existing `AugmentationPipeline`), lower discriminator learning rate, and contrastive learning techniques. A per-model HiFi-GAN trained on small datasets will produce audio specialized to that model's sound world -- even with modest quality, it eliminates the BigVGAN mel adapter's Griffin-Lim round-trip quality loss that is currently the system's main audio bottleneck.

**Primary recommendation:** Build the HiFi-GAN V2 training loop as a new `distill.vocoder.hifigan` module following the existing training infrastructure patterns (TrainingRunner, metrics callbacks, checkpoint save/load). Adapt the V2 config for 128-band 48kHz mels. Store vocoder state in the existing `vocoder_state` slot in .distillgan files. Remove Griffin-Lim from `AudioSpectrogram.mel_to_waveform` and `MelAdapter` simultaneously -- the per-model HiFi-GAN provides a direct VAE-mel-to-waveform path that bypasses the mel adapter entirely.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Vocoder training controls live in a **section below the existing VAE training controls** in the Train tab
- Section only enabled/visible after VAE training is complete for the selected model
- **Full parameter control** exposed: epochs, learning rate, batch size, checkpoint interval
- Training progress shows **live-updating loss curve chart** (generator + discriminator loss) plus text stats (epoch, current loss, ETA)
- **Periodic audio preview samples** generated during training so the user can hear improvement over time (every N epochs -- Claude decides frequency)
- CLI audio preview samples: Claude's discretion on whether to save periodic WAV previews to disk during CLI training
- **Mirror UI parameters exactly**: --epochs, --lr, --batch-size, --checkpoint-interval
- Training output uses **Rich live table** display showing epoch, loss values, ETA, and progress bar (consistent with existing Rich console usage)
- On cancel: **immediately save checkpoint** at current epoch, confirm with "Checkpoint saved at epoch N. Resume anytime." -- graceful stop, no data loss
- On restart with existing checkpoint: **ask the user** "Resume from epoch N or start fresh?" -- explicit choice every time
- Checkpoints stored **inside the .distillgan model file** itself -- self-contained, no external checkpoint files to manage
- **Retrain with confirmation**: if model already has a trained vocoder, warn "This model already has a trained vocoder. Replace it?" before proceeding
- After Griffin-Lim removal, if no vocoder is available: **auto-download BigVGAN** transparently on first generate (consistent with Phase 15 lazy download design)
- Quality badge after generation unchanged from Phase 15 decisions -- shows vocoder name alongside seed/sample-rate/bit-depth

### Claude's Discretion
- CLI command structure (subcommand vs top-level)
- Audio preview frequency during training (every N epochs)
- Whether CLI saves periodic audio preview WAVs to disk
- Auto-selection quality threshold policy
- Training defaults for each parameter (sensible defaults for small datasets)
- Data augmentation specifics (TRAIN-03)
- Adversarial loss implementation details (TRAIN-02)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | User can train HiFi-GAN V2 vocoder on a model's training audio | HiFi-GAN V2 architecture researched; config adapted for 128-band 48kHz mels; training loop pattern mirrors existing VAE training |
| TRAIN-02 | Training uses adversarial loss (MPD+MSD discriminators) + mel reconstruction loss + feature matching loss | Loss functions documented: GAN loss + L1 mel loss (weight 45) + feature matching loss; MPD periods [2,3,5,7,11], MSD 3 scales |
| TRAIN-03 | Data augmentation applied during training to prevent discriminator overfitting on small datasets | Existing AugmentationPipeline (speed perturbation, noise injection, volume variation) reusable; discriminator-specific augmentation researched |
| TRAIN-04 | Trained vocoder weights bundled into .distill model file | vocoder_state slot already exists in persistence layer; save/load patterns documented |
| TRAIN-05 | Training supports cancel with checkpoint save and resume | Checkpoint-inside-model design with cancel_event pattern from existing TrainingRunner |
| VOC-05 | Vocoder auto-selects best available: per-model HiFi-GAN > BigVGAN universal | resolve_vocoder() already has Phase 16 stubs; auto-selection logic documented |
| UI-03 | Train tab has "Train Vocoder" option for models with completed VAE training | Train tab architecture understood; vocoder section added below existing VAE controls |
| UI-04 | Vocoder training shows loss curve and progress in UI | Existing loss_chart.py and metrics callback pattern reusable with dual-loss (gen+disc) chart |
| CLI-02 | train-vocoder CLI command trains per-model HiFi-GAN vocoder | CLI patterns documented; Typer subcommand with Rich progress bar |
| GEN-04 | Griffin-Lim reconstruction code fully removed | Griffin-Lim locations identified: AudioSpectrogram.mel_to_waveform, MelAdapter.convert; removal plan documented |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | Generator, discriminator models, training loop | Already in project; GAN training uses standard torch.nn, torch.optim |
| torchaudio | (bundled) | Audio transforms, mel spectrogram computation | Already in project; MelSpectrogram transform used by VAE |
| Gradio | (existing) | UI for vocoder training controls | Already used for Train tab |
| Typer | (existing) | CLI train-vocoder command | Already used for all CLI commands |
| Rich | (existing) | CLI progress display | Already used for training progress |
| matplotlib | (existing) | Loss curve visualization | Already used via loss_chart.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| soundfile | (existing) | Preview WAV saving | During vocoder training preview generation |
| numpy | (existing) | Audio array manipulation | Training data pipeline |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom HiFi-GAN V2 | Vendored BigVGAN discriminators | BigVGAN discriminators are available in vendor/ but are coupled to BigVGAN's config system (AttrDict, d_mult); cleaner to implement standalone HiFi-GAN V2 discriminators |
| HiFi-GAN V1 (3.7M params) | HiFi-GAN V2 (0.92M params) | V2 is the right choice -- smaller, fits in .distillgan files, trains faster on small datasets |
| HiFi-GAN V3 (1.46M params) | HiFi-GAN V2 (0.92M params) | V3 uses ResBlock2 (simpler) but has more parameters; V2 is the sweet spot |

**No additional installation needed** -- all dependencies are already in the project.

## Architecture Patterns

### Recommended Project Structure
```
src/distill/vocoder/
├── __init__.py             # Updated: get_vocoder handles "hifigan", resolve_vocoder implements auto-select
├── base.py                 # VocoderBase (unchanged)
├── bigvgan_vocoder.py      # BigVGANVocoder (unchanged)
├── mel_adapter.py          # MODIFIED: Remove Griffin-Lim round-trip, rewrite for direct conversion
├── weight_manager.py       # BigVGAN weight management (unchanged)
├── hifigan/                # NEW: Per-model HiFi-GAN V2 module
│   ├── __init__.py         # Public API
│   ├── generator.py        # HiFi-GAN V2 generator (128-band, 48kHz adapted)
│   ├── discriminator.py    # MPD + MSD discriminators
│   ├── losses.py           # GAN loss, feature matching loss, mel loss
│   ├── config.py           # HiFiGANConfig dataclass with sensible defaults
│   ├── trainer.py          # Training loop (gen/disc alternation, metrics emission)
│   └── vocoder.py          # HiFiGANVocoder(VocoderBase) -- inference wrapper
```

### Pattern 1: GAN Alternating Training Loop
**What:** Generator and discriminator are updated in alternation within each training step. The discriminator is updated first (classifying real vs generated), then the generator is updated to fool the discriminator while also minimizing mel reconstruction loss and feature matching loss.
**When to use:** Every vocoder training step.
**Example:**
```python
# Discriminator step
y_g_hat = generator(mel)
# MPD
y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
# MSD
y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
loss_disc_all = loss_disc_s + loss_disc_f
# Backprop discriminator
optim_d.zero_grad()
loss_disc_all.backward()
optim_d.step()

# Generator step
y_g_hat = generator(mel)
y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
loss_gen_f, _ = generator_loss(y_df_hat_g)
loss_gen_s, _ = generator_loss(y_ds_hat_g)
loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
# Backprop generator
optim_g.zero_grad()
loss_gen_all.backward()
optim_g.step()
```

### Pattern 2: Vocoder State Bundling in .distillgan
**What:** The trained HiFi-GAN state_dict, config, and training metadata are stored in the existing `vocoder_state` slot of the .distillgan file format. Checkpoints during training are stored as temporary entries in this same slot, overwritten on save.
**When to use:** After vocoder training completes or on cancel/checkpoint.
**Example:**
```python
vocoder_state = {
    "type": "hifigan_v2",
    "generator_state_dict": generator.state_dict(),
    "config": asdict(hifigan_config),
    "training_metadata": {
        "epochs": completed_epochs,
        "final_loss": final_gen_loss,
        "training_date": datetime.now(timezone.utc).isoformat(),
    },
    # Checkpoint fields (only present during training, removed on final save)
    "checkpoint": {
        "epoch": current_epoch,
        "optim_g_state_dict": optim_g.state_dict(),
        "optim_d_state_dict": optim_d.state_dict(),
        "scheduler_g_state_dict": scheduler_g.state_dict(),
        "scheduler_d_state_dict": scheduler_d.state_dict(),
        "mpd_state_dict": mpd.state_dict(),
        "msd_state_dict": msd.state_dict(),
    },
}
```

### Pattern 3: Per-Model HiFi-GAN Vocoder as VocoderBase
**What:** The HiFiGANVocoder implements VocoderBase, loading the generator from vocoder_state. Unlike BigVGANVocoder which uses MelAdapter (Griffin-Lim round-trip), HiFiGANVocoder accepts VAE mel directly -- it was trained on VAE mels so no conversion is needed.
**When to use:** When a loaded model has vocoder_state and auto-selection or explicit hifigan is chosen.
**Example:**
```python
class HiFiGANVocoder(VocoderBase):
    def __init__(self, vocoder_state: dict, device: str = "auto"):
        config = HiFiGANConfig(**vocoder_state["config"])
        self._generator = HiFiGANGenerator(config)
        self._generator.load_state_dict(vocoder_state["generator_state_dict"])
        self._generator.eval()
        self._generator.to(self._device)

    def mel_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        # VAE mel [B, 1, 128, T] -> squeeze channel -> [B, 128, T]
        mel_squeezed = mel.squeeze(1).to(self._device)
        # Undo log1p normalization: expm1 -> power mel
        mel_linear = torch.expm1(mel_squeezed.clamp(min=0))
        with torch.inference_mode():
            wav = self._generator(mel_linear)  # [B, 1, T*hop_length]
        return wav

    @property
    def sample_rate(self) -> int:
        return 48000  # Trained on VAE's native 48kHz
```

### Pattern 4: Metrics Callback for Dual-Loss GAN Training
**What:** Extend the existing MetricsCallback pattern to emit GAN-specific metrics (gen_loss, disc_loss, mel_loss, feature_loss) alongside the existing epoch/step/ETA fields.
**When to use:** Vocoder training loop emits events consumed by UI timer and CLI progress.
**Example:**
```python
@dataclass
class VocoderEpochMetrics:
    epoch: int
    total_epochs: int
    gen_loss: float
    disc_loss: float
    mel_loss: float
    feature_loss: float
    learning_rate: float
    eta_seconds: float
    elapsed_seconds: float
```

### Anti-Patterns to Avoid
- **Training on BigVGAN mels instead of VAE mels:** The per-model HiFi-GAN must be trained on the VAE's log1p mel format (128 bands, 48kHz, HTK filterbank). If trained on BigVGAN's format, it would just be a worse BigVGAN clone. The whole point is learning the VAE-mel-to-waveform mapping directly.
- **Sharing discriminator weights across models:** Each per-model HiFi-GAN learns the specific audio characteristics of that model's training dataset. Discriminator weights are model-specific.
- **Storing checkpoints as separate files:** User decision: checkpoints stored inside the .distillgan file itself. No external checkpoint files.
- **Keeping Griffin-Lim as fallback:** User decision: Griffin-Lim is fully removed. Neural vocoder is the only reconstruction method.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MPD/MSD discriminators | Custom discriminator architecture | Standard HiFi-GAN discriminator design (periods [2,3,5,7,11], 3-scale MSD) | Well-validated in HiFi-GAN paper; these specific periods capture audio periodicity patterns effectively |
| Feature matching loss | Custom perceptual loss | L1 distance between discriminator intermediate features (standard HiFi-GAN feature_loss) | Paper-proven to improve training stability and output quality |
| Mel spectrogram loss | Custom reconstruction metric | L1 distance between mel spectrograms of real and generated audio, weight 45 | Core stabilizer for GAN training; weight 45 is paper-validated |
| Learning rate scheduling | Custom scheduler | ExponentialLR with gamma=0.999 (per HiFi-GAN) | Proven to work with GAN training dynamics; gradual decay prevents mode collapse |
| GAN loss formulation | Custom adversarial loss | Least-squares GAN loss (standard HiFi-GAN choice) | More stable than vanilla GAN loss; no log(0) issues |
| Audio augmentation | New pipeline | Existing `AugmentationPipeline` (speed perturbation, noise, volume) | Already tested and proven in VAE training; reuse for discriminator robustness |

**Key insight:** HiFi-GAN's loss function design is its main innovation -- the specific combination of mel loss (weight 45), feature matching, and adversarial loss from MPD+MSD is what produces high-fidelity audio. Don't change the loss weights or discriminator architecture without strong evidence.

## Common Pitfalls

### Pitfall 1: Discriminator Overfitting on Small Datasets
**What goes wrong:** With only 5-50 training files, the discriminator memorizes the training data and produces perfect real/fake classifications, causing the generator loss to plateau or diverge. Generated audio sounds metallic or buzzy.
**Why it happens:** Small datasets don't provide enough variety for the discriminator to learn general audio quality features.
**How to avoid:**
- Apply data augmentation to discriminator inputs (speed perturbation, noise injection, volume variation using existing AugmentationPipeline)
- Use lower learning rate for discriminator (0.5x generator LR)
- Monitor discriminator loss -- if it hits 0 while generator loss is still high, the discriminator is overfitting
- Consider starting from a pre-trained universal vocoder checkpoint (transfer learning) rather than random init
**Warning signs:** Discriminator loss drops to near-zero within first few epochs while generator loss remains high.

### Pitfall 2: Mode Collapse in Generator
**What goes wrong:** Generator produces identical or near-identical waveforms for different mel inputs. The output is a single buzzing tone regardless of input.
**Why it happens:** Generator finds a single output that satisfies the discriminator and mel loss simultaneously; common when mel loss weight is too low relative to adversarial loss.
**How to avoid:**
- Keep mel spectrogram loss weight at 45 (paper-validated)
- Use feature matching loss (L1 on discriminator intermediate features) to provide diverse gradient signals
- Monitor mel loss separately -- if it's not decreasing while GAN loss improves, mode collapse is occurring
**Warning signs:** Mel loss plateaus while GAN loss improves; generated audio sounds identical across different inputs.

### Pitfall 3: Generator Upsample Rate Mismatch
**What goes wrong:** The generator produces waveforms with incorrect length relative to the input mel frames, causing aliasing, clicks, or truncation.
**Why it happens:** HiFi-GAN V2 defaults are for 22.05kHz with hop_length=256 (product of upsample rates [8,8,2,2] = 256). This project uses 48kHz with hop_length=512.
**How to avoid:** Adapt the upsample rates so their product equals the hop_length (512). Use [8,8,4,2] or [8,4,4,4] to achieve product=512. This is the single most critical configuration change.
**Warning signs:** Output audio length doesn't match expected length; spectrograms show aliasing at Nyquist frequency.

### Pitfall 4: Checkpoint-in-Model File Size Explosion
**What goes wrong:** Storing full training state (generator + discriminator + 2 optimizers + 2 schedulers) inside the .distillgan file makes it huge during training.
**Why it happens:** Discriminator weights (~5-10MB) and optimizer states are large. Combined with the VAE model state, the file could grow 3-5x.
**How to avoid:**
- Only store the checkpoint dict during active training; remove checkpoint key after training completes (final save only has generator weights)
- Generator V2 is 0.92M params (~3.7MB as float32); this is the only persistent addition to file size
- Discriminator + optimizer states are ephemeral -- only needed for resume, removed from final model
**Warning signs:** .distillgan file exceeds 50MB during training (normal final size should be under 20MB).

### Pitfall 5: Griffin-Lim Removal Breaking MelAdapter
**What goes wrong:** After removing Griffin-Lim from AudioSpectrogram, the MelAdapter (used by BigVGAN) breaks because it relies on Griffin-Lim for mel format conversion.
**Why it happens:** MelAdapter.convert() uses AudioSpectrogram.mel_to_waveform() (which calls Griffin-Lim) as step 1 of its VAE-mel-to-BigVGAN-mel conversion.
**How to avoid:** When removing Griffin-Lim, simultaneously update MelAdapter to use a different conversion strategy. Options:
  1. Use the per-model HiFi-GAN as the intermediate waveform reconstructor instead of Griffin-Lim
  2. Replace with a direct mel-to-mel conversion (transfer matrix between HTK and Slaney filterbanks)
  3. Use a lightweight neural converter (small FC network trained on mel pairs)
  The cleanest approach: when a model has a per-model vocoder, use it directly (no MelAdapter needed). For BigVGAN-only models, keep a minimal mel conversion path without Griffin-Lim.
**Warning signs:** BigVGAN vocoder path crashes after Griffin-Lim removal.

### Pitfall 6: MPS Compatibility Issues
**What goes wrong:** GAN training crashes or produces NaN on Apple Silicon MPS backend.
**Why it happens:** MPS has known issues with certain operations (weight_norm, spectral_norm, some conv2d configurations).
**How to avoid:** Use `use_cuda_kernel=False` (already done for BigVGAN). Test discriminator operations on MPS. Apply the same NaN-detection-and-skip pattern used in VAE training (existing in training/loop.py).
**Warning signs:** NaN loss values, SIGABRT on MPS, silent numerical corruption.

## Code Examples

### HiFi-GAN V2 Generator Configuration (Adapted for 128-band 48kHz)
```python
@dataclass
class HiFiGANConfig:
    """HiFi-GAN V2 config adapted for this project's mel parameters."""
    # Generator
    resblock_type: int = 1  # ResBlock1 (V2 uses this)
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 4, 2])  # product = 512 = hop_length
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 8, 4])
    upsample_initial_channel: int = 128
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    # Audio
    num_mels: int = 128  # Project uses 128 mels (not 80)
    sample_rate: int = 48000  # Project native rate
    hop_size: int = 512  # Must match SpectrogramConfig.hop_length
    # Training defaults (sensible for small datasets)
    learning_rate: float = 0.0002
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    segment_size: int = 24576  # 0.512s at 48kHz (multiple of hop_size*product_upsample)
    batch_size: int = 8  # Smaller for small datasets
    # Discriminator
    mpd_periods: list[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
```

### Loss Functions
```python
def generator_loss(disc_outputs: list[torch.Tensor]) -> tuple[torch.Tensor, list]:
    """Least-squares GAN generator loss."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses

def discriminator_loss(
    disc_real_outputs: list[torch.Tensor],
    disc_generated_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list, list]:
    """Least-squares GAN discriminator loss."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses

def feature_loss(
    fmap_r: list[list[torch.Tensor]],
    fmap_g: list[list[torch.Tensor]],
) -> torch.Tensor:
    """L1 feature matching loss between discriminator intermediate features."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2  # Weight factor from original HiFi-GAN
```

### Vocoder Auto-Selection Logic
```python
def resolve_vocoder(selection, loaded_model, device, tqdm_class=None):
    has_per_model = loaded_model.vocoder_state is not None

    if selection == "auto":
        if has_per_model:
            # Per-model HiFi-GAN always preferred over BigVGAN universal
            vocoder = HiFiGANVocoder(loaded_model.vocoder_state, device=device)
            epochs = loaded_model.vocoder_state.get("training_metadata", {}).get("epochs", 0)
            return vocoder, {
                "name": "per_model_hifigan",
                "selection": "auto",
                "reason": f"per-model vocoder ({epochs} epochs)",
                "warning": f"Trained for {epochs} epochs -- quality may be limited" if epochs < 20 else None,
            }
        # Fall through to BigVGAN
        return get_vocoder("bigvgan", device=device, tqdm_class=tqdm_class), {
            "name": "bigvgan_universal",
            "selection": "auto",
            "reason": "no per-model vocoder",
        }
    # ... explicit selection handling
```

### Vocoder Training Data Pipeline
```python
def create_vocoder_dataloader(
    model_path: Path,
    config: HiFiGANConfig,
    augmentation_pipeline=None,
):
    """Create a DataLoader that yields (mel, waveform) pairs from the model's training audio.

    The mel is computed via the VAE's spectrogram pipeline (log1p, HTK, 48kHz)
    to match what the generator will receive at inference time.
    """
    # Load the model to get its spectrogram config
    loaded = load_model(model_path)
    spectrogram = loaded.spectrogram

    # Load original training audio files
    # (stored as dataset_name in metadata -- resolve from data dir)
    # Create random segments of segment_size samples
    # Compute mel via spectrogram.waveform_to_mel()
    # Return (mel_segment, waveform_segment) pairs
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Griffin-Lim (iterative, ~100ms/chunk) | Neural vocoder (BigVGAN, ~10ms/chunk) | Phase 12-14 (2026-02) | 10x faster, dramatically higher audio quality |
| BigVGAN via mel adapter (Griffin-Lim round-trip) | Direct VAE-mel-to-waveform via per-model HiFi-GAN | Phase 16 (this phase) | Eliminates mel conversion quality loss; specialized to user's audio |
| 80-band mel at 22kHz (HiFi-GAN paper) | 128-band mel at 48kHz (this project) | Adaptation | Higher frequency resolution; professional audio production standard |

**Deprecated/outdated:**
- Griffin-Lim reconstruction: Being fully removed in this phase. Was already superseded by BigVGAN in Phase 12 but kept as intermediate step in MelAdapter.
- VAE mel -> Griffin-Lim -> resample -> BigVGAN mel pipeline: Replaced by direct per-model HiFi-GAN when available.

## Open Questions

1. **Griffin-Lim removal and BigVGAN MelAdapter**
   - What we know: MelAdapter currently depends on Griffin-Lim for the mel format conversion round-trip. Per-model HiFi-GAN doesn't need MelAdapter at all (trained on VAE mels directly).
   - What's unclear: How to handle BigVGAN path after Griffin-Lim removal for models without a per-model vocoder. The MelAdapter needs a non-Griffin-Lim waveform reconstruction method.
   - Recommendation: Keep `mel_to_waveform` method on AudioSpectrogram but change its implementation. Instead of InverseMelScale+GriffinLim, use a simple pseudo-inverse approach or accept that BigVGAN quality through the mel adapter will be slightly different. Alternatively, since the per-model HiFi-GAN can also serve as the intermediate reconstructor for MelAdapter (replace Griffin-Lim with HiFi-GAN for the waveform round-trip). For models without per-model vocoder, BigVGAN needs SOME way to get its mel format. The simplest approach: keep a minimal Griffin-Lim-like path only inside MelAdapter (not exposed as public API) OR use a direct filterbank transfer matrix (no waveform round-trip).

2. **Training Data Access for Vocoder Training**
   - What we know: The .distillgan model stores metadata about the dataset (dataset_name, file_count) but not the actual audio file paths. The user needs to point at the original training data.
   - What's unclear: How to present this in the UI -- the model was trained in a previous session, the original data directory may have moved.
   - Recommendation: UI and CLI both require the user to specify the audio directory for vocoder training. The model's dataset_name metadata serves as a hint for auto-discovery.

3. **Optimal Epoch Count for Small Datasets**
   - What we know: HiFi-GAN paper trains for 2500 epochs on LJSpeech (~24 hours of audio). Our datasets are 5-500 files, each ~5-60 seconds.
   - What's unclear: Convergence behavior with 5 minutes vs 5 hours of training data. No published benchmarks for HiFi-GAN on such small datasets.
   - Recommendation: Default to 200 epochs for conservative preset, 500 for balanced, 1000 for aggressive. User can always stop early based on audio preview quality. The preview samples are critical for "when to stop" decisions.

## Sources

### Primary (HIGH confidence)
- [HiFi-GAN GitHub Repository](https://github.com/jik876/hifi-gan) - Architecture, config_v2.json, training code, loss functions
- [HiFi-GAN Paper (arXiv:2010.05646)](https://arxiv.org/abs/2010.05646) - V2 architecture specification, discriminator design, loss formulations
- Project codebase analysis - All existing patterns verified by reading source files directly

### Secondary (MEDIUM confidence)
- [Enhancing GAN-Based Vocoders with Contrastive Learning Under Data-limited Condition](https://arxiv.org/html/2309.09088) - Discriminator overfitting prevention on small datasets
- [SpeechBrain HiFi-GAN Documentation](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.HifiGAN.html) - Implementation reference for loss modules

### Tertiary (LOW confidence)
- HiFi-GAN convergence behavior on 5-50 file datasets is unvalidated (noted in STATE.md as known concern)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies; all libraries already in project
- Architecture: HIGH - HiFi-GAN V2 is well-documented with reference implementations; adaptation for 128-band 48kHz is straightforward
- Pitfalls: HIGH - GAN training pitfalls are well-known; small-dataset overfitting is the main risk, mitigated by augmentation and monitoring
- Griffin-Lim removal: MEDIUM - The interaction between MelAdapter and Griffin-Lim removal needs careful handling; the per-model HiFi-GAN path is clean but BigVGAN fallback needs attention

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable domain -- HiFi-GAN architecture is not changing)
