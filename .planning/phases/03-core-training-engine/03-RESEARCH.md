# Phase 3: Core Training Engine - Research

**Researched:** 2026-02-12
**Domain:** VAE training for audio mel-spectrograms, overfitting prevention, checkpoint management, training monitoring
**Confidence:** HIGH

## Summary

Phase 3 builds the core training engine: a convolutional VAE that trains on mel-spectrogram representations of audio files, with a full training loop featuring progress monitoring, audio preview generation, checkpoint saving/resume, and layered overfitting prevention. The existing Phase 2 infrastructure provides audio I/O (soundfile), preprocessing (resampling to 48kHz, normalization), augmentation (pitch shift, speed, noise, volume), and a Dataset class that stores metadata without loading waveforms. Phase 3 builds on this by adding: (1) a mel-spectrogram representation layer, (2) a convolutional VAE model, (3) a training loop with metrics collection, (4) audio preview generation via mel inversion, (5) checkpoint save/resume, and (6) overfitting detection and prevention.

All critical APIs have been verified working in the current environment: `torchaudio.transforms.MelSpectrogram`, `GriffinLim`, `InverseMelScale`, `torch.optim.AdamW`, `torch.optim.lr_scheduler.CosineAnnealingLR`, `torch.optim.lr_scheduler.ReduceLROnPlateau`, `torch.save`/`torch.load` for checkpoints, and `soundfile.write` for audio preview export. The full pipeline (waveform -> mel spectrogram -> normalize -> denormalize -> InverseMelScale -> GriffinLim -> WAV) has been tested end-to-end. Conv2d encoder/decoder architecture with 4 stride-2 layers works on MPS with ~3.1M parameters total. TensorBoard is NOT installed and should NOT be added as a dependency; instead, use a lightweight dataclass-based metrics collector that the Gradio UI (Phase 8) can consume directly.

**Primary recommendation:** Build a convolutional VAE operating on log-mel spectrograms (128 mels, n_fft=2048, hop_length=512) with 64-dimensional latent space, using KL annealing to prevent posterior collapse, AdamW optimizer with cosine annealing, and a threading.Event-based training runner for cancellation support. Audio previews use InverseMelScale + GriffinLim for mel-to-waveform conversion. Metrics are collected via callback pattern into dataclasses, not TensorBoard.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Training progress feedback:**
- Live dashboard (not scrolling logs) showing real-time metrics
- Detailed metrics: loss curves (train + val), KL divergence, reconstruction loss, learning rate, epoch, step, ETA, GPU/device memory usage
- Hybrid update frequency: step-level updates for training loss, epoch-level updates for validation metrics

**Audio preview behavior:**
- Generate 1 audio preview every N epochs (configurable interval)
- Manual playback only -- preview appears with play button, no auto-play
- Keep all previews visible in a scrollable list with epoch labels -- user can hear the model improve over time

**Overfitting controls:**
- Layered control system: fully automatic defaults, 2-3 presets (Conservative/Balanced/Aggressive), and advanced toggles (dropout, weight decay, augmentation strength) for power users
- On overfitting detection (val loss diverging from train loss): warn visually on dashboard but continue training -- user decides when to stop
- Auto-adapt regularization strategy based on dataset size -- smaller datasets (5-50 files) get stronger regularization, more augmentation, fewer default epochs
- Automatic validation split based on dataset size (e.g., 80/20 for larger sets, adaptive for tiny sets)

**Checkpoint & resume UX:**
- Automatic checkpoint saving at regular intervals

### Claude's Discretion

- Quality indicator approach (intuitive score vs raw metrics only)
- Checkpoint retention count (balancing disk space vs rollback flexibility)
- Cancel behavior (immediate checkpoint save vs finish current epoch)
- Resume flow (summary screen vs seamless auto-continue)
- Manual checkpoint save button (include or omit)
- Exact preview interval default (every N epochs)
- Specific preset parameter values for overfitting presets
- Validation split ratios for different dataset sizes

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10.0 (installed) | Model definition, training loop, optimizers, schedulers | Core framework, already a dependency |
| torchaudio | 2.10.0 (installed) | MelSpectrogram, InverseMelScale, GriffinLim transforms | Verified working; GPU-accelerable; standard for audio ML |
| soundfile | >=0.13.0 (installed) | Audio preview WAV export during training | Already used for all audio I/O in Phase 2 |
| numpy | >=1.26 (installed) | Tensor/array bridge for audio export | Already a dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| threading (stdlib) | N/A | Background training thread with cancellation | Training runner needs to run in background for UI responsiveness |
| dataclasses (stdlib) | N/A | Training metrics, checkpoint metadata, config structures | Type-safe structured data throughout training engine |
| json (stdlib) | N/A | Metrics history serialization to disk | Persisting training history alongside checkpoints |
| time (stdlib) | N/A | ETA computation, step timing | Training progress timing |
| logging (stdlib) | N/A | Training event logging | Matches project-wide logging pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom metrics collector | TensorBoard (torch.utils.tensorboard) | TensorBoard not installed; adds heavy dependency; Gradio UI in Phase 8 consumes metrics directly -- TensorBoard would be redundant middleware |
| Custom training loop | PyTorch Lightning | Lightning adds significant complexity and opinionated structure; project is small enough that a clean custom loop is simpler and more controllable |
| InverseMelScale + GriffinLim | HiFi-GAN vocoder | HiFi-GAN produces higher quality but requires its own pre-trained model; Griffin-Lim is sufficient for training previews and has zero additional dependencies |
| threading.Event | asyncio | Training is CPU/GPU-bound, not I/O-bound; threading is simpler for this use case; Gradio uses threads internally |

### NOT Adding as Dependencies
- **tensorboard**: Not installed, not needed. Metrics go directly to dataclass structures that Gradio can consume.
- **pytorch-lightning**: Too opinionated for this project size. Custom training loop gives full control over checkpoint format, metrics collection, and cancellation.
- **wandb/mlflow**: External services, overkill for local-only application.

**Installation:**
```bash
# No new dependencies needed -- all from existing stack
```

## Architecture Patterns

### Recommended Project Structure
```
src/small_dataset_audio/
├── models/
│   ├── __init__.py              # Public API re-exports
│   ├── vae.py                   # ConvVAE model (encoder + decoder + reparameterize)
│   └── losses.py                # VAE loss (reconstruction + KL with annealing)
├── training/
│   ├── __init__.py              # Public API re-exports
│   ├── config.py                # TrainingConfig, OverfittingPreset, RegularizationConfig
│   ├── loop.py                  # Training loop (single epoch, full training)
│   ├── runner.py                # TrainingRunner (thread management, cancel, resume)
│   ├── metrics.py               # TrainingMetrics, MetricsHistory, ETA computation
│   ├── checkpoint.py            # Save/load/manage checkpoints
│   ├── preview.py               # Generate audio previews from VAE decoder
│   └── dataset.py               # PyTorch Dataset wrapping Phase 2 Dataset for training
├── audio/
│   ├── spectrogram.py           # Mel spectrogram computation and inversion (new)
│   └── (existing io.py, augmentation.py, preprocessing.py, etc.)
```

### Pattern 1: Mel-Spectrogram Representation Layer
**What:** All audio-to-spectrogram and spectrogram-to-audio conversions go through a dedicated module that encapsulates torchaudio transform parameters.
**When to use:** Every time audio enters or leaves the VAE.
**Why:** Centralizes the mel parameters (n_fft, n_mels, hop_length, sample_rate) so they are consistent everywhere. The VAE never touches raw waveforms directly.

```python
# Source: torchaudio.transforms API (verified in environment)
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class SpectrogramConfig:
    """Mel spectrogram parameters -- must be consistent across training and inference."""
    sample_rate: int = 48_000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None  # None = sample_rate / 2
    power: float = 2.0

class AudioSpectrogram:
    """Converts between waveforms and normalized log-mel spectrograms."""

    def __init__(self, config: SpectrogramConfig | None = None):
        from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim

        self.config = config or SpectrogramConfig()
        c = self.config

        self.mel_transform = MelSpectrogram(
            sample_rate=c.sample_rate,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            n_mels=c.n_mels,
            f_min=c.f_min,
            f_max=c.f_max,
            power=c.power,
        )
        self.inverse_mel = InverseMelScale(
            n_stft=c.n_fft // 2 + 1,
            n_mels=c.n_mels,
            sample_rate=c.sample_rate,
            f_min=c.f_min,
            f_max=c.f_max,
        )
        self.griffin_lim = GriffinLim(
            n_fft=c.n_fft,
            n_iter=64,
            hop_length=c.hop_length,
            power=c.power,
        )

    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform [B, 1, samples] to normalized log-mel [B, 1, n_mels, time]."""
        mel = self.mel_transform(waveform.squeeze(1))  # [B, n_mels, time]
        mel_log = torch.log1p(mel)  # log(1+x), handles zeros
        return mel_log.unsqueeze(1)  # [B, 1, n_mels, time]

    def mel_to_waveform(self, mel_log: torch.Tensor) -> torch.Tensor:
        """Convert log-mel [B, 1, n_mels, time] back to waveform [B, 1, samples]."""
        mel = torch.expm1(mel_log.squeeze(1).clamp(min=0))  # inverse of log1p
        linear_spec = self.inverse_mel(mel.cpu())  # InverseMelScale is CPU-only
        waveform = self.griffin_lim(linear_spec)
        return waveform.unsqueeze(1)  # [B, 1, samples]
```

### Pattern 2: Convolutional VAE with Pad-Then-Crop
**What:** Encoder pads mel spectrogram time dimension to nearest multiple of 16 (for 4 stride-2 layers), decoder crops output back to original size.
**When to use:** The VAE model forward pass.
**Why:** Strided convolutions require divisible dimensions. Padding/cropping is cleaner than asymmetric output_padding. Verified working: input [B, 1, 128, 94] -> latent [B, 64] -> output [B, 1, 128, 94] with exact shape match.

```python
# Source: Verified in project environment on MPS
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # 4 conv blocks: each halves spatial dims via stride 2
        # Input: [B, 1, 128, padded_time] -> [B, 256, 8, padded_time/16]
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        # For 1s@48kHz: padded 96 -> 96/16=6 time frames, 128/16=8 mel frames
        self.flatten_dim = 256 * 8 * 6
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pad time dimension to multiple of 16
        if x.shape[-1] % 16 != 0:
            pad_size = 16 - (x.shape[-1] % 16)
            x = F.pad(x, (0, pad_size))
        h = self.convs(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)
```

### Pattern 3: Training Metrics via Callback Dataclasses
**What:** Training loop emits metrics via a callback function. Metrics are typed dataclasses. A MetricsHistory collects them for plotting.
**When to use:** Every training step and epoch boundary.
**Why:** Decouples training from display. The Gradio UI (Phase 8) subscribes to the same callback. For Phase 3, metrics are collected in memory and optionally serialized to JSON.

```python
from dataclasses import dataclass, field
from typing import Callable
import time

@dataclass
class StepMetrics:
    """Emitted every training step."""
    epoch: int
    step: int
    total_steps: int
    train_loss: float
    recon_loss: float
    kl_loss: float
    kl_weight: float
    learning_rate: float
    step_time_s: float

@dataclass
class EpochMetrics:
    """Emitted every epoch."""
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_recon_loss: float
    val_kl_loss: float
    kl_divergence: float       # raw KL for monitoring posterior collapse
    overfitting_gap: float     # (val_loss - train_loss) / train_loss
    eta_seconds: float
    elapsed_seconds: float

@dataclass
class PreviewEvent:
    """Emitted when an audio preview is generated."""
    epoch: int
    audio_path: str            # Path to saved WAV file
    sample_rate: int

# Callback type: receives any of these event types
MetricsCallback = Callable[[StepMetrics | EpochMetrics | PreviewEvent], None]
```

### Pattern 4: Training Runner with Thread + Event Cancellation
**What:** Training runs in a background thread. A `threading.Event` provides cancellation. The runner manages the thread lifecycle.
**When to use:** All training invocations.
**Why:** Verified working: `threading.Event` + background thread provides clean cancellation, progress callbacks to the main thread, and checkpoint saving on cancel. This pattern works well with Gradio's threading model.

```python
import threading
from dataclasses import dataclass

@dataclass
class TrainingState:
    """Persisted state for pause/resume."""
    epoch: int
    step: int
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict
    metrics_history: list
    kl_weight: float

class TrainingRunner:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._state: TrainingState | None = None

    def start(self, config, dataset, callback):
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run_training,
            args=(config, dataset, callback, self._cancel_event),
            daemon=True,
        )
        self._thread.start()

    def cancel(self):
        """Signal cancellation. Training saves checkpoint before stopping."""
        self._cancel_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
```

### Pattern 5: Checkpoint Format
**What:** Save model, optimizer, scheduler, epoch, metrics, and config as a single `.pt` file. Store audio previews alongside.
**When to use:** Regular intervals + on cancellation.
**Why:** Standard PyTorch checkpoint pattern. Verified: `torch.save`/`torch.load` work correctly with `weights_only=True` for state dicts.

```python
# Source: PyTorch official tutorial (docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
checkpoint = {
    'epoch': epoch,
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'kl_weight': kl_weight,
    'config': training_config.__dict__,
    'spectrogram_config': spec_config.__dict__,
    'metrics_history': metrics_history,
}
torch.save(checkpoint, checkpoint_path)
```

### Anti-Patterns to Avoid
- **Training on raw waveforms with a VAE:** Waveforms have too much temporal detail for a vanilla VAE to learn effectively. Always convert to mel spectrograms first -- the frequency-domain representation is lower-dimensional and captures perceptually relevant features.
- **Using MSE loss directly on mel spectrograms without log scaling:** Raw mel magnitudes span many orders of magnitude. Use log1p(mel) normalization to compress the dynamic range, then reconstruct in log space. Denormalize with expm1() for audio conversion.
- **Ignoring the KL term initially (beta=0 forever):** The model becomes a standard autoencoder with no regularized latent space. Use KL annealing to gradually introduce the KL term.
- **Saving only model weights in checkpoints:** Without optimizer and scheduler state, resumed training loses momentum, adaptive learning rates, and warmup progress. Save everything.
- **Running InverseMelScale on GPU:** Verified that `InverseMelScale` uses `torch.linalg.lstsq` which can have issues on MPS. Always run mel inversion on CPU.
- **Creating transforms inside the training loop:** MelSpectrogram, GriffinLim, etc. allocate internal buffers. Create once, reuse across batches. Matches Phase 2 pattern (pre-create reusable transforms).
- **Using the same augmentation pipeline for train and validation:** Validation data must NOT be augmented. Only training data gets augmentation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mel spectrogram computation | Custom STFT + mel filterbank | `torchaudio.transforms.MelSpectrogram` | GPU-accelerable, handles windowing, padding, filterbank construction |
| Spectrogram-to-audio inversion | Custom phase estimation | `torchaudio.transforms.InverseMelScale` + `GriffinLim` | InverseMelScale uses least-squares optimization; GriffinLim uses momentum-based phase recovery with convergence guarantees |
| Learning rate scheduling | Custom LR decay logic | `torch.optim.lr_scheduler.CosineAnnealingLR` | Handles edge cases (warmup interaction, state serialization, step counting) |
| Weight decay | Manual L2 penalty in loss | `torch.optim.AdamW` (decoupled weight decay) | AdamW correctly decouples weight decay from gradient update; manual L2 penalty interacts badly with Adam's adaptive learning rates |
| Checkpoint serialization | Custom pickle/JSON | `torch.save` / `torch.load` | Handles tensor serialization, device mapping, memory-mapped loading |
| Batch construction/shuffling | Manual epoch iteration | `torch.utils.data.DataLoader` | Handles shuffling, batching, pin_memory, num_workers, and collation |
| KL divergence computation | Manual formula | Closed-form: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))` | Standard formula, but must sum over latent dims and mean over batch -- easy to get wrong |

**Key insight:** The training engine is mostly orchestration of well-tested PyTorch primitives. The novel work is (1) the VAE architecture sized for small audio datasets, (2) the KL annealing schedule tuned for posterior collapse prevention, and (3) the overfitting detection/prevention heuristics.

## Common Pitfalls

### Pitfall 1: Posterior Collapse (KL Divergence -> 0)
**What goes wrong:** The decoder ignores the latent code entirely, generating the same output regardless of input. KL divergence drops to near-zero.
**Why it happens:** If the reconstruction loss dominates early training, the encoder learns to set mu=0, logvar=0 (matching the prior exactly) because the decoder can reconstruct well enough without latent information.
**How to avoid:** Use KL annealing: start with kl_weight=0, linearly increase to 1.0 over the first 30-50% of epochs. Additionally, use the "free bits" strategy: per-dimension KL floor of ~0.5 bits ensures each latent dimension encodes some information.
**Warning signs:** KL divergence < 0.5 across all latent dimensions; generated samples all look identical; latent space has no structure.
**Success criterion (from requirements):** KL divergence remains above 0.5.

### Pitfall 2: Reconstruction Loss Dominates (Blurry Output)
**What goes wrong:** Generated mel spectrograms are overly smooth/blurry, producing washed-out audio.
**Why it happens:** MSE loss averages over frequency bins, encouraging the model to predict the mean. With small datasets, the model never learns fine spectral detail.
**How to avoid:** (1) Use spectral loss that weights high frequencies more, or add a multi-scale STFT loss. (2) Consider spectral convergence loss alongside MSE. (3) For Phase 3, MSE on log-mel is acceptable -- quality refinement is Phase 4.
**Warning signs:** All generated previews sound similar and muffled; reconstruction loss plateaus early.

### Pitfall 3: Overfitting on Small Datasets (5-50 files)
**What goes wrong:** Training loss drops steadily but validation loss increases after a few epochs. Generated audio is just memorized snippets.
**Why it happens:** With 5-50 files, even after augmentation (10x expansion = 50-500 samples), the model has enough capacity to memorize the training set.
**How to avoid:** (1) Smaller model capacity for smaller datasets. (2) Higher dropout (0.3-0.5 for <50 files). (3) Stronger weight decay (0.05-0.1). (4) More aggressive augmentation (higher expansion ratio). (5) Fewer epochs (50-100 vs 200-500). (6) Monitor val/train loss ratio -- warn when gap exceeds 20%.
**Warning signs:** Val loss increasing while train loss decreasing; overfitting gap > 0.2.

### Pitfall 4: MPS-Specific Numerical Issues
**What goes wrong:** Training produces NaN losses or wildly different results on MPS vs CUDA/CPU.
**Why it happens:** MPS has known numerical precision differences for certain operations (batch norm, reduction operations).
**How to avoid:** (1) Use float32 throughout (no float16 on MPS). (2) Gradient clipping (max_norm=1.0). (3) Check for NaN after each step and skip the update if detected. (4) Test model output determinism across devices.
**Warning signs:** NaN in loss; divergent training curves on MPS vs CPU.

### Pitfall 5: Inconsistent Mel Spectrogram Parameters
**What goes wrong:** Training uses different mel parameters than preview generation or checkpoint loading, producing garbled audio.
**Why it happens:** Mel parameters (n_fft, n_mels, hop_length) are scattered across multiple functions.
**How to avoid:** Store `SpectrogramConfig` in every checkpoint. Validate on resume that checkpoint config matches current config. Centralize all mel operations through `AudioSpectrogram` class.
**Warning signs:** Audio previews sound completely wrong; loaded checkpoints produce garbage.

### Pitfall 6: Variable-Length Audio Without Chunking
**What goes wrong:** Audio files of different lengths cannot be batched together. DataLoader crashes or produces inconsistent shapes.
**Why it happens:** Convolutional VAE requires fixed spatial dimensions. Different audio durations produce different mel spectrogram widths.
**How to avoid:** Chunk all audio to a fixed duration (1 second recommended) during preprocessing. Shorter files are zero-padded. Longer files produce multiple chunks. Use a custom collate function if needed.
**Warning signs:** Shape mismatch errors in DataLoader; batch dimension inconsistencies.

### Pitfall 7: Validation Split Too Small for Tiny Datasets
**What goes wrong:** With 5 files, a 80/20 split gives 1 validation file. Validation loss is extremely noisy and unreliable.
**Why it happens:** Standard split ratios assume hundreds+ of samples.
**How to avoid:** Adaptive split: <10 files: use leave-one-out or 50/50; 10-50 files: 70/30; 50-200 files: 80/20; 200+: 90/10. Split at the file level (not chunk level) to prevent data leakage.
**Warning signs:** Validation loss oscillates wildly epoch to epoch; single outlier file dominates val loss.

### Pitfall 8: Checkpoint File Proliferation
**What goes wrong:** Saving every epoch for a 500-epoch run with ~100MB checkpoints fills disk (50GB+).
**Why it happens:** No retention policy.
**How to avoid:** Keep only the N most recent checkpoints plus the best (lowest val_loss) checkpoint. Recommended: retain 3 most recent + 1 best = 4 checkpoints max. Delete old ones automatically.
**Warning signs:** Disk usage grows linearly with training; user runs out of space.

## Code Examples

Verified patterns from official sources and live environment testing:

### Full VAE Loss Function with KL Annealing
```python
# Source: VAE loss literature + vae-annealing library pattern
import torch
import torch.nn.functional as F

def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    free_bits: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss with free bits and KL annealing weight.

    Args:
        recon: Reconstructed mel spectrogram [B, 1, n_mels, time].
        target: Original mel spectrogram [B, 1, n_mels, time].
        mu: Latent mean [B, latent_dim].
        logvar: Latent log-variance [B, latent_dim].
        kl_weight: Annealing weight for KL term (0.0 to 1.0).
        free_bits: Minimum KL per dimension (prevents posterior collapse).

    Returns:
        (total_loss, recon_loss, kl_loss) tuple.
    """
    # Reconstruction loss: MSE on log-mel spectrograms
    recon_loss = F.mse_loss(recon, target, reduction='mean')

    # KL divergence per dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Free bits: clamp each dimension to minimum of free_bits
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Sum over latent dims, mean over batch
    kl_loss = kl_per_dim.sum(dim=1).mean()

    # Total loss with annealing weight
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss
```

### KL Annealing Schedule
```python
# Source: vae-annealing library pattern (github.com/hubertrybka/vae-annealing)
def get_kl_weight(epoch: int, total_epochs: int, warmup_fraction: float = 0.3) -> float:
    """Linear KL annealing: weight goes from 0 to 1 over warmup_fraction of training.

    Args:
        epoch: Current epoch (0-indexed).
        total_epochs: Total training epochs.
        warmup_fraction: Fraction of training to anneal over.

    Returns:
        KL weight between 0.0 and 1.0.
    """
    warmup_epochs = int(total_epochs * warmup_fraction)
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)
```

### PyTorch Dataset for Training (Wrapping Phase 2 Dataset)
```python
# Source: torch.utils.data docs + project Phase 2 Dataset class
import torch
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path

class AudioTrainingDataset(TorchDataset):
    """PyTorch Dataset that loads preprocessed mel spectrograms for VAE training.

    Wraps the Phase 2 Dataset class. Loads audio from cached .pt files
    or on-the-fly from source files, converts to mel spectrogram chunks.
    """

    def __init__(
        self,
        file_paths: list[Path],
        spectrogram_config: "SpectrogramConfig",
        chunk_samples: int = 48000,  # 1 second at 48kHz
        augmentation_pipeline: "AugmentationPipeline | None" = None,
    ):
        self.file_paths = file_paths
        self.spec_config = spectrogram_config
        self.chunk_samples = chunk_samples
        self.augmentation = augmentation_pipeline
        # Build index: (file_idx, chunk_start_sample) pairs
        self._chunks: list[tuple[int, int]] = []
        self._build_chunk_index()

    def _build_chunk_index(self):
        """Pre-scan files to determine chunk boundaries."""
        from small_dataset_audio.audio.io import get_metadata
        for file_idx, path in enumerate(self.file_paths):
            meta = get_metadata(path)
            total_samples = int(meta.duration_seconds * self.spec_config.sample_rate)
            for start in range(0, total_samples, self.chunk_samples):
                self._chunks.append((file_idx, start))

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, chunk_start = self._chunks[idx]
        # Load audio, extract chunk, convert to mel
        # (implementation details in actual code)
        ...
```

### Audio Preview Generation
```python
# Source: Verified pipeline in project environment
import soundfile as sf
import numpy as np
from pathlib import Path

def generate_preview(
    model: "ConvVAE",
    spectrogram: "AudioSpectrogram",
    output_dir: Path,
    epoch: int,
    device: torch.device,
    num_samples: int = 1,
) -> list[Path]:
    """Generate audio previews from random latent vectors.

    Returns list of saved WAV file paths.
    """
    model.eval()
    previews = []

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)
        mel_recon = model.decode(z)

        # Move to CPU for mel inversion (InverseMelScale is CPU-only)
        waveforms = spectrogram.mel_to_waveform(mel_recon.cpu())

        for i, waveform in enumerate(waveforms):
            audio_np = waveform.squeeze().numpy()
            # Peak normalize
            peak = np.abs(audio_np).max()
            if peak > 0:
                audio_np = audio_np / peak

            path = output_dir / f"preview_epoch{epoch:04d}_{i:02d}.wav"
            sf.write(str(path), audio_np, 48000, subtype='PCM_16')
            previews.append(path)

    model.train()
    return previews
```

### Checkpoint Save and Load
```python
# Source: PyTorch official tutorial (docs.pytorch.org)
import torch
from pathlib import Path

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    kl_weight: float,
    config: dict,
    metrics_history: list,
) -> None:
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'kl_weight': kl_weight,
        'config': config,
        'metrics_history': metrics_history,
    }, path)

def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict:
    """Load training checkpoint and restore state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.train()
    return checkpoint
```

### Overfitting Detection
```python
def compute_overfitting_gap(train_loss: float, val_loss: float) -> float:
    """Compute relative gap between validation and training loss.

    Returns (val_loss - train_loss) / train_loss.
    Positive values indicate overfitting. >0.2 (20%) triggers warning.
    """
    if train_loss <= 0:
        return 0.0
    return (val_loss - train_loss) / train_loss

def get_adaptive_split(file_count: int) -> float:
    """Return validation fraction based on dataset size.

    Smaller datasets get larger validation fractions to ensure
    meaningful validation signal.
    """
    if file_count < 10:
        return 0.5      # 50/50 for tiny datasets
    elif file_count < 50:
        return 0.3      # 70/30 for small datasets
    elif file_count < 200:
        return 0.2      # 80/20 for medium datasets
    else:
        return 0.1      # 90/10 for larger datasets
```

## Discretion Recommendations

For items marked as Claude's discretion in CONTEXT.md:

### Quality Indicator: Include an Intuitive Score
**Recommendation:** Show a simple "Training Quality" indicator alongside raw metrics. Compute as: `quality = 1.0 - overfitting_gap` clamped to [0, 1], displayed as a colored bar (green = good, yellow = caution, red = overfitting). This gives beginners a single number to watch while power users still have full metrics.
**Rationale:** The user specifically wants a "proper ML training monitor" feel. Raw metrics alone are intimidating for beginners. A quality bar bridges the gap between the layered control system's "automatic defaults" tier and the "advanced toggles" tier.

### Checkpoint Retention: Keep 3 + 1 Best
**Recommendation:** Retain the 3 most recent checkpoints plus 1 best-val-loss checkpoint. Delete older ones automatically. This limits disk usage to ~4x checkpoint size (~400MB for a 3M parameter model) while providing both rollback flexibility and the ability to resume from the best point.
**Rationale:** 3 recent checkpoints cover ~3 checkpoint intervals of rollback. The best checkpoint is critical for recovery if training overshoots. More than 4 wastes disk without practical benefit.

### Cancel Behavior: Immediate Checkpoint Save
**Recommendation:** On cancel signal, save a checkpoint immediately at the current step (mid-epoch), then stop. Do NOT wait to finish the current epoch -- users expect cancel to be responsive. The checkpoint includes the partial epoch progress so resume can continue from the exact step.
**Rationale:** Waiting to finish an epoch could mean minutes of delay on large datasets. Immediate save is more responsive. The checkpoint format already includes step-level granularity, so mid-epoch resume is clean.

### Resume Flow: Summary Screen Then Continue
**Recommendation:** When resuming from a checkpoint, display a brief summary (epoch, loss, metrics snapshot, time elapsed) and require the user to click "Continue Training" rather than auto-resuming. This gives the user a moment to review the state and optionally adjust parameters before continuing.
**Rationale:** Auto-resume could surprise users who loaded a checkpoint just to inspect it. The summary provides orientation after a break.

### Manual Checkpoint Save: Include It
**Recommendation:** Include a "Save Checkpoint Now" button in the training dashboard. It triggers an immediate checkpoint save without interrupting training.
**Rationale:** Users may want to bookmark a particular training state before experimenting with parameter changes. This is low implementation cost (just trigger the existing save logic) and high user value.

### Preview Interval: Default Every 5 Epochs
**Recommendation:** Default preview interval of 5 epochs. For short training runs (<50 epochs), use every 2 epochs. Configurable via the training config.
**Rationale:** 5 epochs is frequent enough to track quality evolution without overwhelming disk or slowing training (Griffin-Lim inversion takes ~1 second per preview).

### Overfitting Preset Values
**Recommendation:**

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| Dataset size target | 5-50 files | 50-200 files | 200-500 files |
| Dropout | 0.4 | 0.2 | 0.1 |
| Weight decay | 0.05 | 0.01 | 0.001 |
| Augmentation expansion | 15x | 10x | 5x |
| Max epochs | 100 | 200 | 500 |
| Learning rate | 5e-4 | 1e-3 | 2e-3 |
| KL warmup fraction | 0.5 | 0.3 | 0.2 |
| Gradient clip norm | 0.5 | 1.0 | 5.0 |

**Auto-adapt:** When dataset size is detected, automatically select the matching preset as default. User can override by selecting a different preset or adjusting individual values.

### Validation Split Ratios
**Recommendation:**

| File Count | Val Fraction | Rationale |
|------------|-------------|-----------|
| 5-9 | 0.5 (50%) | Every file matters; need meaningful val signal |
| 10-49 | 0.3 (30%) | Balance training data with val reliability |
| 50-199 | 0.2 (20%) | Standard split, sufficient val samples |
| 200+ | 0.1 (10%) | Plenty of data for both |

Split at the FILE level, not the chunk level, to prevent data leakage (chunks from the same file appearing in both train and val).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Raw waveform autoencoders | Mel-spectrogram VAEs | ~2020 | Much lower dimensionality; captures perceptual features; enables Griffin-Lim or neural vocoder inversion |
| Standard VAE (beta=1) | KL annealing + free bits | ~2019 | Prevents posterior collapse; required for meaningful latent spaces |
| Manual L2 regularization | AdamW decoupled weight decay | PyTorch 1.x | Correct weight decay implementation; AdamW is now the default recommendation |
| Fixed learning rate | Cosine annealing with warmup | ~2020 | Better convergence, especially for small datasets where the loss landscape is noisy |
| TensorBoard for monitoring | Direct metrics collection (dataclass/callback) | Evolving | For local apps with Gradio UI, direct collection avoids middleware; TensorBoard remains standard for research |
| Global augmentation toggle | Independent per-augmentation probability | ~2021 | Prevents over-augmentation; allows fine-tuning augmentation mix |
| adam optimizer | AdamW (decoupled weight decay) | PyTorch 1.8+ | Standard for modern training; proper weight decay without interaction with Adam's moment estimates |

**Deprecated/outdated:**
- `torch.optim.Adam` with manual L2 penalty: Use `AdamW` instead (decoupled weight decay).
- `torchaudio.save()` without torchcodec: Broken in torchaudio 2.10. Use soundfile.write() for saving audio.
- TensorBoard as mandatory training monitor: For local apps, direct metrics collection is simpler.

## Open Questions

1. **Optimal latent dimension for small audio datasets**
   - What we know: MelSpecVAE uses 64 for general audio. Research paper uses 13-40 for speech. Larger dims risk more inactive dimensions with small datasets.
   - What's unclear: Optimal dimension for 5-500 file datasets with diverse audio content (ambient, electronic, experimental, acoustic).
   - Recommendation: Default to 64. Make configurable. Log per-dimension KL to detect inactive dimensions -- if more than 50% of dims have KL < 0.1, suggest reducing latent_dim.

2. **Chunk duration for training**
   - What we know: 1 second (48000 samples) produces 128x94 mel spectrograms. Shorter chunks (0.5s) give more training samples but less temporal context. Longer chunks (2-4s) capture more structure but fewer samples.
   - What's unclear: Best duration for small-dataset training where maximizing sample count matters but temporal coherence also matters.
   - Recommendation: Default to 1 second. This balances sample count (a 30-second file produces 30 chunks) with temporal context. Make configurable for power users.

3. **Griffin-Lim iteration count for previews vs final generation**
   - What we know: 32 iterations is fast but has audible artifacts. 64 is good quality. 1000+ is near-optimal but slow.
   - What's unclear: Whether 64 iterations is good enough for previews that let users assess training quality.
   - Recommendation: Use 64 for previews (fast enough, reasonable quality). Phase 4 can improve with a neural vocoder. Store the mel spectrogram in the checkpoint so previews can be regenerated with better inversion later.

4. **Batch size for small datasets**
   - What we know: Config has max_batch_size=32 from hardware benchmark. But with 5 files and 1s chunks, total training samples may be only ~50-150 (after augmentation = 500-1500 chunks).
   - What's unclear: Whether batch_size=32 is too large for very small datasets (batch noise dominates).
   - Recommendation: Default batch_size = min(32, len(train_dataset) // 4). For tiny datasets, batch_size=8 or even 4 may work better. Make configurable through the preset system.

## Sources

### Primary (HIGH confidence)
- torchaudio 2.10.0 transforms verified in project environment: MelSpectrogram, GriffinLim, InverseMelScale all confirmed working
- [torchaudio MelSpectrogram docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html) -- full API verified
- [torchaudio GriffinLim docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html) -- full API verified
- [torchaudio InverseMelScale docs](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html) -- full API verified
- [PyTorch saving/loading models tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) -- checkpoint format
- [PyTorch AdamW docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) -- optimizer API
- [PyTorch CosineAnnealingLR docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) -- scheduler API
- Full pipeline verified: waveform -> MelSpectrogram -> log1p normalize -> InverseMelScale -> GriffinLim -> soundfile.write -> WAV
- Conv2d encoder/decoder verified on MPS: ~3.1M params, exact shape match with pad-then-crop strategy
- threading.Event cancellation verified working

### Secondary (MEDIUM confidence)
- [Modern PyTorch VAE Tutorial](https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/) -- VAE implementation patterns, loss scaling, softplus parameterization
- [vae-annealing library](https://github.com/hubertrybka/vae-annealing) -- KL annealing implementation (linear, cosine, logistic schedules)
- [MelSpecVAE](https://github.com/moiseshorta/MelSpecVAE) -- Reference convolutional VAE for mel-spectrogram audio synthesis (latent_dim=64, 5-layer conv encoder)
- [Convolutional VAE for Audio Spectrogram Compression (arXiv:2410.02560)](https://arxiv.org/html/2410.02560v1) -- 2-layer conv encoder, 13-dim latent, KL weight=-0.0005, 100 epochs
- [KL vanishing techniques summary](https://github.com/zheng-yanan/techniques-for-kl-vanishing) -- Free bits, KL annealing, lagging encoder strategies
- [Early stopping in PyTorch](https://github.com/Bjarten/early-stopping-pytorch) -- Patience-based early stopping implementation

### Tertiary (LOW confidence)
- Optimal latent dimension for diverse audio datasets: Based on interpolation between speech (13-40 dim) and general audio (64 dim) research; needs empirical validation for this specific use case
- Overfitting preset parameter values: Based on general ML best practices and audio augmentation experience; specific values need tuning during implementation
- Batch size recommendations for tiny datasets: Based on general deep learning intuition; needs empirical validation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified in live environment; no new dependencies needed
- Architecture (VAE model): HIGH -- encoder/decoder verified on MPS with exact shape matching; pad-then-crop strategy tested
- Architecture (training loop): HIGH -- all PyTorch primitives verified (AdamW, CosineAnnealingLR, torch.save/load, DataLoader)
- Architecture (metrics/runner): MEDIUM -- callback pattern tested with threading.Event; Gradio integration deferred to Phase 8
- Mel spectrogram pipeline: HIGH -- full waveform -> mel -> inversion -> audio pipeline verified end-to-end
- Overfitting prevention: MEDIUM -- techniques are well-established (KL annealing, free bits, dropout, weight decay) but preset parameter values need empirical tuning
- Checkpoint management: HIGH -- PyTorch save/load verified; retention policy is straightforward
- Audio preview generation: HIGH -- InverseMelScale + GriffinLim -> soundfile.write verified

**Research date:** 2026-02-12
**Valid until:** 2026-03-12 (stable -- PyTorch 2.10, torchaudio 2.10 are mature releases)
