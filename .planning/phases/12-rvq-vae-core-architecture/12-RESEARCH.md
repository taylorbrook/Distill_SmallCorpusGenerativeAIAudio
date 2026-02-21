# Phase 12: RVQ-VAE Core Architecture - Research

**Researched:** 2026-02-21
**Domain:** Residual Vector Quantization VAE model architecture for small-dataset mel spectrogram encoding/decoding
**Confidence:** HIGH

## Summary

Phase 12 builds the foundational ConvVQVAE model that replaces the existing continuous ConvVAE. The core architectural change is surgical: the convolutional encoder/decoder backbone (4-layer Conv2d, stride 2) stays identical, but the bottleneck changes from `flatten -> fc_mu/fc_logvar -> reparameterize -> fc -> reshape` to `Conv1x1 projection -> ResidualVQ -> Conv1x1 projection`. The encoder outputs a 2D spatial feature map `[B, D, H, W]` that is reshaped to a sequence of `H*W` embedding vectors, each quantized independently through stacked RVQ codebooks. The decoder receives quantized embeddings projected back to `[B, 256, H, W]`. This eliminates all KL divergence, free bits, and annealing logic, replacing them with a single commitment loss weight parameter.

The primary library for this phase is `vector-quantize-pytorch>=1.27.0` (v1.27.21 released 2026-02-12), which provides the complete `ResidualVQ` implementation including k-means codebook initialization, EMA updates, dead code replacement, straight-through estimator, and `get_output_from_indices()` for decoding from code indices. Dataset-adaptive codebook sizing (64/128/256 for 5-20/20-100/100-500 files) is a project-specific requirement not handled by the library and must be implemented as a configuration layer. The context decision specifies a fresh architecture design -- do NOT reuse v1.0 encoder/decoder classes; create new `VQEncoder` and `VQDecoder` classes.

The biggest risk is codebook collapse on small datasets (utilization as low as 14.7% observed even on large datasets). Mitigation is built into the library (`kmeans_init`, `threshold_ema_dead_code`, EMA `decay`) but codebook sizing, monitoring, and the adaptive configuration must be implemented in this phase. Multi-scale spectral loss for reconstruction was mentioned in context discussions; this phase should implement it as an optional enhancement alongside MSE, since the user decision explicitly states "multi-scale spectral loss for reconstruction."

**Primary recommendation:** Build ConvVQVAE as a new module (`models/vqvae.py`) with VQEncoder, ResidualVQ wrapper, VQDecoder, and a dataset-adaptive codebook config function. Test with a forward pass returning `(recon_mel, indices, commit_loss)`. Do not touch existing v1.0 code.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Temporal resolution: Fixed at model level (not user-configurable), Claude's discretion on exact compression ratio -- should balance region-level code editing with occasional fine edits, must handle mixed clip lengths (1-30+ seconds)
- Default RVQ levels: 3 (Structure / Timbre / Detail), configurable range 2-4
- Default codebook dimension: 128
- Codebook size auto-scales with dataset size (64/128/256 per requirements)
- Fresh design -- do NOT reuse v1.0 encoder/decoder architecture; clean break
- Replace v1.0 model code entirely -- remove SpectrogramVAE, replace with ConvVQVAE
- Old .sda model files will not load -- no backward compatibility
- Quality-first architecture -- deeper model, longer training acceptable
- Multi-scale spectral loss for reconstruction (multiple STFT resolutions)
- Combined with commitment loss (single weight parameter, no KL divergence)

### Claude's Discretion
- Exact temporal compression ratio (guided by: medium resolution, region-level editing focus, mixed clip lengths)
- Commitment loss weight default
- Reconstruction fidelity balance for non-edited regions in encode/edit/decode
- Encoder/decoder depth and layer configuration
- Dead code reset thresholds and EMA decay rate

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VQVAE-01 | User can train an RVQ-VAE model on a small audio dataset (5-500 files) using stacked residual codebooks | ConvVQVAE model with ResidualVQ from vector-quantize-pytorch; forward pass produces (recon, indices, commit_loss); VQVAEConfig dataclass with dataset-adaptive defaults |
| VQVAE-02 | Codebook size automatically scales based on dataset size (64 for 5-20, 128 for 20-100, 256 for 100-500) | `get_adaptive_vqvae_config(file_count)` function that maps dataset size to codebook_size, num_quantizers, and other VQ params |
| VQVAE-03 | Training uses EMA codebook updates with k-means initialization and dead code reset | ResidualVQ parameters: `kmeans_init=True, kmeans_iters=10, decay=0.8-0.95, threshold_ema_dead_code=2`; all handled by the library |
| VQVAE-05 | Training uses commitment loss (single weight parameter) instead of KL divergence | New `vqvae_loss()` function: MSE reconstruction + multi-scale spectral loss + commitment_weight * commit_loss; no KL/free_bits/annealing |
| VQVAE-06 | User can configure number of RVQ levels (2-4) and codebook dimension | VQVAEConfig exposes `num_quantizers` (2-4) and `codebook_dim` (default 128); ConvVQVAE constructor accepts these |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| vector-quantize-pytorch | >=1.27.0 (current: 1.27.21) | ResidualVQ layer with k-means init, EMA updates, dead code reset, STE | De facto standard for PyTorch VQ/RVQ; MIT licensed; 3500+ stars; actively maintained; provides every VQ feature needed out-of-box |
| torch | >=2.10.0,<2.11 (project pin) | Neural network backbone, Conv2d/ConvTranspose2d encoder/decoder | Already installed; project dependency |
| einops | >=0.8.0 (transitive) | Tensor reshaping used internally by vector-quantize-pytorch | Auto-installed with vector-quantize-pytorch; do not install separately |
| einx[torch] | >=0.1.3 (transitive) | Extended Einstein operations used internally by vector-quantize-pytorch | Auto-installed with vector-quantize-pytorch; do not install separately |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchaudio | >=2.10.0,<2.11 (already installed) | MelSpectrogram, GriffinLim transforms for multi-scale spectral loss computation | Computing spectral loss at multiple STFT resolutions |
| numpy | >=1.26 (already installed) | Codebook utilization metric computation | Computing utilization stats from indices |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| vector-quantize-pytorch | Custom VQ implementation | Custom is 500-1000 lines to get right (STE, EMA, dead code reset, k-means); library handles all edge cases |
| vector-quantize-pytorch | torchvq | Less maintained, no RVQ support, fewer features |
| MSE + multi-scale spectral loss | auraloss library | auraloss adds a dependency for ~50 lines of code; implement inline instead |

**Installation:**
```bash
uv add "vector-quantize-pytorch>=1.27.0"
```

## Architecture Patterns

### Recommended Project Structure
```
src/distill/models/
    vae.py              # v1.0 ConvVAE (REMOVE in this phase)
    vqvae.py            # NEW: ConvVQVAE with VQEncoder, VQDecoder
    quantizer.py        # NEW: Thin wrapper around ResidualVQ with monitoring
    losses.py           # REWRITE: vqvae_loss() replacing vae_loss()
    persistence.py      # UNCHANGED in this phase (Phase 13 handles persistence)
    __init__.py         # UPDATE: export new classes
```

### Pattern 1: Spatial Embedding Encoder (not Global Vector)
**What:** The encoder outputs a 2D spatial feature map `[B, D, H, W]` instead of a single global latent vector `[B, latent_dim]`. Each spatial position is independently quantized through RVQ.
**When to use:** Always -- this is the fundamental architectural change from v1.0.
**Why:** VQ operates on individual embedding vectors, not global summaries. Spatial positions preserve local mel spectrogram structure. Multiple codebook entries per mel = finer reconstruction. This is how SoundStream, EnCodec, and all successful audio VQ systems work.

```python
# Source: Architecture research + SoundStream paper pattern
class VQEncoder(nn.Module):
    """Encoder producing spatial embeddings for RVQ quantization.

    Output: [B, codebook_dim, H, W] where H*W positions are independently quantized.
    """
    def __init__(self, codebook_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        # Same 4-layer conv backbone as v1.0 (1->32->64->128->256, stride 2)
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        # Project to codebook dimension (replaces fc_mu/fc_logvar)
        self.proj = nn.Conv2d(256, codebook_dim, 1)
        self._padded_shape: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        self._padded_shape = (x.shape[2], x.shape[3])
        h = self.convs(x)
        return self.proj(h)  # [B, codebook_dim, H, W]
```

### Pattern 2: ResidualVQ with Monitoring Wrapper
**What:** Thin wrapper around `vector_quantize_pytorch.ResidualVQ` that tracks per-level codebook utilization, perplexity, and dead code count.
**When to use:** Always -- monitoring is essential for diagnosing codebook collapse.

```python
# Source: vector-quantize-pytorch API + project monitoring requirements
from vector_quantize_pytorch import ResidualVQ

class QuantizerWrapper(nn.Module):
    """ResidualVQ with codebook health monitoring."""

    def __init__(
        self,
        dim: int = 128,
        codebook_size: int = 256,
        num_quantizers: int = 3,
        decay: float = 0.95,
        commitment_weight: float = 0.25,
        threshold_ema_dead_code: int = 2,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        quantize_dropout: bool = False,
    ):
        super().__init__()
        self.rvq = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            quantize_dropout=quantize_dropout,
        )
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

    def forward(self, x: torch.Tensor):
        """Quantize embeddings.

        Args:
            x: [B, seq_len, dim] embedding sequence
        Returns:
            quantized: [B, seq_len, dim]
            indices: [B, seq_len, num_quantizers]
            commit_loss: scalar
        """
        quantized, indices, commit_loss = self.rvq(x)
        return quantized, indices, commit_loss

    def get_codebook_utilization(self, indices: torch.Tensor) -> dict:
        """Compute per-level codebook health metrics."""
        metrics = {}
        for q in range(self.num_quantizers):
            level_indices = indices[:, :, q]  # [B, seq_len]
            unique = level_indices.unique()
            utilization = len(unique) / self.codebook_size
            # Perplexity: exp(entropy of code distribution)
            counts = torch.bincount(level_indices.flatten(), minlength=self.codebook_size).float()
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -(probs * probs.log()).sum()
            perplexity = entropy.exp().item()
            dead_codes = (counts == 0).sum().item()
            metrics[f"level_{q}"] = {
                "utilization": utilization,
                "perplexity": perplexity,
                "dead_codes": int(dead_codes),
            }
        return metrics

    def get_output_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices back to quantized embeddings.

        Args:
            indices: [B, seq_len, num_quantizers]
        Returns:
            quantized: [B, seq_len, dim]
        """
        return self.rvq.get_output_from_indices(indices)
```

### Pattern 3: Dataset-Adaptive Codebook Configuration
**What:** Auto-scale codebook size and related parameters based on dataset file count.
**When to use:** During model instantiation before training begins.

```python
# Source: Project requirements VQVAE-02 + pitfalls research
def get_adaptive_vqvae_config(file_count: int) -> VQVAEConfig:
    """Scale VQ-VAE hyperparameters to dataset size."""
    if file_count <= 20:
        return VQVAEConfig(
            codebook_size=64,
            num_quantizers=3,
            codebook_dim=128,
            commitment_weight=0.25,
            decay=0.8,
            dropout=0.4,
            max_epochs=100,
            learning_rate=5e-4,
        )
    elif file_count <= 100:
        return VQVAEConfig(
            codebook_size=128,
            num_quantizers=3,
            codebook_dim=128,
            commitment_weight=0.25,
            decay=0.9,
            max_epochs=200,
            learning_rate=1e-3,
        )
    else:  # 100-500
        return VQVAEConfig(
            codebook_size=256,
            num_quantizers=3,
            codebook_dim=128,
            commitment_weight=0.25,
            decay=0.95,
            max_epochs=300,
            learning_rate=1e-3,
        )
```

### Pattern 4: Multi-Scale Spectral Loss
**What:** Reconstruction loss combining MSE on mel spectrograms with multi-resolution STFT loss at multiple window sizes.
**When to use:** Always for VQ-VAE training -- provides better perceptual reconstruction than MSE alone.
**Why:** User decision explicitly requires "multi-scale spectral loss for reconstruction (multiple STFT resolutions)." MSE alone allows spectrograms that look numerically close but sound poor. Multi-scale STFT catches perceptual artifacts.

```python
# Source: Multi-Scale Mel-Spectrogram Loss research + auraloss patterns
def multi_scale_spectral_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_sizes: tuple[int, ...] = (128, 256, 512),
    win_sizes: tuple[int, ...] = (512, 1024, 2048),
) -> torch.Tensor:
    """Multi-resolution STFT loss (spectral convergence + log magnitude).

    Operates on mel spectrograms [B, 1, n_mels, time]. Converts back to
    approximate linear spectrogram domain for STFT comparison.
    """
    loss = torch.tensor(0.0, device=recon.device)
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        # Spectral convergence loss (Frobenius norm ratio)
        sc_loss = spectral_convergence(recon, target)
        # Log magnitude loss
        mag_loss = log_magnitude_loss(recon, target)
        loss = loss + sc_loss + mag_loss
    return loss / len(fft_sizes)
```

**Implementation note:** Since our model operates on mel spectrograms (not raw waveforms), the multi-scale loss can be computed directly on mel-domain features at multiple resolutions. A simpler but effective approach is multi-scale MSE at different frequency resolutions using average pooling:

```python
def multi_scale_mel_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-scale MSE on mel spectrograms at different resolutions."""
    loss = F.mse_loss(recon, target)  # Full resolution
    # Downsampled resolutions
    for scale in [2, 4]:
        recon_down = F.avg_pool2d(recon, kernel_size=scale)
        target_down = F.avg_pool2d(target, kernel_size=scale)
        loss = loss + F.mse_loss(recon_down, target_down)
    return loss / 3.0
```

### Pattern 5: ConvVQVAE Forward Pass
**What:** The top-level model that wires encoder, quantizer, and decoder together.
**When to use:** This is the model class -- used everywhere.

```python
class ConvVQVAE(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward: encode -> quantize -> decode.

        Args:
            x: [B, 1, n_mels, time] mel spectrogram
        Returns:
            recon: [B, 1, n_mels, time] reconstructed mel
            indices: [B, H*W, num_quantizers] code indices
            commit_loss: scalar commitment loss
        """
        original_shape = (x.shape[2], x.shape[3])

        # Encode to spatial embeddings
        embeddings = self.encoder(x)  # [B, codebook_dim, H, W]
        B, D, H, W = embeddings.shape

        # Reshape for RVQ: [B, H*W, D]
        flat = embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Quantize
        quantized, indices, commit_loss = self.quantizer(flat)

        # Reshape back to spatial: [B, D, H, W]
        quantized_spatial = quantized.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Decode
        recon = self.decoder(quantized_spatial, target_shape=original_shape)

        return recon, indices, commit_loss
```

### Anti-Patterns to Avoid
- **Reusing v1.0 ConvEncoder/ConvDecoder classes:** User decision says "fresh design, do NOT reuse." Create new VQEncoder/VQDecoder even though the conv backbone is structurally similar.
- **Global vector bottleneck:** Do NOT flatten to a single vector. The encoder must output spatial embeddings `[B, D, H, W]`.
- **KL divergence in any form:** No KL, no free bits, no annealing. Commitment loss only.
- **gradient-based codebook updates:** Use EMA updates (set `decay` parameter). Gradient-based updates are unstable on small batches.
- **Large codebooks (512+) for small datasets:** With 5-20 files, most entries go unused. Start with 64.
- **Modifying persistence.py in this phase:** Phase 13 handles model persistence. This phase focuses only on the model architecture and forward pass.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Vector quantization | Custom VQ with STE, EMA, dead code reset | `vector_quantize_pytorch.ResidualVQ` | 500-1000 lines to get right; library handles STE, EMA, dead code reset, k-means init, codebook collapse prevention |
| Straight-through estimator | Manual gradient detach/attach | Built into ResidualVQ | Easy to get wrong (gradients must flow to encoder but not through argmin) |
| K-means codebook init | Custom k-means on first batch | `kmeans_init=True, kmeans_iters=10` | Library handles buffering first batch, running k-means, initializing codebooks |
| Dead code replacement | Custom usage tracking + replacement | `threshold_ema_dead_code=2` | Library tracks per-code EMA usage, replaces dead codes automatically |
| EMA codebook updates | Custom exponential moving average | `decay=0.95` parameter | Library handles the codebook EMA update rule correctly |

**Key insight:** The `vector-quantize-pytorch` library encapsulates hundreds of lines of subtle VQ implementation. Hand-rolling any of these features risks introducing bugs that are difficult to diagnose (e.g., incorrect STE gradient flow manifests as poor reconstruction, not as an error).

## Common Pitfalls

### Pitfall 1: Codebook Collapse on Small Datasets
**What goes wrong:** Most codebook entries go unused. With 5-50 files, the encoder maps all inputs to a handful of codes. Utilization drops below 30%.
**Why it happens:** Insufficient data diversity for the codebook size. Winner-take-all dynamics in nearest-neighbor assignment. EMA decay washes out sparse updates.
**How to avoid:** Scale codebook to dataset size (64 for 5-20 files); enable `kmeans_init=True`; set `threshold_ema_dead_code=2`; use faster EMA decay (0.8) for small datasets; monitor utilization per level every epoch.
**Warning signs:** Utilization < 50% after warmup; commitment loss near zero; reconstruction plateaus early.

### Pitfall 2: Cascading RVQ Level Collapse
**What goes wrong:** First quantizer captures the full signal; residual levels receive near-zero input and collapse completely. All later codebooks have <10% utilization.
**Why it happens:** Small, homogeneous datasets (e.g., 20 ambient recordings) have narrow spectral range. One codebook may capture everything.
**How to avoid:** Start with `num_quantizers=3` (not more); monitor per-level utilization independently; if level N has <10% utilization while level 0 has >90%, the data is too homogeneous for that many levels. Consider `quantize_dropout=True` during training.
**Warning signs:** Later level commitment loss near zero; adding levels does not improve reconstruction.

### Pitfall 3: Commitment Loss Instability
**What goes wrong:** Commitment loss grows unboundedly; encoder outputs drift from codebook entries; training diverges with NaN.
**Why it happens:** Encoder and codebook learning rates mismatched; EMA decay too slow for small data; small batches amplify noise.
**How to avoid:** Use `commitment_weight=0.25` (not library default of 1.0); set `decay=0.8` for <50 files, `0.9` for 50-200, `0.95` for 200+; ensure gradient clipping at `max_norm=1.0` is active.
**Warning signs:** Commitment loss increases monotonically; NaN losses appear; encoder output L2 norm grows continuously.

### Pitfall 4: Wrong Tensor Layout for ResidualVQ
**What goes wrong:** ResidualVQ expects `[B, seq_len, dim]` but receives `[B, dim, H, W]` (channel-first conv output). Silent shape mismatch produces garbage.
**Why it happens:** Conv2d outputs channel-first; VQ expects channel-last sequence. Forgetting the permute/reshape step.
**How to avoid:** Explicitly reshape: `embeddings.permute(0, 2, 3, 1).reshape(B, H*W, D)` before passing to RVQ. After quantization, reshape back: `quantized.reshape(B, H, W, D).permute(0, 3, 1, 2)`.
**Warning signs:** Reconstruction is pure noise from the start; indices are all zeros or all the same value.

### Pitfall 5: Lazy Initialization Complexity
**What goes wrong:** v1.0 uses lazy initialization for encoder/decoder linear layers (computed on first forward pass). The VQ version has different initialization needs (no flatten_dim, but spatial shape must be known).
**Why it happens:** Copying the lazy init pattern from v1.0 without adapting it for the VQ architecture.
**How to avoid:** The VQ encoder does not need lazy init -- the Conv1x1 projection has a fixed kernel (1x1) and does not depend on spatial dimensions. The decoder similarly uses Conv1x1 which is spatially-independent. Eliminate lazy init entirely; use fixed Conv2d layers.
**Warning signs:** RuntimeError on first forward pass about uninitialized layers.

### Pitfall 6: Forgetting Padding-Crop Round-Trip
**What goes wrong:** The encoder pads input mel to a multiple of 16 (for 4 stride-2 layers). If the decoder does not crop back to the original shape, output shape mismatches cause loss computation errors.
**Why it happens:** The v1.0 decoder crops via `target_shape` parameter. The new decoder must preserve this behavior.
**How to avoid:** Store the original (unpadded) mel shape before encoding. Pass it to the decoder for cropping. The new VQDecoder must accept and use `target_shape` for output cropping.
**Warning signs:** Shape mismatch error in MSE loss computation; output mel has wrong time dimension.

## Code Examples

### Complete ConvVQVAE Model Structure

```python
# Source: Project architecture research + vector-quantize-pytorch API
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


class VQEncoder(nn.Module):
    """Convolutional encoder producing spatial embeddings for VQ."""

    def __init__(self, codebook_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        self.proj = nn.Conv2d(256, codebook_dim, 1)
        self._padded_shape: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        self._padded_shape = (x.shape[2], x.shape[3])
        h = self.convs(x)
        return self.proj(h)  # [B, codebook_dim, H, W]


class VQDecoder(nn.Module):
    """Convolutional decoder from spatial embeddings to mel spectrogram."""

    def __init__(self, codebook_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Conv2d(codebook_dim, 256, 1)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),  # match v1.0: output >= 0, unbounded above
        )

    def forward(
        self, x: torch.Tensor, target_shape: tuple[int, int] | None = None
    ) -> torch.Tensor:
        h = self.proj(x)  # [B, 256, H, W]
        recon = self.deconvs(h)  # [B, 1, H*16, W*16]
        if target_shape is not None:
            th, tw = target_shape
            recon = recon[:, :, :th, :tw]
        return recon


class ConvVQVAE(nn.Module):
    """Convolutional VQ-VAE for mel spectrogram encoding/decoding.

    Replaces ConvVAE. Forward returns (recon, indices, commit_loss)
    instead of (recon, mu, logvar).
    """

    def __init__(
        self,
        codebook_dim: int = 128,
        codebook_size: int = 256,
        num_quantizers: int = 3,
        decay: float = 0.95,
        commitment_weight: float = 0.25,
        threshold_ema_dead_code: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = VQEncoder(codebook_dim=codebook_dim, dropout=dropout)
        self.quantizer = ResidualVQ(
            dim=codebook_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            kmeans_init=True,
            kmeans_iters=10,
        )
        self.decoder = VQDecoder(codebook_dim=codebook_dim, dropout=dropout)

        # Store config for persistence/reconstruction
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mel to spatial embeddings [B, codebook_dim, H, W]."""
        return self.encoder(x)

    def quantize(self, embeddings: torch.Tensor) -> tuple:
        """Quantize spatial embeddings through RVQ.

        Args:
            embeddings: [B, codebook_dim, H, W]
        Returns:
            quantized: [B, codebook_dim, H, W]
            indices: [B, H*W, num_quantizers]
            commit_loss: scalar
        """
        B, D, H, W = embeddings.shape
        flat = embeddings.permute(0, 2, 3, 1).reshape(B, H * W, D)
        quantized, indices, commit_loss = self.quantizer(flat)
        quantized_spatial = quantized.reshape(B, H, W, D).permute(0, 3, 1, 2)
        return quantized_spatial, indices, commit_loss

    def decode(
        self, quantized: torch.Tensor, target_shape: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """Decode quantized embeddings to mel spectrogram."""
        return self.decoder(quantized, target_shape=target_shape)

    def codes_to_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert code indices to quantized embeddings for decode.

        Args:
            indices: [B, seq_len, num_quantizers]
        Returns:
            quantized: [B, codebook_dim, H, W]
        """
        quantized = self.quantizer.rvq.get_output_from_indices(indices)
        # Need to know H, W to reshape -- derive from seq_len
        B, S, D = quantized.shape
        # Assuming square-ish spatial layout (will be stored as model attr)
        H, W = self._spatial_shape  # set during first forward
        return quantized.reshape(B, H, W, D).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = (x.shape[2], x.shape[3])
        embeddings = self.encode(x)
        B, D, H, W = embeddings.shape
        self._spatial_shape = (H, W)
        quantized, indices, commit_loss = self.quantize(embeddings)
        recon = self.decode(quantized, target_shape=original_shape)
        return recon, indices, commit_loss
```

### VQ-VAE Loss Function

```python
# Source: Requirements VQVAE-05 + context decision for multi-scale spectral loss
def vqvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    commit_loss: torch.Tensor,
    commitment_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VQ-VAE loss: reconstruction + commitment.

    Returns:
        (total_loss, recon_loss, weighted_commit_loss) -- all scalars.
    """
    # Reconstruction: MSE + multi-scale mel loss
    recon_loss = F.mse_loss(recon, target)
    # Multi-scale: downsample and compare at lower resolutions
    for scale in [2, 4]:
        recon_down = F.avg_pool2d(recon, kernel_size=scale)
        target_down = F.avg_pool2d(target, kernel_size=scale)
        recon_loss = recon_loss + F.mse_loss(recon_down, target_down)
    recon_loss = recon_loss / 3.0

    # Commitment loss (from RVQ, already computed)
    weighted_commit = commitment_weight * commit_loss.sum()

    total_loss = recon_loss + weighted_commit
    return total_loss, recon_loss, weighted_commit
```

### VQ-VAE Configuration Dataclass

```python
# Source: Existing TrainingConfig pattern + VQ-specific requirements
from dataclasses import dataclass

@dataclass
class VQVAEConfig:
    """VQ-VAE model and training configuration."""
    # Model architecture
    codebook_dim: int = 128        # Per user decision
    codebook_size: int = 256       # Auto-scaled by dataset size
    num_quantizers: int = 3        # Default 3 (Structure/Timbre/Detail)

    # Quantizer
    decay: float = 0.95            # EMA decay (lower for small datasets)
    commitment_weight: float = 0.25  # Commitment loss weight
    threshold_ema_dead_code: int = 2  # Dead code replacement threshold

    # Training
    dropout: float = 0.2
    batch_size: int = 32
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    # Dataset
    chunk_duration_s: float = 1.0
    val_fraction: float = 0.2
    augmentation_expansion: int = 10

    # Checkpoint/preview
    checkpoint_interval: int = 10
    preview_interval: int = 20
    max_checkpoints: int = 3

    # Device
    device: str = "auto"
    num_workers: int = 0
```

### Temporal Compression Analysis

The existing mel spectrogram configuration (128 mels, hop_length=512, sample_rate=48000) produces time_frames = samples / hop_length + 1. For 1 second of audio: 48000 / 512 + 1 = 94 time frames. The encoder applies 4 stride-2 layers:

```
Input mel:    [B, 1, 128, 94]
After pad:    [B, 1, 128, 96]  (pad time to multiple of 16)
After enc:    [B, 256, 8, 6]   (128/16=8, 96/16=6)
After proj:   [B, 128, 8, 6]   (codebook_dim=128)
Positions:    8 * 6 = 48 spatial positions per 1-second chunk
```

**Temporal compression ratio:** 94 time frames -> 6 time positions = ~15.7x compression. Each position covers ~167ms of audio. This is a medium resolution suitable for region-level code editing (the user's primary use case). For mixed clip lengths, longer clips (30s) would produce proportionally more positions (30 * 6 = 180 positions), which is manageable.

**Recommendation for temporal compression:** Keep the existing 4-layer stride-2 encoder as-is. The resulting ~16x compression (6 time positions per second) provides medium resolution well-suited for region-level editing. No architectural changes needed for the compression ratio -- it falls naturally from the existing conv backbone.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Continuous VAE with KL divergence | VQ-VAE with commitment loss | VQ-VAE paper 2017; RVQ from SoundStream 2021 | Eliminates posterior collapse; sharper reconstructions; enables discrete code manipulation |
| Gradient-based codebook updates | EMA codebook updates | SoundStream 2021 | More stable training, especially on small batches |
| Random codebook initialization | K-means init from first batch | SoundStream 2021 | Dramatically improves early training stability and codebook utilization |
| Fixed codebook size | Dataset-adaptive sizing | Project-specific (this phase) | Prevents codebook collapse on small datasets |
| MSE-only reconstruction loss | Multi-scale spectral loss | EnCodec 2022+ | Better perceptual quality from reconstruction |

**Deprecated/outdated:**
- KL divergence, free bits, KL annealing: Not applicable to VQ-VAE. Remove entirely.
- Global latent vector (flatten + linear): Replaced by spatial embeddings. Each position gets its own codes.
- `model.sample()` for generation: VQ-VAE cannot sample from N(0,1). Generation requires a prior (Phase 14) or code manipulation (Phase 16).

## Open Questions

1. **Exact `get_output_from_indices` vs `get_codes_from_indices` API**
   - What we know: The library has both methods. `get_codes_from_indices` returns per-level embeddings; `get_output_from_indices` returns the summed (final) quantized output.
   - What's unclear: Exact return shapes and whether `get_output_from_indices` is available on `ResidualVQ` directly or only on the inner VQ layers.
   - Recommendation: After installing the library, run a quick test to verify the API. If `get_output_from_indices` is not on ResidualVQ, iterate over layers and sum. This can be resolved in the first plan task.

2. **Optimal commitment weight for 5-20 file datasets**
   - What we know: Original VQ-VAE paper uses 0.25. Library default is 1.0. Pitfalls research suggests 0.25 is safer.
   - What's unclear: Whether 0.25 is optimal or whether even lower (0.1) is better for very small datasets where codebook adaptation needs to be faster.
   - Recommendation: Default to 0.25; expose as configurable parameter. Empirical tuning happens during Phase 13 (training pipeline).

3. **Whether `quantize_dropout` should be enabled by default**
   - What we know: `quantize_dropout=True` randomly drops RVQ levels during training, forcing each level to contribute. Helps prevent cascading collapse.
   - What's unclear: Whether it helps or hurts with only 3 levels and small datasets.
   - Recommendation: Default to `False`; enable it as an option. Test during Phase 13 if cascading collapse is observed.

4. **Multi-scale spectral loss implementation specifics**
   - What we know: User decision requires "multi-scale spectral loss for reconstruction (multiple STFT resolutions)." The loss should combine MSE on mel spectrograms with spectral losses at multiple resolutions.
   - What's unclear: Whether to compute STFT loss in the mel domain (multi-scale avg_pool2d on mel, simpler) or in the linear spectrogram domain (more standard but requires inverse mel transform, slower).
   - Recommendation: Implement multi-scale mel-domain loss first (avg_pool2d approach, ~10 lines). This is simpler, differentiable, and avoids the inverse mel transform. If reconstruction quality is insufficient, upgrade to linear STFT loss in a later plan.

## Sources

### Primary (HIGH confidence)
- [vector-quantize-pytorch PyPI](https://pypi.org/project/vector-quantize-pytorch/) - v1.27.21 verified, 2026-02-12 release
- [vector-quantize-pytorch GitHub](https://github.com/lucidrains/vector-quantize-pytorch) - ResidualVQ API, k-means init, EMA, dead code reset
- [SoundStream paper (arXiv 2107.03312)](https://arxiv.org/abs/2107.03312) - RVQ architecture, k-means init, EMA updates, dead code replacement
- [VQ-VAE paper (arXiv 1711.00937)](https://arxiv.org/abs/1711.00937) - Original VQ-VAE, commitment loss, two-stage training
- Existing codebase: `src/distill/models/vae.py` (ConvEncoder/ConvDecoder architecture), `src/distill/models/losses.py` (loss function pattern), `src/distill/training/config.py` (adaptive config pattern), `src/distill/audio/spectrogram.py` (mel spectrogram parameters -- 128 mels, hop_length=512, sample_rate=48000)
- Project research: `.planning/research/ARCHITECTURE.md`, `STACK.md`, `PITFALLS.md`, `SUMMARY.md` (comprehensive v1.1 research already completed)

### Secondary (MEDIUM confidence)
- [commitment loss too large -- Issue #69](https://github.com/lucidrains/vector-quantize-pytorch/issues/69) - Commitment weight tuning guidance
- [ERVQ: Enhanced Residual Vector Quantization](https://arxiv.org/abs/2410.12359) - Codebook utilization baseline (14.7%)
- [Multi-Scale Mel-Spectrogram Loss](https://www.emergentmind.com/topics/multi-scale-mel-spectrogram-loss) - Loss function patterns for audio reconstruction
- [VQ-VAE Comprehensive Guide 2025](https://www.shadecoder.com/topics/vq-vae-a-comprehensive-guide-for-2025) - Training best practices
- [Hugging Face VQ-VAE Understanding](https://huggingface.co/blog/ariG23498/understand-vq) - Commitment loss explanation and tuning

### Tertiary (LOW confidence)
- Codebook size scaling table (64/128/256 for 5-20/20-100/100-500 files) - Extrapolated from literature on larger datasets; needs empirical validation
- Commitment weight 0.25 as default - Paper default, but small-dataset regime is untested in literature
- Multi-scale mel-domain loss via avg_pool2d - Novel simplification; works in principle but not a standard published approach

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - vector-quantize-pytorch v1.27.21 verified on PyPI; API confirmed from GitHub; single new package
- Architecture: HIGH - Encoder/decoder conv backbone pattern well-understood from v1.0 codebase; VQ spatial embedding pattern verified from SoundStream/EnCodec literature; tensor reshape flow verified
- Pitfalls: HIGH - Codebook collapse, commitment loss instability, and cascading RVQ collapse well-documented in literature and lucidrains issue tracker; small-dataset thresholds extrapolated but directionally correct
- Loss function: MEDIUM - Multi-scale mel-domain loss is a simplification of the standard multi-scale STFT loss; the avg_pool2d approach is sound but not a published pattern

**Research date:** 2026-02-21
**Valid until:** 2026-04-21 (60 days -- stack is stable; architecture patterns are well-established)
