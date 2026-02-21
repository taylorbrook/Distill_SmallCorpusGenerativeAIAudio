# Technology Stack: v1.1 RVQ-VAE + Autoregressive Prior

**Project:** Distill -- Small Dataset Generative Audio
**Milestone:** v1.1 VQ-VAE
**Researched:** 2026-02-21
**Scope:** NEW dependencies only. Existing stack (PyTorch, TorchAudio, Gradio, Typer, Rich, soundfile, mutagen, sofar) is validated and not re-researched.

## Recommended Stack Additions

### VQ/RVQ Layer: vector-quantize-pytorch (lucidrains)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| vector-quantize-pytorch | >=1.27.0 | RVQ layers (codebook, EMA, dead code reset, k-means init) | The de facto standard VQ/RVQ library for PyTorch. Actively maintained (1.27.21 released 2026-02-12). Provides `ResidualVQ` with every feature the project needs out of the box: k-means codebook initialization, EMA updates, dead code replacement via `threshold_ema_dead_code`, configurable `num_quantizers` and `codebook_size`. Writing a custom implementation would take 500-1000 lines and miss edge cases this library handles (gradient straight-through, codebook collapse prevention, rotation trick). Used by SoundStream/Encodec-style systems throughout the research community. |

**Confidence:** HIGH -- PyPI verified version, MIT licensed, 3500+ GitHub stars, actively maintained by lucidrains (prolific PyTorch library author).

**Key API for this project:**

```python
from vector_quantize_pytorch import ResidualVQ

rvq = ResidualVQ(
    dim=256,                    # encoder output dimension
    codebook_size=256,          # codes per quantizer (start small for 5-500 files)
    num_quantizers=4,           # coarse-to-fine layers
    kmeans_init=True,           # init codebook from first batch (SoundStream paper)
    kmeans_iters=10,            # k-means iterations for init
    decay=0.99,                 # EMA decay for codebook updates
    threshold_ema_dead_code=2,  # replace codes with cluster size < 2
)

x = torch.randn(1, 94, 256)  # [batch, time_frames, dim]
quantized, indices, commit_loss = rvq(x)
# quantized: [1, 94, 256] -- same shape, quantized
# indices:   [1, 94, 4]   -- code indices per quantizer
# commit_loss: [1, 4]     -- commitment loss per quantizer
```

**Return values integrate directly with the existing loss computation pattern** (see `src/distill/models/losses.py`). The `commit_loss` replaces KL divergence as the regularization term. Reconstruction loss (MSE on mel spectrograms) stays the same.

### VQ Library Dependencies

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| einops | >=0.8.0 | Tensor reshaping (required by vector-quantize-pytorch) | Required dependency of vector-quantize-pytorch. Provides readable tensor operations (`rearrange`, `reduce`). Already proven in the ecosystem (0.8.2 released 2026-01-26). Pure Python, zero compatibility risk. |
| einx[torch] | >=0.1.3 | Extended Einstein operations (required by vector-quantize-pytorch) | Required dependency of vector-quantize-pytorch. Provides additional tensor notation used internally by the library. Lightweight, no additional transitive dependencies beyond torch. |

**Confidence:** HIGH -- these are hard dependencies of vector-quantize-pytorch, versions verified from setup.py analysis.

### Autoregressive Prior: No New Library Needed

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| torch.nn (LSTM) | (bundled with PyTorch) | Autoregressive prior over discrete codes | Use a small LSTM (2-3 layers, 256-512 hidden) as the prior model. For 5-500 audio files producing sequences of ~94 time steps with 4 quantizer levels, a transformer would be massively overparameterized and prone to overfitting. LSTMs have stronger sequential inductive bias, need fewer parameters (under 2M vs 10M+ for a transformer), and train reliably on tiny datasets. PyTorch's `nn.LSTM` is battle-tested on all backends (CUDA, MPS, CPU). No new dependency required. |

**Confidence:** HIGH -- this is a pure architectural decision using existing PyTorch primitives. Multiple sources confirm LSTMs outperform transformers on small datasets due to inductive bias and parameter efficiency.

**Rationale for LSTM over Transformer:**

| Factor | LSTM (2-layer, 512 hidden) | Transformer (4-layer, 4-head, 256 dim) |
|--------|---------------------------|----------------------------------------|
| Parameter count | ~1.5M | ~8-12M |
| Min training data | Works with 50-500 sequences | Needs 5000+ sequences |
| Sequential inductive bias | Built-in (recurrence) | Must be learned (positional encoding) |
| Overfitting risk at 5-100 files | Moderate (manageable with dropout) | Very high |
| MPS/CUDA/CPU compat | Excellent (core PyTorch op) | Excellent (core PyTorch op) |
| Implementation complexity | Simple (`nn.LSTM` + linear head) | Moderate (attention masks, positional encoding) |
| Training speed | Fast (small model) | Slower (attention is O(n^2)) |

**When to upgrade to transformer:** If the dataset grows to 1000+ files and sequence lengths exceed 200 tokens, a small transformer with 2-4 layers could be beneficial. This is a future optimization, not a v1.1 concern.

**Prior architecture sketch:**

```python
class CodePrior(nn.Module):
    """Autoregressive prior over RVQ code sequences."""
    def __init__(self, num_codes=256, num_quantizers=4, hidden=512, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_codes, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers,
                           batch_first=True, dropout=0.3)
        # Predict next code for each quantizer level
        self.heads = nn.ModuleList([
            nn.Linear(hidden, num_codes) for _ in range(num_quantizers)
        ])

    def forward(self, codes):
        # codes: [B, T, num_quantizers] -- integer indices
        # Train: teacher-forced, predict next timestep
        ...
```

### Codebook Visualization: No New Library Needed

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| matplotlib | >=3.9 (already installed) | Codebook utilization heatmaps, code frequency histograms | Already in the project's dependencies. Sufficient for codebook usage heatmaps, per-quantizer utilization bar charts, and dead code monitoring. The existing training loop already uses matplotlib for loss charts. Gradio can display matplotlib figures directly. No need for Plotly -- this project does not need interactive web-based charts for codebook visualization. |

**What to visualize (using existing matplotlib):**

1. **Codebook utilization heatmap:** `[num_quantizers x codebook_size]` matrix showing how often each code is selected. Identifies dead codes (never used) and dominant codes (potential collapse).
2. **Per-quantizer usage histogram:** Bar chart of code frequency per RVQ level. Healthy codebooks show roughly uniform usage.
3. **Code sequence patterns:** Temporal heatmap of `[time_frames x num_quantizers]` showing which codes activate over time for a given audio file.
4. **Reconstruction quality over RVQ depth:** Progressive reconstruction using 1, 2, 3, ... N quantizer levels, showing how each level refines audio quality.

### Audio Processing: No New Dependencies

No new audio processing libraries are needed. The existing stack handles everything:

| Existing Technology | v1.1 Usage |
|---------------------|------------|
| torchaudio MelSpectrogram | Same mel spectrogram pipeline -- RVQ-VAE operates on the same mel representation as the v1.0 VAE |
| torchaudio GriffinLim | Same mel-to-waveform reconstruction for previews and generation |
| soundfile | Same audio I/O for export |
| mutagen | Same metadata embedding |

The RVQ-VAE replaces the *latent representation* (continuous z to discrete codes) but the input/output pipeline stays identical: waveform -> mel spectrogram -> encoder -> **[RVQ here]** -> decoder -> mel spectrogram -> waveform.

## What NOT to Add

| Tempting Addition | Why to Avoid |
|-------------------|--------------|
| PyTorch Lightning | The project already has a well-structured training loop in `src/distill/training/loop.py` with checkpointing, metrics, previews, and cancellation. Lightning would require rewriting all of this for marginal benefit. The codebase pattern (lazy imports, callback system, cancel events) is custom and works. |
| Weights & Biases / MLflow | Over-engineering for a local creative tool with 5-500 files. The existing Rich-based console logging and matplotlib loss charts are sufficient. Adding cloud experiment tracking adds complexity and a network dependency for no user benefit. |
| Hydra config system | The project uses dataclasses (`TrainingConfig`, `SpectrogramConfig`) with JSON serialization. This pattern is simpler and already works. Hydra is for research labs managing hundreds of experiment configs. |
| librosa | The project explicitly chose TorchAudio + soundfile over librosa. TorchAudio provides all needed transforms (mel spectrogram, Griffin-Lim). librosa would add a redundant dependency. |
| pedalboard | Data augmentation is already implemented in `src/distill/audio/augmentation.py` using TorchAudio transforms. Adding Spotify's pedalboard is unnecessary. |
| auraloss (multi-resolution STFT loss) | For v1.1, MSE on mel spectrograms remains the reconstruction loss (same as v1.0). Multi-resolution STFT loss is a potential future improvement but adds a dependency for something that can be implemented in ~50 lines if needed. Focus v1.1 on getting RVQ-VAE working first. |
| DAC / EnCodec | These are complete neural codecs (encoder + RVQ + decoder trained together). This project builds its own encoder/decoder architecture around mel spectrograms. Importing DAC/EnCodec would replace the entire architecture rather than extending it. |
| RAVE | Same issue as DAC/EnCodec -- it is a complete system, not a component. The project's architecture is intentionally custom. |
| plotly | Interactive web charts are unnecessary. Matplotlib figures render in Gradio tabs. The codebook visualizations are diagnostic, not user-facing interactive dashboards. |
| einx (standalone) | Installed automatically as a dependency of vector-quantize-pytorch. Do not install separately to avoid version conflicts. |

## Integration Points with Existing Code

### Model Architecture (`src/distill/models/`)

| Existing File | Change Required | Notes |
|---------------|-----------------|-------|
| `vae.py` | Replace entirely with `vqvae.py` | New `ConvVQVAE` class. Encoder stays convolutional (similar structure). Decoder stays convolutional. The middle bottleneck changes from `(mu, logvar) -> reparameterize -> z` to `encoder_output -> ResidualVQ -> quantized`. |
| `losses.py` | Rewrite for VQ losses | Replace KL divergence with commitment loss from ResidualVQ. Keep MSE reconstruction loss. Remove `free_bits`, `kl_weight`, KL annealing -- these are continuous VAE concepts that do not apply to VQ-VAE. |
| `persistence.py` | Extend for new model format | Save/load must handle: VQ-VAE state dict (different keys), codebook state (EMA buffers), prior model state dict. The `.distill` model format needs a version bump. |

### Training Loop (`src/distill/training/`)

| Existing File | Change Required | Notes |
|---------------|-----------------|-------|
| `loop.py` | Rewrite training/validation epochs | VQ-VAE forward pass returns `(recon, indices, commit_loss)` not `(recon, mu, logvar)`. Loss function changes. Remove KL annealing. Add codebook utilization monitoring. Add prior model training (can be same loop or separate phase). |
| `config.py` | Extend with VQ-specific params | Add: `codebook_size`, `num_quantizers`, `commitment_weight`, `ema_decay`, `dead_code_threshold`. Remove: `kl_warmup_fraction`, `kl_weight_max`, `free_bits`. |
| `metrics.py` | Add VQ-specific metrics | New metrics: codebook utilization (active codes / total codes), perplexity, commitment loss. Remove: KL divergence, posterior collapse warning. |

### Controls (`src/distill/controls/`)

| Existing File | Change Required | Notes |
|---------------|-----------------|-------|
| `analyzer.py` | Replace PCA analysis with codebook analysis | PCA-based latent space analysis is continuous-VAE specific. Replace with: codebook usage statistics, code co-occurrence patterns, per-quantizer analysis. |
| `mapping.py` | Replace slider mapping with code manipulation | PCA sliders -> code swapping/blending UI. The generation paradigm shifts from "slide along PCA axis" to "select/swap/blend discrete codes." |

### UI (`src/distill/ui/`)

| Existing File | Change Required | Notes |
|---------------|-----------------|-------|
| `tabs/generate_tab.py` | Major rewrite | Replace PCA slider generation with: (1) autoregressive prior sampling, (2) code editing interface. |
| New: `tabs/code_tab.py` | New file | Code manipulation UI: encode audio -> view codes as grid/heatmap -> swap/blend/edit -> decode back to audio. |

## Installation

```bash
# In project root (H:/dev/Distill-vqvae)
# Only ONE new package to install (brings einops and einx as dependencies):
pip install "vector-quantize-pytorch>=1.27.0"

# Or with uv (project uses uv):
uv add "vector-quantize-pytorch>=1.27.0"
```

**That's it.** No other new dependencies are needed for v1.1.

## pyproject.toml Addition

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "vector-quantize-pytorch>=1.27.0",
]
```

## Version Compatibility

| Package | Version | Compatible With | Verified |
|---------|---------|-----------------|----------|
| vector-quantize-pytorch | 1.27.21 | PyTorch >= 2.0 (no upper bound pinned), Python >= 3.9 | PyPI metadata, 2026-02-12 release |
| einops | 0.8.2 | PyTorch (all versions), Python >= 3.8 | PyPI metadata, 2026-01-26 release |
| einx[torch] | 0.3.0 | PyTorch (all versions), Python >= 3.9 | PyPI metadata, 2024-06-11 release |
| PyTorch nn.LSTM | (bundled) | CUDA, MPS, CPU -- all backends | Core PyTorch operation |

**Note on installed PyTorch version:** The pyproject.toml specifies `torch>=2.10.0,<2.11` but the current venv has `torch==2.4.0+cu118`. vector-quantize-pytorch works with both -- it has no upper or lower bound on torch beyond requiring it to be installed. When the venv is updated to match pyproject.toml, no issues expected.

## MPS / CUDA / CPU Compatibility

### vector-quantize-pytorch

| Backend | Status | Notes |
|---------|--------|-------|
| **CUDA** | Fully supported | All operations run on GPU. No known issues. |
| **CPU** | Fully supported | All operations run on CPU. Slower but functional. |
| **MPS** | Partially supported, workaround available | A historical crash was reported (issue #55) due to `aten::lerp.Scalar_out` not being supported on MPS and ComplexFloat dtype incompatibility. The issue is **closed** on GitHub. For safety: (1) the existing project pattern of running InverseMelScale on CPU provides a template for MPS workarounds, (2) k-means init should use `kmeans_init=True, kmeans_iters=10` which runs during the first forward pass and may need CPU fallback on older MPS, (3) EMA updates use `lerp_` which has improved MPS support in PyTorch 2.4+. **Recommendation:** Test on MPS early in development. If issues arise, the VQ layer can be wrapped to run on CPU (it is not the bottleneck -- the conv encoder/decoder are). |

### nn.LSTM (Prior Model)

| Backend | Status | Notes |
|---------|--------|-------|
| **CUDA** | Fully supported | cuDNN-accelerated. Fast. |
| **CPU** | Fully supported | Standard implementation. |
| **MPS** | Fully supported | LSTM is well-supported on MPS since PyTorch 2.0. |

**Overall MPS strategy:** Same as v1.0 -- use `float32` throughout (no float16 on MPS), apply the existing smoke test pattern, fall back selectively to CPU for operations that fail. The project already handles this in `src/distill/hardware/device.py`.

## Small Dataset Sizing Recommendations

For 5-500 audio files, codebook and model sizes must be scaled down from the literature (SoundStream uses 1024 codes, 8 quantizers on millions of hours of audio):

| Dataset Size | Codebook Size | Num Quantizers | Prior Hidden | Rationale |
|-------------|--------------|----------------|--------------|-----------|
| 5-20 files | 64 | 2-3 | 256 | Tiny dataset: more codes = more dead codes. 2 quantizers give coarse + fine. |
| 20-100 files | 128-256 | 3-4 | 256-512 | Medium dataset: 128 codes is usually sufficient. 3-4 quantizers for detail. |
| 100-500 files | 256-512 | 4-6 | 512 | Larger dataset: can support bigger codebook. 4-6 quantizers for nuance. |

**Key insight:** At 5-20 files producing ~50-200 mel spectrogram chunks each, the total training set is 250-4000 sequences of ~94 time steps. A codebook of 1024 would have most entries unused. Start with `codebook_size=128, num_quantizers=4` as a default and expose these as training config parameters.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| VQ Library | vector-quantize-pytorch | Custom VQ implementation | Custom VQ is 500-1000 lines to get right (straight-through estimator, EMA updates, dead code replacement, k-means init, codebook collapse prevention). The library handles all edge cases and is well-tested. The "not invented here" cost is not worth it for VQ. |
| VQ Library | vector-quantize-pytorch | torchvq | Less maintained, fewer features, no RVQ support. |
| Prior Model | LSTM | Transformer | Transformers overfit on small datasets (5-500 files produce <5000 training sequences). LSTMs have sequential inductive bias built in and need 5-10x fewer parameters. |
| Prior Model | LSTM | WaveNet-style dilated convolutions | More complex to implement, marginal benefit for short sequences (~94 time steps). WaveNet shines on very long sequences (thousands of steps). |
| Prior Model | LSTM | PixelCNN / PixelSNAIL | Designed for 2D grids (images). Code sequences are 1D temporal. LSTM is more natural. |
| Codebook Viz | matplotlib | Plotly | Already installed, renders in Gradio, sufficient for diagnostic heatmaps. Plotly adds a heavy dependency for no added value in a local creative tool. |
| Codebook Viz | matplotlib | tensorboard | Already installed but the project does not use TensorBoard. Adding TensorBoard logging for one feature is over-engineering. |

## Sources

**HIGH Confidence (Official/Verified):**
- [vector-quantize-pytorch PyPI](https://pypi.org/project/vector-quantize-pytorch/) -- v1.27.21, 2026-02-12
- [vector-quantize-pytorch GitHub](https://github.com/lucidrains/vector-quantize-pytorch) -- ResidualVQ API, features, parameters
- [einops PyPI](https://pypi.org/project/einops/) -- v0.8.2, 2026-01-26
- [einx PyPI](https://pypi.org/project/einx/) -- v0.3.0, 2024-06-11
- [vector-quantize-pytorch setup.py dependencies](https://github.com/OpenDocCN/python-code-anls/blob/master/docs/lucidrains/vector-quantize-pytorch----setup.py.md) -- einops>=0.7.0, einx[torch]>=0.1.3, torch

**MEDIUM Confidence (Verified with multiple sources):**
- [vector-quantize-pytorch MPS Issue #55](https://github.com/lucidrains/vector-quantize-pytorch/issues/55) -- MPS crash (closed/resolved)
- [SoundStream Paper](https://arxiv.org/abs/2107.03312) -- RVQ architecture, k-means init, codebook design
- [LSTM vs Transformer for small datasets](https://discuss.pytorch.org/t/cant-get-transformer-to-exceed-lstm-help/210178) -- PyTorch community discussion
- [Optimizing Deeper Transformers on Small Datasets](https://aclanthology.org/2021.acl-long.163.pdf) -- ACL 2021 paper
- [VQ-VAE + Transformer Systems](https://www.emergentmind.com/topics/vq-vae-transformer-systems) -- Architecture overview

**LOW Confidence (Single source, needs validation during implementation):**
- MPS compatibility of vector-quantize-pytorch with PyTorch 2.10+ -- the closed issue is from PyTorch 2.0 era. Modern PyTorch likely resolves it but should be smoke-tested.
- Codebook size recommendations for very small datasets (5-20 files) -- extrapolated from literature on larger datasets. No published research on this exact regime.

---
*Stack research for: Distill v1.1 RVQ-VAE + Autoregressive Prior*
*Researched: 2026-02-21*
