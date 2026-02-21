# Architecture: RVQ-VAE + Autoregressive Prior Integration

**Domain:** Integrating RVQ-VAE and autoregressive prior into existing Distill codebase
**Researched:** 2026-02-21
**Confidence:** HIGH (core VQ/RVQ patterns well-established; library API verified)

## Executive Summary

The v1.1 architecture replaces the continuous Gaussian latent space with Residual Vector Quantization (RVQ) inserted between the existing convolutional encoder and decoder. The encoder output is projected to a sequence of spatial embeddings, quantized through stacked codebooks (coarse-to-fine), then decoded. An autoregressive prior (lightweight Transformer) is trained in a separate second stage on the frozen VQ-VAE's code indices, enabling generation of new code sequences that decode to novel audio. A code manipulation UI lets users encode audio to codes, edit them, and decode back.

## Current Architecture (v1.0)

```
Audio -> Mel Spectrogram -> [Encoder] -> mu,logvar -> [Reparameterize] -> z -> [Decoder] -> Mel -> Griffin-Lim -> Audio
              |                |              |                          |          |
         [B,1,128,94]    [B,64] each    [B,64]                    [B,64]   [B,1,128,94]
```

Key characteristics:
- Encoder: 4-layer Conv2d (1->32->64->128->256, stride 2), flatten, linear -> mu, logvar
- Latent: 64-dim continuous Gaussian vector per mel spectrogram (entire 1-second chunk)
- Decoder: linear -> reshape, 4-layer ConvTranspose2d (256->128->64->32->1, stride 2)
- Loss: MSE reconstruction + KL divergence with free bits and annealing
- Generation: sample z ~ N(0,1), decode to mel, Griffin-Lim to waveform

## Target Architecture (v1.1)

```
Audio -> Mel -> [Encoder] -> embeddings -> [RVQ] -> codes -> [Decoder] -> Mel -> Griffin-Lim -> Audio
                    |              |            |          |         |
              [B,1,128,94]  [B,D,H,W]    [B,H*W,Q]  [B,D,H,W]  [B,1,128,94]
                                              |
                                    [Prior Model] (stage 2)
                                     generates new code sequences
```

### Data Flow Detail

```
TRAINING (Stage 1: VQ-VAE):

  mel [B,1,128,94]
    |
  Encoder convolutions (4 layers, stride 2)
    |
  feature_map [B,256,H,W]  where H=8, W=6 (128/16, 94->96->96/16=6)
    |
  Conv1x1 project to [B,D,H,W] where D = codebook embedding dim (e.g., 256)
    |
  Reshape to [B, H*W, D] = [B, 48, 256]  (sequence of spatial positions)
    |
  ResidualVQ: quantize each of 48 positions through Q codebooks
    |
  quantized [B, 48, 256], indices [B, 48, Q], commit_loss [B, Q]
    |
  Reshape to [B, D, H, W]
    |
  Decoder transposed convolutions (4 layers, stride 2)
    |
  recon_mel [B,1,128,94]

  Loss = MSE(recon_mel, mel) + commit_weight * commit_loss.sum()


TRAINING (Stage 2: Autoregressive Prior):

  Freeze VQ-VAE encoder + codebooks entirely.

  mel [B,1,128,94]
    |
  Frozen Encoder -> Frozen RVQ -> indices [B, 48, Q]
    |
  Flatten to code sequence: [B, 48*Q] or model hierarchically
    |
  Transformer (causal): predict next code token given previous tokens
    |
  Loss = cross-entropy per token


GENERATION (from prior):

  Prior Model autoregressively generates code indices [48, Q]
    |
  ResidualVQ.get_codes_from_indices(indices)  ->  quantized [48, D]
    |
  Reshape to [D, H, W]
    |
  Decoder -> mel -> Griffin-Lim -> audio


CODE MANIPULATION:

  Audio -> Mel -> Encoder -> RVQ -> indices [48, Q]
    |
  UI: visualize codes as grid (48 positions x Q layers)
  User: swap codes at positions, copy from other audio, randomize
    |
  Modified indices -> RVQ.get_codes_from_indices -> Decoder -> Mel -> Audio
```

## Component Boundaries

### New Components

| Component | File | Responsibility | Communicates With |
|-----------|------|----------------|-------------------|
| **ConvVQVAE** | `models/vqvae.py` | Top-level VQ-VAE model wrapping encoder, RVQ, decoder | Training loop, generation pipeline, persistence |
| **ResidualVQ wrapper** | `models/quantizer.py` | Thin wrapper around `vector_quantize_pytorch.ResidualVQ` with codebook monitoring | ConvVQVAE |
| **VQ losses** | `models/losses.py` (extend) | Reconstruction + commitment loss; codebook usage metrics | Training loop |
| **AutoregressivePrior** | `models/prior.py` | Lightweight Transformer for code sequence prediction | Prior training loop, generation pipeline |
| **Prior training loop** | `training/prior_loop.py` | Stage 2 training: freeze VQ-VAE, train prior on code indices | Training runner |
| **Code manipulator** | `inference/codes.py` | Encode to codes, decode from codes, code editing operations | Generation pipeline, UI |
| **Codes UI tab** | `ui/tabs/codes_tab.py` | Gradio tab for code visualization, editing, encode/decode | Code manipulator, generation pipeline |

### Modified Components

| Component | File | Change |
|-----------|------|--------|
| **Persistence** | `models/persistence.py` | New format version, store codebook config, model type discriminator |
| **Checkpoint** | `training/checkpoint.py` | Store VQ-VAE state + prior state, codebook stats |
| **Training config** | `training/config.py` | VQ-specific hyperparameters (codebook size, num quantizers, commitment weight) |
| **Training loop** | `training/loop.py` | Replace vae_loss with VQ loss, add codebook utilization logging |
| **Generation pipeline** | `inference/generation.py` | Two generation modes: prior-based and code-based (replacing latent vector) |
| **Chunking** | `inference/chunking.py` | Replace latent vector interpolation with code-sequence-based generation |
| **Model catalog** | `library/catalog.py` | Model type field, prior model association |
| **Train tab** | `ui/tabs/train_tab.py` | VQ training parameters, codebook stats display, two-stage training UI |
| **Generate tab** | `ui/tabs/generate_tab.py` | Prior-based generation controls, remove slider/PCA controls |

### Unchanged Components

| Component | Reason |
|-----------|--------|
| `audio/spectrogram.py` | Mel conversion stays identical |
| `audio/io.py`, `audio/validation.py` | Audio I/O unchanged |
| `data/dataset.py` | Dataset management unchanged |
| `training/dataset.py` | PyTorch Dataset unchanged (still yields waveform chunks) |
| `inference/export.py` | Export pipeline unchanged |
| `inference/spatial.py`, `inference/stereo.py` | Spatial processing unchanged |
| `audio/filters.py` | Anti-aliasing unchanged |
| `inference/quality.py` | Quality scoring unchanged |
| `presets/manager.py` | Presets will change semantics but can remain structurally similar |
| `hardware/` | Device detection unchanged |
| `cli/` | CLI commands extend, not replace |

## Architectural Decisions

### Decision 1: RVQ as a drop-in layer between encoder and decoder (not end-to-end waveform codec)

**Choice:** Keep the mel spectrogram representation. Insert RVQ between existing encoder feature maps and decoder, operating on 2D spatial positions.

**Why not SoundStream/EnCodec style (raw waveform in, raw waveform out)?**
- The existing encoder/decoder already handles mel spectrograms well
- Griffin-Lim is sufficient for non-real-time generation (quality > latency)
- Building a waveform codec requires a neural vocoder (HiFi-GAN) which is a huge addition
- 2D spatial positions from the mel encoder give natural coarse-to-fine structure
- Preserves all existing audio pipeline (spectrogram, filters, spatial processing)

**Confidence:** HIGH. This is the simplest integration path that achieves discrete codes while reusing 90% of existing infrastructure.

### Decision 2: Separate two-stage training (VQ-VAE first, prior second)

**Choice:** Train VQ-VAE to convergence, freeze it, then train the autoregressive prior on the resulting code indices.

**Why not joint training?**
- Joint training couples two complex objectives, making debugging harder
- The VQ-VAE must converge first to produce stable codebook assignments
- Prior quality depends on stable codes; unstable codes = unstable prior training
- Two-stage is the proven approach from VQ-VAE, VQ-VAE-2, and SoundStream
- Simpler to implement, debug, and reason about
- User can train VQ-VAE once, then experiment with different prior architectures

**Why not end-to-end like SoundStream?**
- SoundStream trains encoder+decoder+RVQ jointly because it has adversarial+reconstruction losses that need end-to-end gradients
- Our setup uses MSE on mel spectrograms, which works fine with straight-through estimator through VQ
- End-to-end is only needed when the discriminator needs to see final output quality

**Confidence:** HIGH. Two-stage is standard and simpler for this use case.

### Decision 3: Lightweight causal Transformer for prior (not PixelCNN)

**Choice:** Small GPT-style Transformer (4-6 layers, 256-dim, 4 heads) trained on flattened code sequences.

**Why Transformer over PixelCNN?**
- PixelCNN was designed for 2D spatial data (images); our code grid is small (48 positions)
- Transformer handles variable-length sequences naturally
- Attention over 48*Q tokens is computationally trivial (no memory issues)
- Better long-range dependencies for temporal coherence in audio
- Easier to implement with PyTorch's `nn.TransformerDecoder`

**Why small?**
- Code vocabulary is small (codebook_size, e.g., 256-512)
- Sequence length is short (48 positions * 4-8 quantizers = 192-384 tokens)
- Small datasets (5-500 files) mean limited training data for the prior
- Overfitting is the main risk, not underfitting
- Target: ~1-5M parameters

**Confidence:** HIGH. Standard approach, well-validated architecture.

### Decision 4: Flatten RVQ codes for prior modeling (not hierarchical)

**Choice:** Flatten the `[positions, quantizers]` code grid into a single sequence and model autoregressively.

The flattening interleaves quantizer levels at each position: `[pos0_q0, pos0_q1, ..., pos0_qQ, pos1_q0, ...]`. This means the prior first predicts all Q quantizer codes for position 0, then all Q for position 1, etc.

**Why this ordering?**
- Within a position, quantizer codes are strongly correlated (coarse-to-fine residuals)
- Across positions, temporal patterns emerge naturally from left-to-right
- The prior can learn both local (within-position) and global (across-position) patterns
- Simple to implement: just reshape and predict next token

**Alternative considered:** Model quantizer levels separately (VQ-VAE-2 style with top/bottom priors). Rejected because our RVQ has uniform resolution (not hierarchical top/bottom), and the added complexity is not justified for 48 positions.

**Confidence:** MEDIUM. The flat approach works but may be less efficient for many quantizer levels. If Q > 8, consider hierarchical modeling.

### Decision 5: Codebook sizes adapted to dataset scale

**Choice:** Default codebook configuration scales with dataset size:

| Dataset Size | Codebook Size (K) | Num Quantizers (Q) | Effective Codes |
|--------------|--------------------|---------------------|-----------------|
| 5-20 files   | 64                 | 4                   | 64^4 ~16M       |
| 20-100 files | 128                | 4                   | 128^4 ~268M     |
| 100-500 files| 256                | 8                   | 256^8 huge      |

**Why scale?**
- Small datasets have limited variety; large codebooks lead to many dead codes
- Dead codes waste capacity and make prior training harder (sparse distribution)
- Smaller codebooks with EMA updates and dead code reset are more stable
- Users can override defaults

**Confidence:** MEDIUM. The exact numbers need empirical tuning, but the principle of scaling with data is well-established.

### Decision 6: Encoder outputs spatial embeddings (not single global vector)

**Choice:** The encoder produces a 2D feature map `[B, D, H, W]` that is treated as a sequence of `H*W` embedding vectors, each quantized independently through RVQ.

**This is a fundamental change from v1.0** where the encoder collapses the entire feature map to a single 64-dim vector via flatten + linear. In v1.1:
- Remove the `fc_mu` and `fc_logvar` linear layers from the encoder
- The conv output `[B, 256, H, W]` becomes the embedding sequence directly
- Add a 1x1 Conv2d to project from 256 channels to the codebook dimension `D`
- Each spatial position is independently quantized

**Why?**
- VQ operates on individual embedding vectors, not global summaries
- Spatial positions preserve local structure in the mel spectrogram
- Multiple codebook entries per mel = finer reconstruction
- This is how SoundStream, EnCodec, and all successful audio VQ systems work

**Confidence:** HIGH. This is the standard approach for VQ on feature maps.

## Encoder/Decoder Modifications

### Encoder: From Global Vector to Spatial Embeddings

```
v1.0 Encoder:
  mel [B,1,128,94] -> convs -> [B,256,8,6] -> flatten -> [B,12288] -> fc_mu -> [B,64]
                                                                     -> fc_logvar -> [B,64]

v1.1 Encoder:
  mel [B,1,128,94] -> convs -> [B,256,8,6] -> conv1x1 -> [B,D,8,6]
                                                              |
                                              reshape to [B, 48, D] for VQ
```

The convolutional backbone stays identical (same 4 conv layers, same strides). Only the head changes: replace `flatten + fc_mu + fc_logvar` with a `Conv2d(256, D, 1)` projection.

### Decoder: From Global Vector to Spatial Embeddings

```
v1.0 Decoder:
  z [B,64] -> fc -> [B,12288] -> reshape -> [B,256,8,6] -> deconvs -> [B,1,128,94]

v1.1 Decoder:
  quantized [B,D,8,6] -> conv1x1 -> [B,256,8,6] -> deconvs -> [B,1,128,94]
```

Replace the `fc` linear layer with a `Conv2d(D, 256, 1)` projection. The transposed conv backbone stays identical.

### Implementation Strategy

Create new classes (`VQEncoder`, `VQDecoder`) that inherit from or wrap the existing conv backbones, rather than modifying `ConvEncoder`/`ConvDecoder` in place. This keeps v1.0 code intact (though v1.0 models are explicitly out of scope for backward compat).

```python
class VQEncoder(nn.Module):
    """Encoder for VQ-VAE: conv backbone -> spatial embeddings."""

    def __init__(self, embed_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        # Same conv backbone as ConvEncoder
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        # Project to codebook dimension
        self.proj = nn.Conv2d(256, embed_dim, 1)
        self._padded_shape: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode mel to spatial embeddings [B, embed_dim, H, W]."""
        # Pad to multiple of 16 (same as v1.0)
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        self._padded_shape = (x.shape[2], x.shape[3])

        h = self.convs(x)
        return self.proj(h)  # [B, embed_dim, H, W]
```

## Quantizer Configuration

### vector-quantize-pytorch ResidualVQ API

```python
from vector_quantize_pytorch import ResidualVQ

quantizer = ResidualVQ(
    dim=256,                      # embedding dimension (must match encoder output)
    num_quantizers=4,             # number of residual quantization levels
    codebook_size=256,            # entries per codebook
    kmeans_init=True,             # initialize codebooks from first batch k-means
    kmeans_iters=10,              # k-means iterations for init
    threshold_ema_dead_code=2,    # replace codes with EMA usage below this
    commitment_weight=1.0,        # commitment loss weight (library handles internally)
    decay=0.99,                   # EMA decay for codebook updates
    stochastic_sample_codes=True, # probabilistic code selection during training
    sample_codebook_temp=1.0,     # temperature for stochastic sampling (anneal to 0)
)

# Forward pass
quantized, indices, commit_loss = quantizer(embeddings)
# quantized: [B, seq_len, dim] - quantized embeddings (straight-through)
# indices:   [B, seq_len, num_quantizers] - code indices per position per level
# commit_loss: [B, num_quantizers] - commitment losses per level

# Decode from indices (for generation / code manipulation)
quantized = quantizer.get_codes_from_indices(indices)
# indices: [B, seq_len, num_quantizers] -> quantized: [B, seq_len, dim]
```

### Codebook Health Monitoring

Track these metrics during training:

| Metric | How | Warning Threshold |
|--------|-----|-------------------|
| Codebook utilization | `unique(indices) / codebook_size` per quantizer | < 50% after warmup |
| Dead code count | Codes with EMA count < threshold | > 30% of codebook |
| Commitment loss | Per-quantizer commitment loss values | Diverging or NaN |
| Perplexity | `exp(entropy(code_distribution))` | < codebook_size/4 |
| Codebook velocity | L2 norm of codebook update per step | Sudden spikes |

## Autoregressive Prior Architecture

### Model Design

```python
class AutoregressivePrior(nn.Module):
    """Lightweight Transformer for generating code sequences."""

    def __init__(
        self,
        vocab_size: int,          # codebook_size (e.g., 256)
        seq_len: int,             # total sequence length = positions * num_quantizers
        embed_dim: int = 256,     # transformer embedding dimension
        num_heads: int = 4,       # attention heads
        num_layers: int = 4,      # transformer layers
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits for a code sequence.

        Parameters
        ----------
        indices : torch.Tensor
            Shape [B, seq_len] of code indices (flattened from [B, positions, quantizers]).

        Returns
        -------
        torch.Tensor
            Logits [B, seq_len, vocab_size].
        """
        B, T = indices.shape
        tok_emb = self.token_embed(indices)
        pos_emb = self.pos_embed(torch.arange(T, device=indices.device))
        x = tok_emb + pos_emb

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=indices.device)

        # Self-attention only (no cross-attention; use decoder layer with memory=None workaround)
        x = self.transformer(x, memory=torch.zeros(B, 1, x.size(-1), device=x.device), tgt_mask=mask)
        return self.head(x)
```

**Note:** For a cleaner implementation, use `nn.TransformerEncoder` with causal mask instead of `nn.TransformerDecoder` (decoder requires cross-attention memory which is unused here). The above is illustrative.

### Prior Training Loop

```python
def train_prior_epoch(
    prior: AutoregressivePrior,
    vqvae: ConvVQVAE,               # frozen
    train_loader: DataLoader,
    spectrogram: AudioSpectrogram,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """One epoch of prior training on frozen VQ-VAE codes."""
    vqvae.eval()
    prior.train()

    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        # Pre-encode all training data to code indices
        # (or do it on-the-fly as shown here)
        pass

    for batch in train_loader:
        batch = batch.to(device)
        mel = spectrogram.waveform_to_mel(batch)

        # Get code indices from frozen VQ-VAE
        with torch.no_grad():
            embeddings = vqvae.encode(mel)
            _, indices, _ = vqvae.quantize(embeddings)
            # indices: [B, positions, num_quantizers]
            flat_indices = indices.reshape(indices.shape[0], -1)
            # flat_indices: [B, positions * num_quantizers]

        # Teacher-forced next-token prediction
        input_seq = flat_indices[:, :-1]   # [B, T-1]
        target_seq = flat_indices[:, 1:]   # [B, T-1]

        logits = prior(input_seq)          # [B, T-1, vocab_size]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_seq.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return {"prior_loss": total_loss / max(steps, 1)}
```

### Generation from Prior

```python
def generate_from_prior(
    prior: AutoregressivePrior,
    vqvae: ConvVQVAE,
    spectrogram: AudioSpectrogram,
    temperature: float = 1.0,
    top_k: int = 0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a mel spectrogram from the prior."""
    prior.eval()
    vqvae.eval()

    seq_len = prior.seq_len
    generated = torch.zeros(1, 0, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(seq_len):
            logits = prior(generated)[:, -1, :]  # [1, vocab_size]
            logits = logits / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # [1, 1]
            generated = torch.cat([generated, next_token], dim=1)

    # Reshape back to [1, positions, num_quantizers]
    num_positions = vqvae.num_positions
    num_quantizers = vqvae.num_quantizers
    indices = generated.reshape(1, num_positions, num_quantizers)

    # Decode from indices
    quantized = vqvae.codes_to_embeddings(indices)
    mel = vqvae.decode(quantized)
    return mel
```

## Model Persistence Changes

### New Format

```python
SAVED_MODEL_VERSION = 2  # bumped from 1

saved = {
    "format": "distill_model",
    "version": 2,
    "model_type": "vqvae",         # NEW: discriminator "vae" vs "vqvae"
    "model_state_dict": vqvae.state_dict(),

    # VQ-specific config (NEW)
    "vqvae_config": {
        "embed_dim": 256,
        "codebook_size": 256,
        "num_quantizers": 4,
        "num_positions": 48,       # H*W from encoder spatial output
    },

    # Prior model (NEW, optional - may be trained separately)
    "prior_state_dict": prior.state_dict() if prior else None,
    "prior_config": {
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 4,
        "seq_len": 192,            # 48 positions * 4 quantizers
    } if prior else None,

    # Unchanged from v1
    "spectrogram_config": {...},
    "training_config": {...},
    "metadata": {...},

    # Removed
    # "latent_dim": 64,            # replaced by vqvae_config
    # "latent_analysis": {...},    # PCA analysis is v1.0 only; replaced by codebook info
}
```

### Loading Compatibility

```python
def load_model(model_path, device="cpu"):
    saved = torch.load(model_path, ...)

    version = saved.get("version", 1)
    model_type = saved.get("model_type", "vae")  # default to v1.0 VAE

    if model_type == "vae":
        # v1.0 path (existing code, kept for reference but out of scope)
        raise ValueError("v1.0 VAE models are not supported in v1.1")
    elif model_type == "vqvae":
        # v1.1 path
        return _load_vqvae(saved, device)
```

## Training Pipeline Changes

### Stage 1: VQ-VAE Training

The training loop structure stays similar to v1.0 but with key changes:

| Aspect | v1.0 (continuous VAE) | v1.1 (VQ-VAE) |
|--------|----------------------|----------------|
| Forward pass | `recon, mu, logvar = model(mel)` | `recon, indices, commit_loss = model(mel)` |
| Loss | `MSE + kl_weight * KL_divergence` | `MSE + commit_weight * commit_loss` |
| KL annealing | Required (prevents posterior collapse) | Not needed (no KL term) |
| Free bits | Required (prevents posterior collapse) | Not applicable |
| Monitoring | KL divergence, posterior collapse | Codebook utilization, dead codes |
| Preview gen | `z ~ N(0,1)` then decode | Encode random training sample, slight code perturbation |

Key simplification: VQ-VAE eliminates the entire KL-balancing complexity (kl_weight, kl_warmup_fraction, free_bits, kl_weight_max, posterior collapse monitoring). The commitment loss has a single weight and is much simpler to tune.

### Stage 2: Prior Training

After VQ-VAE training completes, the UI should offer "Train Prior" as a separate action:

1. Load the trained VQ-VAE model (frozen)
2. Re-encode all training data to code indices (can be cached to disk)
3. Train the Transformer prior on these indices
4. Save the prior model alongside or within the VQ-VAE `.distill` file

This is a much faster training stage (typically converges in fewer epochs than the VQ-VAE since the data is compressed and the model is smaller).

### Training Config Changes

```python
@dataclass
class VQVAETrainingConfig:
    """VQ-VAE specific training configuration."""
    # Encoder/Decoder (inherited from v1.0 spirit)
    embed_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 32
    max_epochs: int = 200
    learning_rate: float = 1e-3

    # VQ-specific
    codebook_size: int = 256
    num_quantizers: int = 4
    commitment_weight: float = 1.0
    codebook_decay: float = 0.99
    kmeans_init: bool = True
    dead_code_threshold: int = 2

    # Regularization (simplified from v1.0)
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    augmentation_expansion: int = 10

    # Checkpoint/preview (same as v1.0)
    checkpoint_interval: int = 10
    preview_interval: int = 20
    max_checkpoints: int = 3
    val_fraction: float = 0.2

@dataclass
class PriorTrainingConfig:
    """Prior model training configuration."""
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
```

## Generation Pipeline Changes

### Current Generation Modes (v1.0)

1. **Random:** Sample z ~ N(0,1), decode
2. **Slider-controlled:** PCA analysis maps sliders to latent directions, construct z, decode
3. **Interpolation:** SLERP between random z vectors for temporal evolution

### New Generation Modes (v1.1)

1. **Prior-based:** Autoregressive prior generates code sequence, decode via VQ-VAE decoder
2. **Code manipulation:** Encode input audio to codes, edit codes, decode
3. **Code interpolation:** Interpolate code sequences between two encoded audio files

The generation pipeline (`GenerationPipeline`) needs to support both a VQ-VAE model and an optional prior model. The `GenerationConfig` gains new fields:

```python
@dataclass
class GenerationConfig:
    # Existing fields (some repurposed)
    duration_s: float = 1.0
    seed: int | None = None
    # ...spatial, export fields unchanged...

    # NEW: Generation mode
    generation_mode: str = "prior"  # "prior", "encode_decode", "code_interpolation"

    # Prior generation controls
    temperature: float = 1.0
    top_k: int = 50

    # Code manipulation (when mode = "encode_decode")
    source_audio_path: str | None = None

    # REMOVED / DEPRECATED
    # latent_vector: no longer applicable (was PCA-based)
    # evolution_amount: replaced by temperature
    # concat_mode: replaced by generation_mode
```

### Multi-chunk Generation from Prior

For audio longer than 1 second, generate multiple code sequences and stitch the decoded mel spectrograms using the existing overlap-add approach:

```
For each chunk:
  1. Prior generates [48, Q] code indices (conditioned on end of previous chunk)
  2. Decode to mel [1, 1, 128, 94]
  3. Overlap-add with Hann window (existing synthesize_continuous_mel logic)

Final mel -> Griffin-Lim -> audio
```

The key change: instead of interpolating between random latent vectors, we generate discrete code sequences. The temporal coherence comes from the prior's autoregressive nature (it learns temporal patterns from training data).

For cross-chunk coherence, the prior can be conditioned on the last few positions of the previous chunk's codes (sliding context window).

## Code Manipulation UI

### Encode/Decode Tab

```
[Audio Input]  -->  [Encode Button]  -->  [Code Grid Display]
                                              |
                                    [Edit Tools: swap, copy, randomize]
                                              |
                                    [Decode Button]  -->  [Audio Output]
```

### Code Grid Visualization

Display codes as a grid: rows = quantizer levels (coarse to fine), columns = spatial positions (time/frequency):

```
Quantizer 0 (coarse): [  42 ] [ 128 ] [  7  ] [ 201 ] ... (48 positions)
Quantizer 1:          [ 156 ] [  33 ] [ 89  ] [ 112 ] ...
Quantizer 2:          [  5  ] [ 210 ] [  44 ] [  67 ] ...
Quantizer 3 (fine):   [ 198 ] [  71 ] [ 153 ] [  29 ] ...
```

### Code Operations

| Operation | Description | UI |
|-----------|-------------|-----|
| **Encode** | Audio -> mel -> encoder -> RVQ -> indices | Button + audio input |
| **Decode** | Indices -> VQ lookup -> decoder -> mel -> audio | Button -> audio output |
| **Swap** | Replace codes at selected positions with codes from another audio | Select source + target positions |
| **Copy** | Copy a region of codes from one encoded audio to another | Select source region -> paste |
| **Randomize** | Replace selected codes with random valid indices | Select positions + randomize button |
| **Interpolate** | Blend codes between two encoded audios | Two audio inputs + blend slider |
| **Coarse edit** | Only modify top quantizer levels (broad changes) | Quantizer level selector |
| **Fine edit** | Only modify bottom quantizer levels (subtle changes) | Quantizer level selector |

## Patterns to Follow

### Pattern 1: Straight-Through Estimator

The VQ operation is non-differentiable (argmin). Use straight-through estimator (STE): in the forward pass, use the quantized output; in the backward pass, pass gradients directly to the encoder output as if quantization did not happen. This is handled automatically by `vector-quantize-pytorch`.

```python
# ResidualVQ handles STE internally:
quantized, indices, commit_loss = self.quantizer(embeddings)
# quantized has gradients flowing through as if embeddings were used
# commit_loss = ||embeddings - sg(quantized)||^2 (pulls encoder toward codebook)
```

### Pattern 2: Codebook Warmup

For the first N steps of training, the codebook initializes via k-means on encoder outputs. During this warmup:
- Reconstruction quality may be poor (expected)
- Do not judge model quality until after warmup
- k-means init with `kmeans_init=True, kmeans_iters=10` is sufficient

### Pattern 3: EMA Codebook Updates (Not Gradient-Based)

Codebook embeddings update via exponential moving average of assigned encoder outputs, not via gradient descent. This is more stable and is the standard approach (SoundStream, EnCodec, all modern VQ systems). Controlled by `decay` parameter in ResidualVQ.

### Pattern 4: Dead Code Reset

When a codebook entry is not selected by any encoder output for too long (measured by EMA usage count falling below `threshold_ema_dead_code`), replace it with a randomly selected encoder output from the current batch. This prevents codebook collapse. Handled automatically by `vector-quantize-pytorch`.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Training Prior and VQ-VAE Jointly from Scratch

**What goes wrong:** The codebook assignments are unstable early in VQ-VAE training. Training the prior on unstable codes produces a useless prior that must be retrained.

**Instead:** Two-stage training. VQ-VAE first, prior second (on frozen codes).

### Anti-Pattern 2: Using Gradient-Based Codebook Updates

**What goes wrong:** Gradient-based codebook updates are unstable, especially with small batch sizes. Codes oscillate and never converge.

**Instead:** Use EMA updates (`decay=0.99`). The library default handles this correctly.

### Anti-Pattern 3: Large Codebooks on Small Datasets

**What goes wrong:** With 5-50 files, most codebook entries are never used (dead codes). The effective codebook is much smaller than configured, wasting capacity and confusing the prior.

**Instead:** Start with small codebooks (64-128) and scale up only if codebook utilization is >80%. Monitor dead code percentage.

### Anti-Pattern 4: Generating from Random Codes Without Prior

**What goes wrong:** Random code combinations produce incoherent noise. Unlike continuous VAE where N(0,1) sampling works (because the encoder is regularized toward the prior), VQ codebook entries have no inherent distributional structure.

**Instead:** Always use the trained prior for generation of new audio. Code manipulation (editing existing encoded audio) is fine because you start from valid code sequences.

### Anti-Pattern 5: Modifying All Quantizer Levels Simultaneously in Code Editing

**What goes wrong:** Changing codes at every quantizer level at once produces dramatic, often incoherent changes. The coarse levels define the broad structure; fine levels add detail.

**Instead:** Encourage users to edit coarse levels (Q=0,1) for broad changes and fine levels (Q=2,3) for subtle refinement. The UI should make this hierarchy explicit.

## Build Order

### Recommended Phase Structure

| Phase | What | Why This Order | Dependencies |
|-------|------|----------------|-------------|
| **1. VQ-VAE core model** | `VQEncoder`, `VQDecoder`, `ConvVQVAE` with `ResidualVQ` | Foundation; everything depends on this | `vector-quantize-pytorch` package |
| **2. VQ-VAE training loop** | Modified training loop with VQ loss, codebook monitoring | Must verify VQ-VAE trains correctly before building on top | Phase 1 |
| **3. Model persistence** | Save/load VQ-VAE models, format version bump | Needed before prior training (load frozen VQ-VAE) | Phase 1 |
| **4. Autoregressive prior** | Transformer model + training loop | Requires frozen VQ-VAE; second-most complex component | Phases 1-3 |
| **5. Generation pipeline** | Prior-based generation, multi-chunk stitching | Requires both VQ-VAE and prior | Phases 1-4 |
| **6. Code manipulation** | Encode/decode/edit code operations | Requires VQ-VAE model only (not prior) | Phases 1, 3 |
| **7. UI integration** | Train tab updates, codes tab, generate tab updates | Requires all backend components | Phases 1-6 |
| **8. CLI updates** | New CLI commands for VQ training, prior training, code ops | Requires all backend components | Phases 1-6 |

**Phase ordering rationale:**
- Phase 1 is the foundation; all other phases depend on a working VQ-VAE model
- Phase 2 must follow immediately to validate the model works end-to-end
- Phase 3 (persistence) before Phase 4 (prior) because prior training needs to load a frozen VQ-VAE
- Phase 4 before Phase 5 because generation needs the prior
- Phase 6 (code manipulation) is independent of the prior and could theoretically run in parallel with Phase 4, but ordering it after ensures the model architecture is stable
- Phases 7-8 (UI/CLI) are presentation layer and come last

## Scalability Considerations

| Concern | At 5-20 files | At 50-200 files | At 200-500 files |
|---------|---------------|-----------------|------------------|
| Codebook size | K=64, Q=4 | K=128, Q=4 | K=256, Q=8 |
| Dead codes | High risk; aggressive reset | Moderate risk | Low risk |
| VQ-VAE training time | Minutes (small model, small data) | 30-60 min on GPU | 1-3 hours on GPU |
| Prior training time | Fast (short code sequences) | 10-30 min | 30-60 min |
| Code manipulation | Interactive speed (ms per encode/decode) | Same | Same |
| Prior generation | < 1 sec per chunk (small transformer) | Same | Same |
| Memory | < 2 GB | 2-4 GB | 4-8 GB |

## Sources

**High Confidence (Official docs, established research):**
- [vector-quantize-pytorch GitHub](https://github.com/lucidrains/vector-quantize-pytorch) - ResidualVQ API, v1.27.21
- [vector-quantize-pytorch PyPI](https://pypi.org/project/vector-quantize-pytorch/) - Latest version 1.27.21, Feb 2026
- [SoundStream paper (arXiv 2107.03312)](https://arxiv.org/abs/2107.03312) - RVQ architecture, encoder-decoder-RVQ integration
- [VQ-VAE paper (arXiv 1711.00937)](https://arxiv.org/abs/1711.00937) - Original VQ-VAE, two-stage training
- [VQ-VAE-2 paper (NeurIPS 2019)](http://papers.neurips.cc/paper/9625-generating-diverse-high-fidelity-images-with-vq-vae-2.pdf) - Hierarchical VQ, autoregressive priors

**Medium Confidence (Verified from multiple sources):**
- [VQ-VAE-2 comprehensive guide (2025)](https://www.shadecoder.com/topics/vq-vae-2-a-comprehensive-guide-for-2025) - Two-stage training details
- [VQ-VAE-2 implementation](https://github.com/mattiasxu/VQVAE-2) - Reference implementation with autoregressive prior
- [Codebook collapse prevention (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_Addressing_Representation_Collapse_in_Vector_Quantized_Models_with_One_Linear_ICCV_2025_paper.pdf) - Dead code prevention
- [Residual VQ explained](https://drscotthawley.github.io/blog/posts/2023-06-12-RVQ.html) - Detailed RVQ walkthrough
- [Spectrogram Patch Codec (2025)](https://arxiv.org/html/2509.02244v1) - VQ-VAE on mel spectrograms (validates mel-based approach)

**Low Confidence (Architectural recommendations from training knowledge):**
- Prior model sizing (4 layers, 256-dim) -- reasonable starting point but needs empirical validation
- Codebook size scaling by dataset size -- principle is sound but exact numbers are estimated

---

*Architecture research for: RVQ-VAE + Autoregressive Prior Integration*
*Researched: 2026-02-21*
