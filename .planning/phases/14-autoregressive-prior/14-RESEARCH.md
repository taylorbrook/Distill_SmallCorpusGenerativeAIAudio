# Phase 14: Autoregressive Prior - Research

**Researched:** 2026-02-21
**Domain:** Autoregressive sequence modeling over discrete VQ-VAE codes
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- User selects a saved .sda VQ-VAE model from a dropdown to train the prior on
- Prior training automatically uses the same dataset the VQ-VAE was trained on (encodes it to get code sequences)
- No separate dataset selection needed
- Memorization detection uses relaxed sensitivity -- only warn when memorization is very likely (small datasets naturally have low perplexity)
- On detection: show prominent warning with a "Stop and use best checkpoint" button, but don't force early stop
- System automatically tracks best checkpoint (lowest validation perplexity) throughout training -- user can always roll back to pre-memorization weights
- Prior model state is bundled into the .sda file with a `has_prior` flag and metadata (epochs trained, final perplexity, training date)
- Loading a model with a prior auto-loads both VQ-VAE and prior -- ready to generate immediately
- Moderate complexity: epochs, hidden size, number of layers, number of attention heads
- Defaults auto-scale based on dataset size (smaller datasets get smaller priors to prevent overfitting)
- CLI flags mirror UI knobs exactly: `--epochs`, `--hidden-size`, `--layers`, `--heads`

### Claude's Discretion
- Training initiation UX (whether second stage in training tab or separate action)
- Training progress display (same chart with new lines vs separate section)
- Memorization threshold approach (fixed vs adaptive to dataset size)
- Model save behavior (update in-place vs create copy)
- Retrain prior behavior (overwrite vs warn-then-overwrite)
- Prior architecture choice (transformer vs LSTM vs other)
- Learning rate, optimizer, and training hyperparameters
- Adaptive default scaling curves for architecture params

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GEN-01 | User can train an autoregressive prior model on frozen VQ-VAE code sequences | Prior architecture, training loop, code extraction pipeline |
| GEN-05 | Prior model is bundled in the saved model file alongside the VQ-VAE | v2 .distill format extension with `prior_state_dict` and `has_prior` flag |
| GEN-06 | Prior training detects memorization (validation perplexity monitoring) | Perplexity computation, adaptive thresholds, best-checkpoint tracking |
| PERS-02 | Prior model state is bundled in the same model file | Persistence layer extension pattern, LoadedVQModel update |
| CLI-02 | CLI supports prior training on a trained VQ-VAE model | Typer subcommand pattern matching existing `distill train` |
</phase_requirements>

## Summary

Phase 14 builds an autoregressive prior that models the joint distribution of discrete code sequences produced by a frozen VQ-VAE. The prior learns `P(code_t | code_{<t})` over flattened code sequences, enabling generation of new code sequences that can be decoded through the VQ-VAE to produce novel audio. This is the standard two-stage approach used by VQ-VAE-2, Jukebox, and RQ-VAE-Transformer.

The project's VQ-VAE produces indices of shape `[B, seq_len, num_quantizers]` where `seq_len = H * W` (spatial positions, typically 48 for 1-second mel at default settings: 8 height x 6 width) and `num_quantizers` is 2-4 (typically 3). The prior must model this as a sequence. Two main approaches exist: (1) flatten all positions and quantizer levels into a single long sequence (raster-scan order), or (2) predict stacks of codes per position (the RQ-Transformer approach). For this project's small-dataset domain with short sequences (~48 positions x 3 levels = ~144 tokens), a simple flattened approach with a compact transformer is recommended -- the sequence is short enough that a standard causal transformer handles it efficiently without needing the hierarchical RQ-Transformer optimization.

**Primary recommendation:** Use a decoder-only Transformer (GPT-style) built from PyTorch's `nn.TransformerDecoderLayer` with learned positional embeddings, cross-entropy loss on code predictions, and validation perplexity for memorization monitoring. The architecture is simple, well-understood, and PyTorch-native with no additional dependencies.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.nn.TransformerDecoderLayer | PyTorch 2.10 | Transformer decoder blocks | Native PyTorch, no extra deps, supports causal masking via `tgt_is_causal=True` |
| torch.nn.Embedding | PyTorch 2.10 | Code-to-embedding lookup | Standard discrete token embedding |
| torch.nn.functional.cross_entropy | PyTorch 2.10 | Training loss (NLL over code vocabulary) | Standard autoregressive training objective |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.nn.utils.clip_grad_norm_ | PyTorch 2.10 | Gradient clipping | Already used in VQ-VAE training loop |
| torch.optim.AdamW | PyTorch 2.10 | Optimizer | Same as VQ-VAE training (project pattern) |
| torch.optim.lr_scheduler.CosineAnnealingLR | PyTorch 2.10 | LR schedule | Same as VQ-VAE training (project pattern) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Transformer (decoder-only) | LSTM | LSTM is simpler but Transformer generalizes better on structured sequences, especially with positional dependencies across quantizer levels. Transformer also parallelizes training. For ~144-token sequences, Transformer is well within efficiency bounds. |
| Transformer (decoder-only) | PixelCNN | Traditional VQ-VAE-1 approach, but convolutional priors don't naturally handle the 1D flattened code sequence from RVQ as well as attention. PixelCNN is better suited to 2D spatial priors. |
| Flattened sequence | RQ-Transformer (stacked prediction) | RQ-Transformer predicts all quantizer levels per position simultaneously, reducing sequence length. Overkill for 48-position sequences; adds significant complexity. Better for high-res images with thousands of spatial positions. |

**Installation:**
No additional packages needed. Everything is in PyTorch 2.10 (already a project dependency).

## Architecture Patterns

### Recommended Project Structure
```
src/distill/
├── models/
│   ├── prior.py           # CodePrior (transformer decoder-only model)
│   └── persistence.py     # Extended: save/load prior into .distill v2
├── training/
│   ├── prior_loop.py      # train_prior(), train_prior_epoch(), validate_prior_epoch()
│   ├── prior_config.py    # PriorConfig dataclass + get_adaptive_prior_config()
│   └── metrics.py         # Extended: PriorEpochMetrics, PriorStepMetrics
├── cli/
│   └── train_prior.py     # `distill train-prior MODEL_PATH` command
└── ui/
    └── tabs/train_tab.py  # Extended: prior training stage
```

### Pattern 1: Flattened Code Sequence Representation
**What:** Convert `[B, seq_len, num_quantizers]` indices from the VQ-VAE into a flat `[B, seq_len * num_quantizers]` sequence for autoregressive modeling. Each position visits all quantizer levels before moving to the next spatial position: `[pos0_level0, pos0_level1, pos0_level2, pos1_level0, ...]`.
**When to use:** Always -- this is the input format for the prior.
**Why this order:** Interleaving (position-major, level-minor) means each spatial position's codes are predicted together. The coarse level (level 0) at a position is predicted first, then finer levels can condition on it. This matches the residual quantization semantics where level 0 captures structure and later levels capture detail.

```python
def flatten_codes(indices: torch.Tensor) -> torch.Tensor:
    """Flatten [B, seq_len, num_quantizers] -> [B, seq_len * num_quantizers].

    Interleaves: [pos0_q0, pos0_q1, pos0_q2, pos1_q0, pos1_q1, ...]
    """
    B, S, Q = indices.shape
    # Already in [B, S, Q] order -- reshape to [B, S*Q]
    return indices.reshape(B, S * Q)

def unflatten_codes(flat: torch.Tensor, num_quantizers: int) -> torch.Tensor:
    """Unflatten [B, seq_len * num_quantizers] -> [B, seq_len, num_quantizers]."""
    B, L = flat.shape
    S = L // num_quantizers
    return flat.reshape(B, S, num_quantizers)
```

### Pattern 2: Decoder-Only Transformer Prior (GPT-Style)
**What:** A transformer that takes a sequence of code tokens and predicts the next token. Uses learned embeddings + positional embeddings, causal self-attention, and a linear head projecting to codebook vocabulary size.
**When to use:** Core prior architecture.

```python
class CodePrior(nn.Module):
    """Autoregressive prior over flattened VQ-VAE code sequences.

    Architecture: GPT-style decoder-only transformer.
    Input: [B, T] code indices (integers 0..codebook_size-1)
    Output: [B, T, codebook_size] logits for next-token prediction
    """

    def __init__(
        self,
        codebook_size: int,
        seq_len: int,          # max sequence length (spatial_positions * num_quantizers)
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.seq_len = seq_len

        # Token embedding: codebook indices -> hidden dimension
        self.token_emb = nn.Embedding(codebook_size, hidden_size)
        # Positional embedding: learned, covers full flattened sequence
        self.pos_emb = nn.Embedding(seq_len, hidden_size)

        self.drop = nn.Dropout(dropout)

        # Transformer decoder layers (self-attention only, no cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, codebook_size, bias=False)

        # Causal mask (registered as buffer, computed once)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking.

        Parameters
        ----------
        x : torch.Tensor
            [B, T] code indices (long tensor, 0..codebook_size-1)

        Returns
        -------
        torch.Tensor
            [B, T, codebook_size] logits
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.drop(h)

        # Use causal mask to prevent attending to future tokens
        mask = self.causal_mask[:T, :T]

        # nn.TransformerDecoder with self-attention only:
        # pass h as both tgt and memory (decoder-only pattern)
        # memory_mask blocks all cross-attention (not used)
        h = self.transformer(
            tgt=h,
            memory=h,  # dummy -- not used with self-attention-only
            tgt_mask=mask,
        )

        h = self.ln_f(h)
        logits = self.head(h)  # [B, T, codebook_size]
        return logits
```

**Note on decoder-only pattern:** PyTorch's `nn.TransformerDecoder` expects both `tgt` and `memory`. For a decoder-only model (no encoder), pass the same tensor as both `tgt` and `memory`, or (cleaner) build a custom stack of `nn.TransformerEncoderLayer` with `is_causal=True` for the self-attention. The `nn.TransformerEncoderLayer` actually supports causal masking and is simpler for decoder-only use. Recommend using `nn.TransformerEncoder` with causal masking -- architecturally identical to GPT but without the confusing encoder/decoder naming.

### Pattern 3: Code Extraction Pipeline (Frozen VQ-VAE)
**What:** Load a trained VQ-VAE, freeze it, encode the training dataset to get code sequences, then train the prior on those code sequences.
**When to use:** At the start of prior training -- extracting the training data.

```python
def extract_code_sequences(
    model: ConvVQVAE,
    dataloader: DataLoader,
    spectrogram: AudioSpectrogram,
    device: torch.device,
) -> torch.Tensor:
    """Encode entire dataset through frozen VQ-VAE to get code sequences.

    Returns
    -------
    torch.Tensor
        [N, seq_len, num_quantizers] all code indices for the dataset
    """
    model.eval()
    all_indices = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mel = spectrogram.waveform_to_mel(batch)
            _recon, indices, _commit_loss = model(mel)
            all_indices.append(indices.cpu())
    return torch.cat(all_indices, dim=0)
```

### Pattern 4: Best-Checkpoint Tracking with Validation Perplexity
**What:** Track validation perplexity (exp of cross-entropy loss) each epoch. Store the model state_dict of the epoch with lowest validation perplexity. This is the "best checkpoint" the user can roll back to.
**When to use:** Every epoch during prior training.

```python
def compute_perplexity(avg_cross_entropy_loss: float) -> float:
    """Convert average cross-entropy loss to perplexity."""
    import math
    return math.exp(avg_cross_entropy_loss)

# In training loop:
val_loss = validate_prior_epoch(...)  # average cross-entropy
val_perplexity = compute_perplexity(val_loss)

if val_perplexity < best_val_perplexity:
    best_val_perplexity = val_perplexity
    best_prior_state = copy.deepcopy(prior_model.state_dict())
```

### Pattern 5: Model Bundling (Prior into .distill v2)
**What:** Extend the v2 .distill format to include prior model state alongside VQ-VAE weights.
**When to use:** After prior training completes.

The existing v2 format saved dict:
```python
saved = {
    "format": MODEL_FORMAT_MARKER,
    "version": SAVED_MODEL_VERSION_V2,
    "model_type": "vqvae",
    "model_state_dict": model.state_dict(),    # VQ-VAE weights
    "vqvae_config": vqvae_config,
    "spectrogram_config": spectrogram_config,
    "training_config": training_config,
    "codebook_health_snapshot": codebook_health,
    "loss_curve_history": loss_curve_history,
    "metadata": asdict(metadata),
}
```

Extended with prior:
```python
saved.update({
    "has_prior": True,
    "prior_state_dict": prior_model.state_dict(),
    "prior_config": {
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "seq_len": 144,
        "dropout": 0.1,
    },
    "prior_metadata": {
        "epochs_trained": 50,
        "final_perplexity": 12.3,
        "best_perplexity": 11.8,
        "training_date": "2026-02-21T12:00:00Z",
    },
})
```

**Key principle:** Update the existing .distill file in-place. Loading detects `has_prior` and reconstructs both models. The `LoadedVQModel` dataclass gets a `prior` field.

### Anti-Patterns to Avoid
- **Training the VQ-VAE and prior jointly:** The VQ-VAE must be frozen during prior training. Joint training destabilizes both models.
- **Using the same vocabulary embedding for different quantizer levels:** All levels share the same codebook_size, so a single embedding works. But do NOT try to share embeddings across separate codebooks if they have different learned representations (not applicable here since all codes index the same codebook per level in RVQ -- but the spatial positions at different RVQ levels have different semantic meaning).
- **Very long sequences without positional limits:** The project's sequences are ~144 tokens. Do not design for arbitrary-length sequences (that is out of scope per REQUIREMENTS.md).
- **Separate prior model files:** Per user decision, prior state is bundled into the same .distill file. Do NOT create separate files.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Causal attention masking | Custom attention mask logic | `torch.nn.Transformer*` with `is_causal=True` or `tgt_is_causal=True` | PyTorch handles efficient causal masking internally, including fused attention paths |
| Positional encoding | Sinusoidal encoding from scratch | `nn.Embedding(seq_len, hidden_size)` (learned) | For short sequences (~144 tokens), learned positional embeddings work as well or better than sinusoidal and are simpler |
| Gradient clipping | Custom gradient management | `torch.nn.utils.clip_grad_norm_` | Already used throughout the project |
| Perplexity computation | Custom information-theoretic code | `math.exp(cross_entropy_loss)` | It is literally one line; the cross-entropy loss is already computed by the training loop |

**Key insight:** The prior is a standard autoregressive language model where the "vocabulary" is the codebook entries. Everything needed is built into PyTorch. No additional libraries are required.

## Common Pitfalls

### Pitfall 1: Teacher Forcing vs. Autoregressive Gap
**What goes wrong:** The model is trained with teacher forcing (ground-truth previous tokens) but generates autoregressively (its own predictions as input). Errors compound during generation.
**Why it happens:** Standard cross-entropy training always feeds ground truth as context.
**How to avoid:** For this project's small sequences (~144 tokens), the gap is manageable. During training, standard teacher forcing with cross-entropy is correct. During generation (Phase 15), use temperature scaling and top-k/top-p sampling to manage error propagation. No scheduled sampling needed for sequences this short.
**Warning signs:** Generated sequences that start well but degrade quickly or become repetitive.

### Pitfall 2: Memorization on Small Datasets
**What goes wrong:** With 5-20 audio files producing perhaps 50-200 code sequences, the prior can memorize the training set entirely, reaching near-zero training loss and near-perfect perplexity.
**Why it happens:** Small vocabulary (64-256 entries), short sequences (~144 tokens), tiny dataset -- a modest transformer can memorize this easily.
**How to avoid:**
1. Scale prior model size with dataset size (user decision: auto-scaling defaults)
2. Monitor validation perplexity -- if it drops near 1.0, the model is memorizing
3. Use dropout and weight decay as regularization
4. Track best checkpoint -- when memorization starts, validation perplexity rises while training perplexity keeps falling (classic overfitting curve)
**Warning signs:** Validation perplexity dropping then rising while training perplexity continues to fall. For very small datasets, validation perplexity may never rise (dataset is too small for meaningful train/val split), so use absolute thresholds too.

### Pitfall 3: Confusing Code Indices Across Quantizer Levels
**What goes wrong:** The prior treats all tokens as coming from the same vocabulary, but level 0 codes and level 2 codes have very different semantic meaning (structure vs. fine detail).
**Why it happens:** Flattening the sequence loses the level information.
**How to avoid:** Add a **level embedding** alongside the token and position embeddings. Each token gets three embeddings summed: `token_emb(code_index) + pos_emb(position) + level_emb(quantizer_level)`. The level embedding tells the model whether it is predicting a coarse (level 0) or fine (level 2) code.
**Warning signs:** Generated audio that sounds structurally incoherent -- the model treats all codes identically instead of understanding the coarse-to-fine hierarchy.

### Pitfall 4: PyTorch TransformerDecoder Confusion (Encoder-Only vs Decoder-Only)
**What goes wrong:** Using `nn.TransformerDecoder` for a decoder-only model is awkward because it expects cross-attention memory from an encoder.
**Why it happens:** PyTorch's Transformer API was designed for the original encoder-decoder architecture, not GPT-style decoder-only models.
**How to avoid:** Use `nn.TransformerEncoder` with causal masking (`mask` parameter or `is_causal=True` on `nn.TransformerEncoderLayer`). Despite the name, `nn.TransformerEncoder` is just a stack of self-attention layers -- exactly what a decoder-only model needs. The naming is confusing but the architecture is correct.
**Warning signs:** Passing dummy memory tensors, seeing cross-attention warnings, or getting unexpected behavior from the memory pathway.

### Pitfall 5: Forgetting to Freeze VQ-VAE During Prior Training
**What goes wrong:** If the VQ-VAE gradients are not disabled, prior training loss backpropagates into the VQ-VAE, corrupting the codebook.
**Why it happens:** The code extraction step uses `torch.no_grad()`, but if someone accidentally includes the VQ-VAE in the optimizer or calls `.train()` on it.
**How to avoid:** Extract codes ONCE before prior training starts. Store them as a plain tensor dataset. The VQ-VAE model is not needed during prior training at all (only its codes are).
**Warning signs:** VQ-VAE reconstruction quality degrades after prior training.

### Pitfall 6: Update-in-Place File Corruption
**What goes wrong:** Writing the prior state back into the .distill file while it's being read or if the process crashes mid-write, the file is corrupted.
**Why it happens:** `torch.save` is not atomic.
**How to avoid:** Use the project's existing atomic write pattern: write to a temporary file, then `os.replace()` onto the target. The project already uses this pattern for JSON indexes (catalog). Apply the same to .distill file updates.
**Warning signs:** Truncated or corrupt .distill files after prior training.

## Code Examples

### Training Loss: Cross-Entropy on Next-Token Prediction
```python
# Input: [B, T] code indices
# Target: shifted by 1 (predict next token)
# logits: [B, T, codebook_size]

logits = prior_model(input_codes[:, :-1])  # [B, T-1, V]
targets = input_codes[:, 1:]               # [B, T-1]

# Reshape for cross_entropy: [B*(T-1), V] vs [B*(T-1)]
loss = F.cross_entropy(
    logits.reshape(-1, codebook_size),
    targets.reshape(-1),
)
```

### Validation Perplexity Computation
```python
import math

def compute_validation_perplexity(
    prior_model: CodePrior,
    val_codes: torch.Tensor,  # [N_val, seq_len * Q]
    device: torch.device,
    batch_size: int = 32,
) -> float:
    """Compute perplexity on validation code sequences."""
    prior_model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(val_codes), batch_size):
            batch = val_codes[i:i+batch_size].to(device)
            logits = prior_model(batch[:, :-1])
            targets = batch[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, prior_model.codebook_size),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
```

### Adaptive Prior Configuration
```python
def get_adaptive_prior_config(file_count: int) -> PriorConfig:
    """Scale prior model to dataset size to prevent memorization.

    Smaller datasets get smaller models with more regularization.
    Follows the same tier pattern as VQ-VAE config (64/128/256).
    """
    if file_count <= 20:
        return PriorConfig(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            dropout=0.3,
            max_epochs=50,
            learning_rate=3e-4,
        )
    elif file_count <= 100:
        return PriorConfig(
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            dropout=0.2,
            max_epochs=100,
            learning_rate=1e-3,
        )
    else:  # > 100
        return PriorConfig(
            hidden_size=512,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            max_epochs=150,
            learning_rate=1e-3,
        )
```

### Memorization Detection with Adaptive Threshold
```python
def check_memorization(
    val_perplexity: float,
    codebook_size: int,
    dataset_file_count: int,
) -> tuple[bool, str]:
    """Check if validation perplexity indicates memorization.

    Relaxed sensitivity per user decision: only warn when very likely.

    Small datasets naturally have low perplexity because the code
    distribution has less diversity. Threshold scales with dataset size.

    Returns (is_memorizing, message).
    """
    # Theoretical minimum: perplexity 1.0 = perfect prediction
    # Theoretical maximum: perplexity = codebook_size (uniform distribution)
    # Memorization signal: val perplexity near training perplexity AND
    # both approaching 1.0

    # Adaptive threshold: smaller datasets tolerate lower perplexity
    if dataset_file_count <= 20:
        threshold = 2.0   # Very relaxed -- tiny datasets will have low ppl
    elif dataset_file_count <= 100:
        threshold = 3.0   # Moderate
    else:
        threshold = 5.0   # Larger datasets should have higher diversity

    if val_perplexity < threshold:
        return True, (
            f"Validation perplexity ({val_perplexity:.1f}) is very low "
            f"(threshold: {threshold:.1f} for {dataset_file_count} files). "
            f"The prior may be memorizing the training data."
        )

    return False, ""
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PixelCNN prior (VQ-VAE-1, 2017) | Transformer prior (VQ-VAE-2, 2019+) | 2019 | Transformers capture long-range dependencies better than local convolutions |
| Separate prediction per position (VQ-VAE-2) | Stack prediction per position (RQ-VAE, 2022) | 2022 | For RVQ, predict all residual levels per position together -- reduces sequence length |
| Large-scale priors (Jukebox: 72 layers) | Scaled priors for domain size | Ongoing | Small personal datasets need tiny priors to avoid memorization |

**Deprecated/outdated:**
- PixelCNN priors: Replaced by Transformers for VQ-VAE priors. PixelCNN still works but is slower to train and captures fewer long-range dependencies.
- Fixed-size priors: Modern practice scales prior capacity to data availability to prevent memorization.

## Open Questions

1. **Decoder-only via TransformerEncoder vs TransformerDecoder**
   - What we know: `nn.TransformerEncoder` + causal masking is architecturally identical to GPT and simpler than `nn.TransformerDecoder` (no cross-attention). Both work.
   - What's unclear: Whether `is_causal=True` on `nn.TransformerEncoderLayer` is fully supported in PyTorch 2.10 on all backends (CPU, CUDA, MPS).
   - Recommendation: Use `nn.TransformerEncoder` with explicit causal mask tensor (`torch.triu(ones, diagonal=1)`) rather than relying on `is_causal` flag. The mask tensor approach is guaranteed to work everywhere.

2. **Level embedding vs. shared vocabulary**
   - What we know: All RVQ levels share the same codebook_size, so code index 42 at level 0 and code index 42 at level 2 mean different things (structure vs. detail). A level embedding disambiguates.
   - What's unclear: Whether the level embedding makes a meaningful difference for 3 levels (tiny overhead vs. potential quality gain).
   - Recommendation: Include level embeddings -- they are cheap (3 embedding vectors), add no computational cost, and provide the model with crucial structural information about the RVQ hierarchy.

3. **Update-in-place vs. create copy for prior bundling**
   - What we know: User wants prior bundled into the same .sda file.
   - What's unclear: Whether modifying an existing file risks corruption if the user trains a prior on a model they are also generating from.
   - Recommendation: Update in-place using atomic write pattern (write temp file, then `os.replace`). This is safe and matches the user expectation of "one file = one complete model."

4. **Memorization threshold tuning**
   - What we know: Small datasets naturally produce low perplexity. Fixed thresholds may trigger false positives.
   - What's unclear: Exact perplexity values that indicate genuine memorization vs. natural low diversity.
   - Recommendation: Start with adaptive thresholds as coded above. Log perplexity values and tune empirically. The user can always ignore warnings (per the "don't force early stop" decision).

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10 documentation: `nn.TransformerEncoder`, `nn.TransformerDecoderLayer`, `nn.Embedding` -- native API
- Project codebase: `src/distill/models/vqvae.py`, `src/distill/training/loop.py`, `src/distill/models/persistence.py` -- existing patterns
- [RQ-VAE-Transformer (CVPR 2022)](https://arxiv.org/abs/2203.01941) -- autoregressive prior over residual quantized codes
- [Jukebox (OpenAI)](https://openai.com/index/jukebox/) -- multi-scale VQ-VAE with autoregressive Transformer priors for audio

### Secondary (MEDIUM confidence)
- [VQ-VAE-2 implementations](https://github.com/vvvm23/vqvae-2) -- practical examples of autoregressive priors over discrete codes
- [GPT-style Transformer in PyTorch](https://medium.com/@hannojacobs/building-a-gpt-style-autoregressive-transformer-from-scratch-in-pytorch-using-the-147fabebfe90) -- decoder-only implementation pattern
- [Perplexity evaluation for language models](https://huggingface.co/docs/transformers/perplexity) -- standard perplexity computation reference
- [Causal Transformer Decoder](https://github.com/alex-matton/causal-transformer-decoder) -- PyTorch autoregressive decoder implementation

### Tertiary (LOW confidence)
- Memorization threshold values (2.0/3.0/5.0) -- based on reasoning about codebook entropy, not empirically validated. Flag for tuning during implementation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Pure PyTorch, no new dependencies, well-understood autoregressive modeling
- Architecture: HIGH -- GPT-style decoder-only Transformer is the standard approach for autoregressive priors over discrete codes, used by VQ-VAE-2, Jukebox, and RQ-VAE
- Code extraction: HIGH -- Simple forward pass through frozen VQ-VAE, using existing model infrastructure
- Memorization detection: MEDIUM -- Adaptive thresholds are reasoned but not empirically validated; the mechanism (perplexity monitoring) is standard
- Model bundling: HIGH -- Extending the existing v2 .distill format is straightforward; pattern mirrors existing save/load code
- Pitfalls: HIGH -- Well-documented in autoregressive modeling literature

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (stable domain; PyTorch Transformer API is mature)
