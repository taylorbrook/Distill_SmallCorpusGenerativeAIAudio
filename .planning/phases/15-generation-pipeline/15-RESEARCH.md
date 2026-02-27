# Phase 15: Generation Pipeline - Research

**Researched:** 2026-02-27
**Domain:** Autoregressive prior sampling, VQ-VAE decode-from-codes, multi-chunk audio stitching, Gradio UI, Typer CLI
**Confidence:** HIGH

## Summary

Phase 15 wires the trained autoregressive prior (Phase 14) into a generation pipeline where the prior autoregressively samples code sequences, the VQ-VAE decodes those codes to mel spectrograms, and the existing spectrogram-to-waveform path produces audio. The entire v1.0 generation pipeline (continuous latent space, slider-controlled VAE sampling) is irrelevant -- this phase builds a new prior-based generation path from scratch.

The core technical challenge is implementing autoregressive sampling with temperature, top-k, and top-p controls on the `CodePrior` model, then converting the sampled code sequences back through the VQ-VAE decode path (`codes_to_embeddings` -> `decode` -> `mel_to_waveform`). Multi-chunk generation requires generating multiple full-length code sequences and stitching the resulting audio with overlap-add crossfade. The UI needs to be substantially reworked from the v1.0 slider-based generate tab to a prior-based tab with sampling controls and a duration slider.

**Primary recommendation:** Build a standalone `generate_from_prior()` function in `inference/generation.py` that accepts a `LoadedVQModel` (with prior), sampling parameters, and duration -- completely independent of the v1.0 `GenerationPipeline` class. Replace the v1.0 generate tab contents with prior-based controls. Add a new CLI command (`distill generate-prior` or extend `distill generate` to detect VQ-VAE models).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Default temperature: 1.0 (neutral)
- Temperature slider range: 0.1 -- 2.0
- Default top-p: 0.9 (nucleus sampling) -- adapts to context, good for small codebooks (64--256)
- Top-k: off by default
- Top-p slider range: 0 -- 1.0 (0 = disabled)
- Top-k slider range: 0 -- 512 (0 = disabled)
- User controls output duration via a duration slider (not chunk count)
- Default duration: ~10 seconds (2-3 chunks stitched)
- Maximum duration: 30 seconds
- Crossfade/overlap amount is user-configurable (advanced control exposed in UI)
- Leave v1.0 continuous generation behind -- not relevant, will not be used
- Generate tab shows prior-based controls only (temperature, top-k, top-p, duration)
- No sub-tabs or mode switching needed
- Progress bar with chunk counter: "Generating chunk 2/4..."
- Simple and clear, no chunk-by-chunk audio preview

### Claude's Discretion
- Crossfade default overlap amount and range
- Error state handling (no prior trained, generation fails mid-chunk)
- Exact slider step sizes and UI layout

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GEN-02 | User can generate new audio from the trained prior with temperature control | Autoregressive sampling function with temperature scaling on logits before softmax. Prior model's `forward()` returns logits that can be divided by temperature. |
| GEN-03 | User can control generation with top-k and nucleus (top-p) sampling | Standard top-k (zero out logits below k-th largest) and nucleus sampling (cumulative probability cutoff) applied to logits before sampling. Well-established algorithms. |
| GEN-04 | Prior generates multi-chunk audio with overlap-add stitching (existing pipeline) | Generate N independent code sequences, decode each to mel via VQ-VAE, convert to waveform, then use existing `crossfade_chunks()` from `inference/chunking.py` for Hann-windowed overlap-add stitching. |
| UI-04 | Generate tab updated for prior-based generation (temperature, top-k, top-p controls) | Replace v1.0 slider/preset/blend UI with temperature, top-k, top-p sliders, duration slider, crossfade overlap slider, seed input, and generate button. Keep export section. |
| CLI-04 | CLI supports generation from trained prior with sampling controls | Add `--temperature`, `--top-k`, `--top-p` flags to existing `distill generate` command or create a parallel command. Follow `cli/train_prior.py` patterns. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | existing | Autoregressive sampling, softmax, multinomial | Already in project; sampling is pure tensor ops |
| vector-quantize-pytorch | existing | `get_output_from_indices()` for code-to-embedding decode | Already used by QuantizerWrapper; provides the decode-from-indices path |
| torchaudio | existing | Griffin-Lim waveform reconstruction via AudioSpectrogram | Already in project; `mel_to_waveform()` is the standard decode path |
| gradio | existing | UI controls (sliders, buttons, progress) | Already used for generate tab |
| typer | existing | CLI command and flag definitions | Already used for all CLI commands |
| rich | existing | CLI progress display | Already used in train-prior CLI |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | existing | Audio array manipulation for crossfade stitching | Waveform concatenation and overlap-add |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Independent chunk generation + waveform crossfade | Context-conditioned generation (seed next chunk with end of previous) | Context conditioning would produce more coherent multi-chunk audio, but adds complexity and the prior's seq_len is fixed per chunk. Context conditioning is a future enhancement (DEFER-04). |
| Griffin-Lim for waveform | HiFi-GAN vocoder | Higher quality but adds a large dependency and training requirement. Griffin-Lim is sufficient for the current quality tier. |

## Architecture Patterns

### Recommended Code Organization
```
src/distill/
├── inference/
│   └── generation.py     # Add generate_from_prior() alongside existing v1.0 pipeline
├── models/
│   └── prior.py          # Add sample() method to CodePrior class
├── ui/tabs/
│   └── generate_tab.py   # Replace v1.0 slider UI with prior-based controls
├── cli/
│   └── generate.py       # Extend with --temperature, --top-k, --top-p flags
└── ui/
    └── state.py          # Add loaded_vq_model field for VQ-VAE model state
```

### Pattern 1: Autoregressive Sampling with Temperature/Top-k/Top-p
**What:** Token-by-token generation from the CodePrior, applying temperature scaling, top-k filtering, and nucleus (top-p) sampling to logits at each step.
**When to use:** Every generation call.
**Confidence:** HIGH -- this is the standard GPT-style sampling algorithm, well-documented.

```python
@torch.no_grad()
def sample_prior(
    prior: CodePrior,
    num_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Autoregressively sample a code sequence from the prior.

    Returns shape [1, num_tokens] of code indices.
    """
    prior.eval()

    # Start with a random first token
    generated = torch.randint(0, prior.codebook_size, (1, 1), device=device)

    for _ in range(num_tokens - 1):
        logits = prior(generated)          # [1, T, codebook_size]
        next_logits = logits[:, -1, :]     # [1, codebook_size] -- last position

        # Temperature scaling
        next_logits = next_logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < threshold] = float('-inf')

        # Nucleus (top-p) sampling
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            # Scatter back to original indexing
            next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample from filtered distribution
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        generated = torch.cat([generated, next_token], dim=1)

    return generated  # [1, num_tokens]
```

### Pattern 2: Code Sequence to Audio Decode Path
**What:** Convert sampled code indices back to audio through the VQ-VAE decode path.
**When to use:** After prior sampling to produce audible output.
**Confidence:** HIGH -- all components already exist.

```python
def codes_to_audio(
    vqvae: ConvVQVAE,
    spectrogram: AudioSpectrogram,
    code_indices: torch.Tensor,  # [1, seq_len, num_quantizers]
    spatial_shape: tuple[int, int],  # (H, W) from VQ-VAE encoder
    mel_shape: tuple[int, int],  # (n_mels, time_frames) for output cropping
) -> np.ndarray:
    """Decode code indices through VQ-VAE to audio waveform."""
    # 1. Codes -> quantized embeddings
    quantized = vqvae.codes_to_embeddings(code_indices, spatial_shape)
    # 2. Quantized embeddings -> mel spectrogram
    mel = vqvae.decode(quantized, target_shape=mel_shape)
    # 3. Mel -> waveform via Griffin-Lim
    wav = spectrogram.mel_to_waveform(mel)
    return wav.squeeze().numpy().astype(np.float32)
```

### Pattern 3: Multi-Chunk Generation with Waveform Crossfade
**What:** Generate multiple independent code sequences (one per chunk), decode each to audio, then stitch with existing `crossfade_chunks()`.
**When to use:** When duration > 1 chunk (~1 second).
**Confidence:** HIGH -- reuses existing `crossfade_chunks()` from `inference/chunking.py`.

```python
def generate_multi_chunk(
    prior, vqvae, spectrogram, num_chunks, sampling_params,
    overlap_samples, device, seed,
):
    """Generate multi-chunk audio with crossfade stitching."""
    chunks = []
    for i in range(num_chunks):
        torch.manual_seed(seed + i)
        codes_flat = sample_prior(prior, num_tokens, **sampling_params, device=device)
        codes_3d = unflatten_codes(codes_flat, num_quantizers)
        audio_chunk = codes_to_audio(vqvae, spectrogram, codes_3d, spatial_shape, mel_shape)
        chunks.append(audio_chunk)

    # Stitch with existing crossfade function
    return crossfade_chunks(chunks, overlap_samples=overlap_samples)
```

### Pattern 4: Generate Tab UI (Prior-Based)
**What:** Replace v1.0 slider/preset/blend UI with prior sampling controls.
**When to use:** Building the new generate tab.
**Confidence:** HIGH -- straightforward Gradio component layout.

The generate tab should have:
1. Temperature slider (0.1 -- 2.0, default 1.0, step 0.05)
2. Top-p slider (0 -- 1.0, default 0.9, step 0.05)
3. Top-k slider (0 -- 512, default 0, step 1)
4. Duration slider (1 -- 30 seconds, default 10, step 1)
5. Crossfade overlap slider (advanced, in ms)
6. Seed input with randomize button
7. Generate button with progress bar
8. Audio output player
9. Export section (format, sample rate, bit depth, metadata -- reuse from v1.0)

### Anti-Patterns to Avoid
- **Modifying v1.0 GenerationPipeline:** The existing `GenerationPipeline` class is designed for continuous VAE with latent vectors. Do not try to adapt it for prior-based generation. Build a parallel function.
- **Continuous mel synthesis for prior output:** The v1.0 `synthesize_continuous_mel()` uses latent trajectory interpolation, which is irrelevant for prior-based generation. Prior generates discrete codes, not continuous latent vectors.
- **Token-by-token KV caching:** While KV caching would speed up autoregressive sampling, the prior model uses `nn.TransformerEncoder` which doesn't natively support KV caching. For the small sequence lengths in this project (~18 tokens per chunk with 3 quantizers and 6 spatial positions), full-sequence forward passes are fast enough.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Waveform overlap-add stitching | Custom crossfade | `crossfade_chunks()` from `inference/chunking.py` | Already handles Hann windowing, overlap-add, and edge cases |
| Code indices to embeddings | Manual codebook lookup | `vqvae.codes_to_embeddings()` + `quantizer.get_output_from_indices()` | Already exists and handles multi-level RVQ properly |
| Mel to waveform | Custom Griffin-Lim | `spectrogram.mel_to_waveform()` | Already handles InverseMelScale (CPU forced) + Griffin-Lim |
| CLI bootstrapping | Custom config/device | `cli.bootstrap()` | Shared by all CLI commands |
| Progress display | Custom progress | Gradio `gr.Progress` for UI, Rich console for CLI | Standard patterns already in codebase |

**Key insight:** The VQ-VAE decode path (codes -> embeddings -> mel -> waveform) is fully implemented. The only missing piece is the autoregressive sampling function that produces code sequences.

## Common Pitfalls

### Pitfall 1: Spatial Shape Mismatch in codes_to_embeddings
**What goes wrong:** `codes_to_embeddings()` requires the spatial shape `(H, W)` to reshape the flat sequence back to a 2D spatial map. If the wrong spatial shape is provided, the decode produces garbage.
**Why it happens:** The spatial shape depends on the mel spectrogram dimensions and the 16x downsampling factor. It's stored during the VQ-VAE forward pass (`_spatial_shape`) but not persisted in the saved model.
**How to avoid:** Compute spatial shape from spectrogram config: `H = ceil(n_mels/16)`, `W = ceil(time_frames/16)` where `time_frames = sample_rate // hop_length + 1`. For default config (128 mels, 94 time frames): `H=8, W=6`.
**Warning signs:** Generated audio is pure noise or silence despite valid code sequences.

### Pitfall 2: Sequence Length Mismatch Between Prior and VQ-VAE
**What goes wrong:** The prior's `seq_len` is `spatial_positions * num_quantizers` (e.g., `48 * 3 = 144` for H=8, W=6, Q=3). The sampled sequence must be exactly this length, then unflattened to `[1, spatial_positions, num_quantizers]` for decode.
**Why it happens:** The prior sees a flat interleaved sequence, but the VQ-VAE decode expects `[B, seq_len, num_quantizers]`.
**How to avoid:** Use `unflatten_codes()` from `prior.py` to convert `[1, flat_seq_len]` -> `[1, spatial_positions, num_quantizers]`. The `num_quantizers` is available from the loaded model's `prior_config` or `vqvae_config`.
**Warning signs:** Shape errors in `codes_to_embeddings()`.

### Pitfall 3: First Token Initialization
**What goes wrong:** The autoregressive prior is trained with teacher forcing where every input token is a valid code index. If the first token is not representative, it can bias the entire sequence.
**Why it happens:** There's no explicit start-of-sequence token in the training data.
**How to avoid:** Initialize with a single random token from `[0, codebook_size)` and let the prior's distribution take over. The temperature/top-p controls will ensure diversity. Alternatively, generate from a random index and discard the first token's influence by starting the output from position 1.
**Warning signs:** Generated sequences are always similar regardless of seed changes.

### Pitfall 4: Top-k Value Exceeding Codebook Size
**What goes wrong:** If `top_k > codebook_size`, the top-k filter becomes a no-op, which is fine. But if top_k is close to codebook_size (e.g., top_k=200 with codebook_size=64), it's also a no-op.
**Why it happens:** User may not realize the codebook size is small (64 for tiny datasets).
**How to avoid:** Clamp `top_k` to `min(top_k, codebook_size)` silently. No error needed, just ensure it works correctly.
**Warning signs:** Top-k slider has no perceptible effect on output.

### Pitfall 5: Gradio Progress in Generate Handler
**What goes wrong:** Long multi-chunk generation (30 seconds = ~30 chunks) blocks the UI thread without feedback.
**Why it happens:** Gradio handlers run synchronously; without progress updates, the user sees a spinner with no indication of progress.
**How to avoid:** Use `gr.Progress()` with `progress(fraction, desc="Generating chunk 2/4...")` pattern. The generate handler function should accept a `progress` parameter and update it after each chunk.
**Warning signs:** UI appears frozen during generation.

### Pitfall 6: Model Type Detection for CLI generate Command
**What goes wrong:** The existing `distill generate` command assumes a v1.0 VAE model with latent analysis. If a v2 VQ-VAE model is loaded, the slider/preset logic fails.
**Why it happens:** `resolve_model()` calls `load_model()` (v1 only); v2 VQ-VAE models need `load_model_v2()`.
**How to avoid:** Detect model version before loading. The `.distill` file contains a `version` field (1 or 2) and `model_type` field ("vqvae" for v2). Either extend `resolve_model()` to try v2 first, or create a separate CLI command for prior-based generation.
**Warning signs:** `ValueError: Not a valid .distill model file` when loading v2 models with v1 loader.

### Pitfall 7: Batch Normalization in VQ-VAE Decoder During Eval
**What goes wrong:** BatchNorm layers use running statistics in eval mode. If the model was trained with small batches, the running statistics may not be representative.
**Why it happens:** VQ-VAE encoder/decoder use BatchNorm2d layers. During generation, the model is in eval mode with batch size 1.
**How to avoid:** This should work correctly as long as the model was trained properly (BatchNorm accumulates running stats during training). Just ensure `model.eval()` is called before generation.
**Warning signs:** Generated audio quality is noticeably worse than reconstruction quality during training.

## Code Examples

### Autoregressive Sampling (Complete Implementation)
```python
@torch.no_grad()
def sample_code_sequence(
    prior: CodePrior,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    seed: int | None = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample a complete code sequence from the prior.

    Returns [1, seq_len] flat code indices ready for unflatten_codes().
    """
    if seed is not None:
        torch.manual_seed(seed)

    prior.eval()
    seq_len = prior.seq_len

    # Start with random first token
    generated = torch.randint(0, prior.codebook_size, (1, 1), device=device)

    for step in range(seq_len - 1):
        logits = prior(generated)[:, -1, :]  # [1, codebook_size]

        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            top_vals = torch.topk(logits, k).values
            logits[logits < top_vals[:, -1:]] = float('-inf')

        # Nucleus (top-p)
        if 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)
            # Mask everything after cumulative prob exceeds top_p
            mask = (cum_probs - probs) >= top_p
            sorted_logits[mask] = float('-inf')
            # Unsort
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_token], dim=1)

    return generated  # [1, seq_len]
```

### Full Pipeline: Prior -> Audio
```python
def generate_audio_from_prior(
    loaded: LoadedVQModel,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    duration_s: float = 10.0,
    overlap_samples: int = 2400,
    seed: int | None = None,
    progress_callback: Callable | None = None,
) -> np.ndarray:
    """Generate audio from a trained prior model."""
    prior = loaded.prior
    vqvae = loaded.model
    spectrogram = loaded.spectrogram
    device = loaded.device

    vqvae.eval()
    prior.eval()

    # Compute spatial shape from spectrogram config
    mel_shape = spectrogram.get_mel_shape(48000)  # 1 second of audio
    n_mels, time_frames = mel_shape
    H = math.ceil(n_mels / 16)  # After padding + 4x stride-2
    W = math.ceil(time_frames / 16)
    spatial_shape = (H, W)

    # Number of chunks
    chunk_duration = 1.0  # 1 second per chunk
    num_chunks = math.ceil(duration_s / chunk_duration)

    # Generate chunks
    chunks = []
    actual_seed = seed if seed is not None else random.randint(0, 2**31)

    for i in range(num_chunks):
        if progress_callback:
            progress_callback(i / num_chunks, f"Generating chunk {i+1}/{num_chunks}...")

        # Sample code sequence
        codes_flat = sample_code_sequence(
            prior, temperature, top_k, top_p,
            seed=actual_seed + i, device=device,
        )

        # Unflatten: [1, flat_seq_len] -> [1, spatial_positions, num_quantizers]
        codes_3d = unflatten_codes(codes_flat, prior.num_quantizers)

        # Decode through VQ-VAE
        quantized = vqvae.codes_to_embeddings(codes_3d.to(device), spatial_shape)
        mel = vqvae.decode(quantized, target_shape=mel_shape)
        wav = spectrogram.mel_to_waveform(mel)
        chunks.append(wav.squeeze().numpy().astype(np.float32))

    if progress_callback:
        progress_callback(1.0, "Stitching chunks...")

    # Stitch with crossfade
    if len(chunks) == 1:
        return chunks[0]
    return crossfade_chunks(chunks, overlap_samples=overlap_samples)
```

### Gradio Progress Pattern
```python
def _generate_prior_audio(temperature, top_k, top_p, duration,
                           overlap_ms, seed_val, progress=gr.Progress()):
    """Generate handler with progress updates."""
    # ... validation ...

    num_chunks = math.ceil(duration)
    chunks = []
    for i in range(num_chunks):
        progress((i) / num_chunks, desc=f"Generating chunk {i+1}/{num_chunks}...")
        chunk = generate_single_chunk(...)
        chunks.append(chunk)

    progress(1.0, desc="Stitching audio...")
    audio = crossfade_chunks(chunks, overlap_samples)
    return (sample_rate, audio)
```

## State of the Art

| Old Approach (v1.0) | Current Approach (Phase 15) | Why Changed | Impact |
|---------------------|-----------------------------|-------------|--------|
| Sample z ~ N(0,1), decode through VAE | Autoregressively sample codes from prior, decode through VQ-VAE | VQ-VAE has discrete bottleneck, no continuous latent space to sample from | Completely new generation path |
| Slider-controlled latent vector | Temperature/top-k/top-p sampling controls | Discrete codes don't have PCA-decomposable latent space | Different UX paradigm |
| SLERP latent trajectory for multi-chunk | Independent chunk generation + waveform crossfade | No continuous latent space for interpolation | Simpler stitching, potentially less coherent |
| `GenerationPipeline.generate()` | `generate_audio_from_prior()` (new function) | Completely different model architecture | Parallel implementation |

**Kept from v1.0:**
- `crossfade_chunks()` for waveform stitching (still valid for overlap-add)
- `AudioSpectrogram.mel_to_waveform()` for mel -> audio conversion
- Export infrastructure (format, metadata, sidecar JSON)
- `crossfade_chunks()` for Hann-windowed overlap-add
- CLI patterns (Rich console, JSON output, bootstrap)

## Open Questions

1. **Spatial shape computation vs. stored value**
   - What we know: The VQ-VAE stores `_spatial_shape` during forward pass, but this is not persisted. The shape can be computed from spectrogram config: `H = ceil(n_mels/16)` after padding, `W = ceil(time_frames/16)` after padding.
   - What's unclear: Whether the padding-aware computation matches exactly what the encoder produces. Need to verify: does `ceil(128/16) = 8` and `ceil(94/16)` -- but 94 is padded to 96 first, so `96/16 = 6`. The correct computation is: `H = ceil(n_mels / 16)`, `W = ceil(time_frames / 16)` where the ceiling accounts for padding.
   - Recommendation: Compute from spectrogram config with padding awareness. Verify with a round-trip test: encode real audio, note `_spatial_shape`, compare to computed value.

2. **CLI command structure: extend `distill generate` or new command?**
   - What we know: The existing `distill generate` uses `load_model()` (v1) and has slider/preset/blend logic. VQ-VAE models need `load_model_v2()`.
   - What's unclear: Whether to extend the existing command with model-type detection or create `distill generate-prior`.
   - Recommendation: Extend the existing `distill generate` command. Detect model version by loading the `.distill` file and checking `version` and `model_type`. For v2 VQ-VAE models with a prior, use the prior-based generation path. The `--slider`, `--preset`, and `--blend` flags become irrelevant and should warn/error for VQ-VAE models. The `--temperature`, `--top-k`, `--top-p` flags are new and only apply to prior-based generation.

3. **Crossfade default overlap for prior-generated chunks**
   - What we know: v1.0 uses 2400 samples (50ms at 48kHz) for crossfade. User decided crossfade overlap should be user-configurable in the UI.
   - What's unclear: Optimal default for prior-generated chunks (which may have different boundary characteristics than VAE-generated chunks).
   - Recommendation: Start with 2400 samples (50ms) as default, which is the existing v1.0 default. Allow range 0-200ms (0-9600 samples at 48kHz). Step size of 10ms (480 samples). This gives users enough control to find what sounds best for their particular model.

## Sources

### Primary (HIGH confidence)
- **Codebase analysis** -- `models/prior.py`: CodePrior model with seq_len, num_quantizers, forward() returning logits [B, T, codebook_size]
- **Codebase analysis** -- `models/vqvae.py`: ConvVQVAE with codes_to_embeddings(indices, spatial_shape), decode(quantized, target_shape), QuantizerWrapper.get_output_from_indices()
- **Codebase analysis** -- `inference/chunking.py`: crossfade_chunks() for Hann-windowed overlap-add stitching
- **Codebase analysis** -- `inference/generation.py`: GenerationPipeline (v1.0), GenerationConfig, GenerationResult patterns
- **Codebase analysis** -- `models/persistence.py`: LoadedVQModel with prior field, load_model_v2() with prior reconstruction
- **Codebase analysis** -- `audio/spectrogram.py`: AudioSpectrogram.mel_to_waveform() for Griffin-Lim decode
- **Codebase analysis** -- `ui/tabs/generate_tab.py`: Current v1.0 generate tab layout and handler patterns
- **Codebase analysis** -- `cli/generate.py`: Current CLI command structure and resolve_model() pattern
- **Codebase analysis** -- `cli/train_prior.py`: v1.1 CLI patterns (Rich console, SIGINT, JSON output)
- **Codebase analysis** -- `ui/state.py`: AppState singleton pattern, no LoadedVQModel field yet

### Secondary (MEDIUM confidence)
- GPT-style autoregressive sampling with temperature/top-k/top-p -- standard algorithm documented in Hugging Face transformers, OpenAI papers, and PyTorch tutorials. The implementation pattern is well-established and does not depend on external libraries.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project, no new dependencies needed
- Architecture: HIGH -- all component APIs already exist, the sampling algorithm is standard
- Pitfalls: HIGH -- identified through direct codebase analysis, especially spatial shape handling and sequence length management

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable -- no external dependencies changing)
