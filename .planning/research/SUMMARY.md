# Project Research Summary

**Project:** Distill v1.1 — RVQ-VAE + Autoregressive Prior
**Domain:** Small-dataset discrete audio generation with code-level manipulation
**Researched:** 2026-02-21
**Confidence:** HIGH (stack, architecture), MEDIUM-HIGH (features, pitfalls)

## Executive Summary

Distill v1.1 replaces the continuous Gaussian VAE with a Residual Vector Quantization VAE (RVQ-VAE) and adds an autoregressive prior for generation plus a discrete code manipulation UI. The existing convolutional encoder and decoder backbone are preserved almost entirely — the only structural change is replacing the `fc_mu/fc_logvar` bottleneck with a `Conv1x1` projection into the `vector-quantize-pytorch` `ResidualVQ` layer, which operates on spatial embeddings `[B, H*W, D]` rather than a single global latent vector. This is a surgical replacement, not a ground-up rewrite. One new dependency (`vector-quantize-pytorch>=1.27.0`) covers the entire VQ/RVQ implementation; the autoregressive prior uses `nn.TransformerEncoder` from existing PyTorch; all audio I/O, preprocessing, and export infrastructure is unchanged.

The recommended approach is strict two-stage training: train the VQ-VAE to convergence first, then train a small autoregressive Transformer prior on the frozen code sequences. This is the proven pattern from VQ-VAE, VQ-VAE-2, and SoundStream. The critical practical insight is that codebook sizing must scale with dataset size — the literature defaults (1024 codes, 8 quantizers) are appropriate for millions of files, not 5-500. For this project's target scale, codebooks of 64-256 entries with 2-4 quantizer levels are the correct starting point. The "sound DNA editor" concept — encoding audio to discrete codes, visualizing and editing the code grid, then decoding back — is genuinely novel: no existing tool (RAVE, EnCodec, Jukebox, SoundStream) exposes discrete codes for direct user manipulation.

The primary risks are codebook collapse (the entire codebook converges to a handful of codes), prior memorization (the prior reproduces training sequences verbatim rather than generalizing), and audio quality regression vs. v1.0. All three are preventable through mitigation built into the initial architecture: k-means codebook initialization, EMA updates with dead code reset, aggressive prior regularization, and codebook utilization monitoring from day one. The integration surface is large (186-file codebase with deep continuous-VAE coupling) but the architecture research has mapped every integration point. The correct approach is to build new modules in parallel (`vqvae.py`, `prior.py`, `vq_losses.py`) rather than modifying existing code in-place.

## Key Findings

### Recommended Stack

The stack addition is minimal by design: one package (`vector-quantize-pytorch>=1.27.0`) delivers the entire RVQ implementation including k-means initialization, EMA codebook updates, dead code reset, straight-through estimator, and the `ResidualVQ.get_codes_from_indices()` method needed for generation and code manipulation. Its transitive dependencies (`einops>=0.8.0`, `einx[torch]>=0.1.3`) install automatically. The autoregressive prior requires no new packages — `nn.TransformerEncoder` with a causal mask covers the architecture. All audio processing, visualization (matplotlib), and export (soundfile) libraries are already in place and unchanged.

**Core technologies:**
- `vector-quantize-pytorch>=1.27.0`: RVQ layer (k-means init, EMA updates, dead code reset) — de facto standard, MIT licensed, 3500+ stars, actively maintained (v1.27.21 released 2026-02-12)
- `einops>=0.8.0` and `einx[torch]>=0.1.3`: Transitive dependencies auto-installed with the above; no separate install needed
- `torch.nn.TransformerEncoder` (bundled with PyTorch): Autoregressive prior — no new dependency, well-supported on CUDA/MPS/CPU
- `matplotlib` (already installed): Codebook utilization heatmaps and code frequency histograms — no Plotly needed

**What NOT to add:** PyTorch Lightning, WandB/MLflow, librosa, pedalboard, auraloss, DAC/EnCodec/RAVE as components, Plotly, Hydra — each rejected for adding complexity without proportional benefit to a local creative tool at this scale. See STACK.md "What NOT to Add" section for detailed rationale.

### Expected Features

The feature landscape divides cleanly into P1 (required to prove the VQ-VAE + code manipulation value proposition) and P2 (elevates from functional to compelling). Everything depends on a trained model with populated codebooks, so RVQ-VAE training must come first.

**Must have (P1 — v1.1 core):**
- RVQ-VAE training with codebook health monitoring — the foundation; everything else depends on it
- Encode audio to discrete codes — bridge from audio to the code domain
- Decode codes to audio — bridge back; required for all code manipulation
- Code visualization (timeline grid: rows = quantizer levels, columns = time positions) — the "aha moment"; gates all creative operations
- Basic code editing (cell-level index change and code swapping between audio files) — minimum viable manipulation; the headline feature "mix the DNA of two sounds"
- Autoregressive prior with temperature control — VQ-VAE cannot generate novel material without a prior
- Updated model persistence (version 2 format bundling VQ-VAE + prior) — save/load trained models

**Should have (P2 — v1.1.x after core works):**
- Per-layer code manipulation (coarse/structure vs. fine/texture) with layer labels
- Codebook entry browser with audio preview (click a code, hear what it sounds like)
- Codebook usage heatmap (post-training diagnostic for model health)
- Code blending in embedding space (smoother than index swapping)
- Top-k / nucleus sampling controls for generation
- Code sequence templates and patterns

**Defer (v1.2+):**
- Conditional generation / audio continuation — requires mature prior
- Encode-Edit-Decode as single fully integrated workflow — polish requiring solid P1+P2
- Code embedding space sliders — only add if users miss continuous control from v1.0

**Anti-features to reject explicitly:** Real-time code editing (Griffin-Lim decode is too slow), text-conditioned generation (massive dependency, unreliable on personal datasets), importing codebooks from other models (codebooks are architecture-coupled and not interchangeable), arbitrary-length prior generation (degrades rapidly on small datasets), continuous PCA sliders alongside discrete codes (two paradigms create confusion).

### Architecture Approach

The architecture is a drop-in bottleneck replacement: the mel spectrogram pipeline and convolutional backbone stay identical. The encoder no longer flattens to a global vector — it outputs a 2D feature map `[B, D, H, W]` that is reshaped to a spatial embedding sequence `[B, H*W, D]` and quantized position-by-position through stacked RVQ codebooks. The decoder receives the quantized spatial embeddings `[B, D, H, W]` in place of the projected global vector. Generation uses a lightweight causal Transformer prior (4-6 layers, 256-dim, 4 heads, ~1-5M parameters) trained on frozen code indices, producing new code sequences decoded via `get_codes_from_indices()` through the existing decoder and Griffin-Lim pipeline.

**Major components:**
1. `models/vqvae.py` (`ConvVQVAE`) — top-level model: `VQEncoder` (same conv backbone + Conv1x1 projection instead of flatten+linear), `ResidualVQ` wrapper, `VQDecoder` (Conv1x1 + same conv backbone); forward returns `(recon_mel, indices, commit_loss)` — replaces `models/vae.py`
2. `models/prior.py` (`AutoregressivePrior`) — causal Transformer over flattened code sequences `[B, positions * num_quantizers]`; trained in stage 2 on frozen VQ-VAE code indices
3. `models/quantizer.py` — thin wrapper around `vector_quantize_pytorch.ResidualVQ` with per-level codebook monitoring (utilization, perplexity, dead code count)
4. `models/losses.py` (rewrite) — `vqvae_loss(recon, mel, commit_loss, commit_weight)`: MSE reconstruction + commitment loss; KL/free-bits/annealing logic removed entirely
5. `training/prior_loop.py` — stage 2: freeze VQ-VAE, encode training set to code indices, teacher-forced next-token prediction, cross-entropy loss
6. `inference/codes.py` (`CodeManipulator`) — encode/decode, code swap/copy/randomize/blend operations
7. `ui/tabs/codes_tab.py` — Gradio code editor tab: Plotly heatmap for visualization + DataFrame/dropdowns for editing; replaces PCA slider controls

**Unchanged components (architecture-independent):** `audio/spectrogram.py`, `audio/io.py`, `data/dataset.py`, `inference/export.py`, `inference/spatial.py`, `inference/stereo.py`, `hardware/`, `audio/filters.py`, `inference/quality.py`

**Key simplification vs. v1.0:** VQ-VAE eliminates the entire KL-balancing complexity — no `kl_warmup_fraction`, `kl_weight_max`, `free_bits`, annealing schedule, or posterior collapse monitoring. Commitment loss has one weight parameter and is much simpler to tune.

### Critical Pitfalls

1. **Codebook collapse on small datasets** — most or all codebook entries go unused (utilization observed as low as 14.7% even on large datasets; far worse on 5-50 files). Prevention: scale `codebook_size` to dataset size (64 for 5-20 files, 128 for 20-100, 256 for 100-500), enable `kmeans_init=True` and `threshold_ema_dead_code=2`, monitor utilization per RVQ level every epoch. If utilization drops below 30%, stop and re-initialize with smaller codebook. This must be built into the quantizer from day one.

2. **Prior memorization of training sequences** — Transformers trivially memorize 5-500 code sequences (4,700-47,000 total tokens) and reproduce them verbatim. Warning sign: validation perplexity below 1.5. Prevention: tiny prior models (2 layers/128-dim for <50 files; 4 layers/256-dim for 50-200 files), dropout 0.3-0.5, aggressive early stopping, 5-10% token corruption during training. Healthy prior shows validation perplexity between 2.0 and 20.0.

3. **Cascading RVQ collapse across quantizer levels** — when the first quantizer captures the full signal, residual levels receive near-zero input and collapse. Unique to RVQ, not present in single-level VQ-VAE. Prevention: start with `num_quantizers=2` for tiny datasets, enable `quantize_dropout=True`, monitor per-level utilization separately. If level N has <10% utilization while level 1 has >90%, reduce num_quantizers.

4. **Audio quality regression vs. v1.0** — quantization artifacts in mel spectrograms become phase-incoherent noise via Griffin-Lim. Prevention: start with `commitment_weight=0.25` (not the library default of 1.0), use `codebook_dim=8-16` (lower-dim projection reduces quantization error), build a side-by-side A/B comparison with v1.0 output as a gating criterion before shipping.

5. **Training loop integration breakage** — the existing codebase is deeply coupled to the continuous VAE interface (`(recon, mu, logvar)` tuples, KL annealing, PCA latent analysis, `model.sample()` generation path). Prevention: build new modules in parallel without touching existing code, define the new VQ-VAE public API as type stubs before implementation, map the full integration surface before writing any VQ code.

## Implications for Roadmap

Based on the dependency graph from FEATURES.md and the build order from ARCHITECTURE.md, research implies a 4-phase structure with a clear critical path. All downstream work depends on Phase 1 producing a stable, well-utilized RVQ-VAE model.

### Phase 1: RVQ-VAE Core + Training Foundation

**Rationale:** Everything downstream — prior training, code manipulation, generation, UI — depends on having a trained RVQ-VAE model with healthy, well-utilized codebooks. No other work can proceed without this. Integration surface mapping (Pitfall 5) must happen here before any code is written.

**Delivers:** Trainable `ConvVQVAE` model; per-level codebook health monitoring (utilization, perplexity, dead code count) visible in training UI; updated model persistence (version 2 format with VQ-specific metadata); `VQVAETrainingConfig` replacing continuous VAE config; `vqvae_loss()` replacing `vae_loss()`; A/B reconstruction quality comparison with v1.0 as gating criterion.

**Addresses (from FEATURES.md P1):** RVQ-VAE training with codebook health monitoring, model persistence (v2 format), training progress.

**Avoids (from PITFALLS.md):** Codebook collapse (k-means init + dead code reset + dataset-adaptive sizing from day one); cascading RVQ collapse (quantize_dropout + per-level monitoring); commitment loss instability (commitment_weight=0.25, EMA decay 0.8-0.95 for small data); audio quality regression (A/B gating criterion); checkpoint format breakage (version 2 format with VQ-specific metadata bundled).

**Research flag:** Small-dataset codebook sizing defaults are extrapolated from large-scale literature. Empirical validation across 5/50/500 file regimes is needed before defaults are finalized. Consider a short research-phase here.

### Phase 2: Autoregressive Prior + Generation Pipeline

**Rationale:** Prior training requires a frozen, converged VQ-VAE (Phase 1 must complete first). Generation requires both the VQ-VAE decoder and the prior. This phase delivers the core generation capability that replaces the v1.0 `z ~ N(0,1)` generation path and completes the MVP feature set.

**Delivers:** Trained `AutoregressivePrior`; prior-based generation in the generate tab; temperature/top-k/nucleus sampling controls; multi-chunk generation via existing overlap-add; prior model bundled into v2 model format; prior training configuration (`PriorTrainingConfig`).

**Addresses (from FEATURES.md P1):** Autoregressive generation, temperature control for generation.

**Avoids (from PITFALLS.md):** Prior memorization (tiny model + heavy dropout + token corruption + validation perplexity monitoring); prior architecture mismatch (match positional encoding to actual sequence length, not 1024+); sequence length mismatch (chunked generation with context carryover between chunks).

**Research flag:** Prior architecture for <20 file datasets needs a decision: STACK.md argues for LSTM (better inductive bias, fewer parameters, lower overfitting risk at tiny scale); ARCHITECTURE.md proposes Transformer. This conflict must be resolved before implementation begins. Recommend LSTM baseline test.

### Phase 3: Code Manipulation UI + Encode/Decode Operations

**Rationale:** Encode/decode infrastructure depends only on Phase 1 (trained model); it does not require the prior. This phase is technically parallelizable with Phase 2 but is ordered after to ensure the model architecture is stable before building UI on top of it. This phase delivers the primary differentiator of the product.

**Delivers:** `inference/codes.py` (`CodeManipulator`) with encode/decode/swap/copy/randomize/blend operations; Gradio codes tab with Plotly heatmap visualization and editing controls; code swapping between two audio files; basic code editing (cell-level index change); audio preview on every decode; undo/redo stack for code edits.

**Addresses (from FEATURES.md P1):** Encode audio to discrete codes, decode codes to audio, code visualization (timeline grid), basic code editing, code swapping between files.

**Avoids (from PITFALLS.md):** Raw index UI (Pitfall 10) — perceptual labeling (spectral centroid/energy analysis of code entries) and per-code audio preview must be included from the start, not retrofitted; code editor without undo — implement undo/redo stack from day one to prevent destructive edits.

**Research flag:** No established prior art for interactive code grid editing in Gradio. The Plotly heatmap + DataFrame hybrid approach (recommended by FEATURES.md) should be prototyped early in this phase before committing to the full editing layer. Standard patterns otherwise.

### Phase 4: P2 Features + Polish

**Rationale:** All P1 features from Phases 1-3 must be stable before adding P2. This phase is intentionally isolated from the critical path — if Phase 1-3 take longer than expected, these features can slip to v1.1.x without blocking the milestone.

**Delivers:** Per-layer code manipulation with labeled layers (Structure/Timbre/Detail); codebook entry browser with audio preview per code; codebook usage heatmap (post-training diagnostic); code blending in embedding space; top-k/nucleus sampling controls in generation tab; CLI updates for VQ training, prior training, and code operations; UX polish (plain-language metric labels, health indicators, jargon-free sampling controls).

**Addresses (from FEATURES.md P2):** Per-layer manipulation, codebook entry browser, usage heatmap, code blending, top-k/nucleus sampling, code sequence templates.

**Avoids (from PITFALLS.md):** Temperature/sampling UX pitfall — map temperature to "Creativity" slider, top-p to "Diversity", top-k to "Focus"; missing UX labeling — codebook utilization shown as green/yellow/red health indicator with plain-language explanation, not raw percentage.

**Research flag:** Per-layer semantic labeling (Structure/Timbre/Detail) is based on Interspeech 2025 research on codec interpretability but attribute entanglement is substantial. Messaging must set appropriate expectations. Standard patterns for the rest.

### Phase Ordering Rationale

- Phase 1 is the only phase with no dependencies on other v1.1 work. It is the critical path bottleneck.
- Phase 2 (prior) is ordered before Phase 3 (code manipulation) because prior training takes real wall-clock time that can be happening while Phase 3 is designed. However Phase 3 can start immediately after Phase 1 if prior training is taking longer than expected — it does not depend on the prior.
- Phase 4 is intentionally isolated from the critical path. If Phases 1-3 take longer than expected, Phase 4 features slip to v1.1.x without blocking the milestone.
- The KL annealing/free-bits complexity removed in Phase 1 is a net simplification vs. v1.0: one `commitment_weight` parameter replaces `kl_warmup_fraction`, `kl_weight_max`, `free_bits`, and posterior collapse monitoring. Phase 1 may be less complex than the equivalent v1.0 training loop work despite the new RVQ mechanics.
- The prior model is independent of code manipulation (Phase 3). The generate tab (prior) and the codes tab (manipulation) are parallel features at the UI level, both enabled by a trained VQ-VAE model.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Empirical codebook sizing defaults for 5/50/500 file datasets — literature values are from large-scale systems; optimal small-dataset defaults require testing across the actual target data regimes before exposing in UI.
- **Phase 2:** LSTM vs. Transformer prior for very small datasets (<20 files) — STACK.md and ARCHITECTURE.md disagree; this architectural decision must be resolved before implementation begins.
- **Phase 3:** Gradio code grid editor — no established prior art; Plotly heatmap + DataFrame hybrid approach needs early prototyping to confirm feasibility.

Phases with standard patterns (skip research-phase):
- **Phase 1 (conv backbone changes):** Replacing `fc_mu/fc_logvar` with `Conv1x1 + ResidualVQ` is well-documented in SoundStream/EnCodec literature and the `vector-quantize-pytorch` API is fully verified.
- **Phase 2 (prior training loop):** Teacher-forced next-token prediction is standard; the `train_prior_epoch` pattern in ARCHITECTURE.md is complete and directly implementable.
- **Phase 4 (sampling controls):** Temperature, top-k, nucleus sampling are standard; UX mapping to "Creativity/Focus/Diversity" is a known pattern.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | `vector-quantize-pytorch` v1.27.21 verified on PyPI (2026-02-12); API confirmed from GitHub source; transitive dependencies verified from setup.py. Single new package covers the entire VQ requirement. MPS historical issue (#55) is closed; current PyTorch 2.4+ likely resolves it but requires a smoke test. |
| Features | MEDIUM-HIGH | RVQ-VAE architecture patterns are well-established in research literature. Code manipulation UI paradigm is genuinely novel with limited prior art — UX design recommendations are reasoned extrapolations, not validated by user studies. Small-dataset codebook sizing thresholds are extrapolated from large-scale findings and must be treated as starting points. |
| Architecture | HIGH | Core VQ patterns (straight-through estimator, EMA updates, dead code reset, two-stage training, spatial embeddings) verified from VQ-VAE, VQ-VAE-2, and SoundStream papers. Integration point mapping is based on direct source analysis of the 186-file codebase. Prior model sizing is a reasonable starting point requiring empirical tuning. |
| Pitfalls | MEDIUM-HIGH | Codebook collapse, prior memorization, cascading RVQ collapse, and commitment loss instability are well-documented (ICCV 2025, Interspeech 2025, lucidrains issue tracker). Small-dataset-specific thresholds (e.g., "30% utilization = crisis") are extrapolated and should be treated as initial benchmarks, not hard rules. |

**Overall confidence:** HIGH for stack and architecture decisions; MEDIUM-HIGH for feature design and pitfall thresholds.

### Gaps to Address

- **Empirical codebook sizing defaults:** The dataset-to-codebook-size mapping (64 codes for 5-20 files, 128 for 20-100, 256 for 100-500) is a reasoned extrapolation. During Phase 1, test defaults on datasets of 5, 20, 50, and 200 files across at least 2 audio domains (ambient, percussive) and adjust before exposing as UI defaults.
- **LSTM vs. Transformer prior decision:** STACK.md argues for LSTM at small scale (parameter efficiency, inductive bias, lower overfitting risk); ARCHITECTURE.md proposes Transformer (better long-range dependencies, natural sequence handling). This conflict must be resolved before Phase 2 begins. Recommended resolution: implement LSTM baseline, measure validation perplexity on a 20-file test dataset, compare to Transformer; use whichever overfits less.
- **MPS compatibility of `vector-quantize-pytorch`:** The MPS crash issue (lucidrains #55) is closed, but the fix is from the PyTorch 2.0 era. The current project venv has `torch==2.4.0+cu118` which should be fine, but MPS-specific smoke testing is required early in Phase 1 before assuming compatibility.
- **Audio quality floor for small codebooks:** With 64 entries per level, there is a fundamental quality ceiling. Research does not establish what codebook size is required to match v1.0 continuous VAE reconstruction quality on the target audio types. This A/B comparison must be performed as part of Phase 1 gating before proceeding to Phase 2.
- **Prior quality floor on <20 file datasets:** For very small datasets, the prior may remain overfit regardless of regularization, producing only slight variations of training data. An honest assessment of minimum viable dataset size for "useful" generation is needed before communicating expected behavior to users. For datasets below this threshold, shuffle-based and template-based code generation may be more honest than autoregressive generation.

## Sources

### Primary (HIGH confidence)
- `vector-quantize-pytorch` PyPI (v1.27.21, 2026-02-12) — package version, API, and dependency verification
- `vector-quantize-pytorch` GitHub (lucidrains) — ResidualVQ API, dead code reset, quantize_dropout, k-means init
- SoundStream paper (arXiv 2107.03312) — RVQ architecture, k-means codebook initialization, EMA codebook updates
- VQ-VAE paper (arXiv 1711.00937) — two-stage training, commitment loss, straight-through estimator
- VQ-VAE-2 paper (NeurIPS 2019) — hierarchical VQ, autoregressive priors over code sequences, two-stage pattern
- einops PyPI (v0.8.2, 2026-01-26) and einx PyPI (v0.3.0) — transitive dependency verification

### Secondary (MEDIUM confidence)
- ERVQ: Enhanced Residual Vector Quantization (arXiv 2410.12359) — codebook utilization 14.7% baseline; improvement strategies
- Addressing Representation Collapse in Vector Quantized Models (ICCV 2025) — dead code prevention techniques
- Bringing Interpretability to Neural Audio Codecs (Interspeech 2025) — per-layer semantic content (structure/timbre/detail) and entanglement findings
- Spectrogram Patch Codec (arXiv 2509.02244) — VQ-VAE on mel spectrograms; validates mel-based approach
- High-Fidelity Audio Compression with Improved RVQGAN (arXiv 2306.06546) — codebook_dim=8 optimal for audio RVQ
- lucidrains issue tracker (#27, #33, #55, #69, #109) — commitment loss instability, MPS crash, RVQ gradient flow issues
- Optimizing Deeper Transformers on Small Datasets (ACL 2021) — LSTM vs. Transformer behavior on small data

### Tertiary (LOW confidence, needs validation during implementation)
- Codebook size scaling table (64/128/256 for 5-20/20-100/100-500 files) — extrapolated from large-scale literature; must be validated empirically on actual dataset sizes
- Prior perplexity thresholds (healthy: 2.0-20.0; memorized: <1.5) — reasonable estimates, not validated against this specific application
- MPS compatibility with current PyTorch version — closed issue from PyTorch 2.0 era; assumed resolved but requires smoke test

---
*Research completed: 2026-02-21*
*Ready for roadmap: yes*
