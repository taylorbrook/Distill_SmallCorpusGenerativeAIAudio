# Feature Research: VQ-VAE Audio Code Manipulation

**Domain:** RVQ-VAE discrete audio generation with code-level manipulation for creative sound design
**Researched:** 2026-02-21
**Confidence:** MEDIUM-HIGH (architecture patterns well-documented; code manipulation UI is novel territory with less prior art)

## Context: What Exists vs What Is New

This research covers **only** the v1.1 milestone features. The following are **already shipped** in v1.0 and not covered here: training pipeline, mel spectrogram generation, continuous latent space exploration (PCA sliders), multi-format export (WAV/MP3/FLAC/OGG), spatial audio (stereo/binaural), model library, presets, generation history, A/B comparison, Gradio GUI, CLI.

The v1.1 milestone replaces the continuous VAE entirely with RVQ-VAE and adds an autoregressive prior for generation plus a code manipulation UI. The PCA slider interface from v1.0 is deprecated in favor of discrete code manipulation.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that any VQ-VAE audio tool must have. Without these, the discrete code approach adds complexity without delivering value.

| Feature | Why Expected | Complexity | Dependencies on Existing Code | Small-Dataset Notes |
|---------|--------------|------------|-------------------------------|---------------------|
| **Encode audio to discrete codes** | Fundamental VQ-VAE operation. Users need to see their audio as codes before they can manipulate them. Without this, codes are invisible and the tool is just a black box | MEDIUM | Reuses `audio.preprocessing`, `audio.spectrogram` for mel conversion. New encoder model replaces `models.vae.ConvEncoder` | Encoding quality depends on codebook training; with 5-20 files, codebook must be small (128-256 entries per layer) to avoid collapse |
| **Decode codes back to audio** | Complementary to encoding. Users must hear the result of any code manipulation. Round-trip encode-decode is the proof that the system works | MEDIUM | Reuses `audio.spectrogram.mel_to_waveform` (Griffin-Lim) or new neural vocoder. Plugs into existing `inference.generation.GenerationPipeline` for spatial/export | Reconstruction quality is the make-or-break metric. Small codebooks = some quality loss vs continuous VAE. Must be audibly acceptable |
| **Visualize code sequences** | Users cannot manipulate what they cannot see. A code grid/timeline showing which codebook entries are active at each time step is essential for the "sound DNA editor" metaphor | MEDIUM | New UI tab in `ui/tabs/`. Can reuse Gradio components (DataFrame, Plot). No backend dependency beyond the encoder output | Visual must make RVQ hierarchy clear: coarse (layer 1) vs fine (layer N) codes. Color-coding by layer is table stakes |
| **RVQ-VAE training on user's dataset** | Users expect to train a model on their own audio, same as v1.0. The architecture changes but the workflow is the same: import data, train, get a model | HIGH | Replaces `training.loop` internals, `models.vae.ConvVAE`, `models.losses`. Reuses `training.runner`, `training.checkpoint`, `training.config`, `training.metrics`, `training.preview`, `data.dataset` | Critical: codebook collapse is the #1 risk with small datasets. Must implement EMA updates + dead code reset + k-means init. Dataset-adaptive codebook sizing (smaller codebook for fewer files) |
| **Training progress with codebook health metrics** | v1.0 already shows loss curves and previews. VQ-VAE adds new failure modes (codebook collapse, dead codes) that users need visibility into | MEDIUM | Extends existing `training.metrics` and `ui.components.loss_chart`. Adds codebook utilization % and perplexity metrics | Codebook utilization is the critical new metric. Below 50% utilization signals collapse. Users need this surfaced prominently, not buried |
| **Model persistence for VQ-VAE format** | Users need to save trained VQ-VAE models and load them later. Different checkpoint format from v1.0 (codebooks + encoder + decoder vs mu/logvar layers) | MEDIUM | Replaces `models.persistence`. Reuses `library.catalog` JSON index pattern and atomic writes. Model metadata schema needs version field | Clean break from v1.0 format (per PROJECT.md). No backward compat needed but forward-looking schema design matters |
| **Autoregressive generation (basic)** | VQ-VAE cannot generate by sampling N(0,1) like continuous VAE. An autoregressive prior is mandatory for generation. Without it, the tool can only reconstruct/manipulate existing audio, not create new material | HIGH | New model architecture (transformer over codes). Integrates with `inference.generation.GenerationPipeline` replacing the latent vector sampling path. Reuses export, spatial, quality pipelines | Overfitting risk on small datasets. Prior must be small (2-4 transformer layers). Consider: is the prior even useful with 5-20 files? May need graceful degradation to random/shuffled code generation |
| **Temperature/randomness control for generation** | Basic generation control. Users need at minimum a "how random/surprising" knob, replacing the seed+evolution controls from v1.0 | LOW | Maps to temperature parameter in autoregressive sampling. Plugs into existing seed infrastructure. UI replaces PCA sliders with simpler control | Temperature interacts with codebook size: small codebook + low temperature = very repetitive output. Need wider default range |

### Differentiators (Competitive Advantage)

Features that make this tool unique. The "sound DNA editor" concept is the core differentiator -- no existing tool offers direct code-level manipulation of personal audio through a visual interface.

| Feature | Value Proposition | Complexity | Dependencies on Existing Code | Small-Dataset Notes |
|---------|-------------------|------------|-------------------------------|---------------------|
| **Code swapping between audio files** | **Core differentiator.** Encode two audio files, swap codes at specific time positions or specific RVQ layers. "Take the texture of sound A and the rhythm of sound B." No existing tool offers this at the discrete code level | MEDIUM | Requires encode + decode infrastructure. New operation is purely index manipulation (swap arrays). UI needs dual-pane code view | With small codebooks, fewer possible swaps but each swap is more dramatic. This is actually an advantage: fewer codes = more audible/meaningful changes |
| **Per-layer code manipulation (coarse vs fine)** | RVQ layers capture different information: layer 1 = coarse structure/content, deeper layers = fine acoustic detail. Letting users edit layers independently enables "keep the structure, change the texture" workflows | MEDIUM-HIGH | Requires RVQ encoder that exposes per-layer indices. Decoder must accept partially modified code stacks. Builds on encode/decode infrastructure | Research shows information is entangled across layers (not cleanly separated). Users need clear labeling: "Layer 1 affects overall structure, Layer 4 affects subtle detail." Manage expectations |
| **Code blending / interpolation** | Blend codes from two audio files by mixing their codebook indices (e.g., alternate codes, weighted selection, or interpolate in embedding space before re-quantizing). Creates hybrid sounds that are "between" two sources | HIGH | Interpolation in embedding space requires accessing codebook vectors (not just indices). lucidrains library provides `get_codes_from_indices`. Decode the blended embedding. More complex than simple index swapping | Blending in embedding space produces smoother results than index-level operations. With small codebooks, nearest-neighbor snapping after interpolation may collapse to one of the two sources. Monitor this |
| **Codebook entry browser/auditioner** | Let users browse individual codebook entries: click a code, hear what it sounds like in isolation (decode a single code to a short audio snippet). Builds intuition about what each code "means" | MEDIUM | Requires single-code decode capability. UI component: grid of clickable cells with preview playback. Reuses `gr.Audio` for playback | With 128-256 entries, this is actually feasible to browse. With 1024+ entries, need clustering/search. Small datasets make this feature more practical, not less |
| **Conditional generation (guide the prior)** | Beyond basic temperature: let users "seed" the autoregressive prior with a partial code sequence (e.g., encode the first 2 seconds of an audio file, then let the prior continue). Continuation/extension of existing audio | HIGH | Autoregressive prior must support prefix conditioning. Encode partial audio + generate remaining codes + decode full sequence. Integration point between encode and generate workflows | The prior may struggle to continue coherently with small training sets. However, "interesting continuation" may be more valuable than "coherent continuation" for creative users |
| **Code sequence templates/patterns** | Let users create repeating patterns, loops, or structured sequences from codes. E.g., "repeat this 4-code pattern" or "mirror this sequence." Compositional code construction without the prior | MEDIUM | Pure array manipulation on code indices. Decode the constructed sequence. UI: pattern editor with drag/copy/mirror tools | Excellent for small datasets: users can manually compose code sequences that the prior might not discover. Extends the creative palette beyond what the model "learned" |
| **Codebook usage heatmap** | Visualize which codebook entries are actually used in a specific audio file or across the training set. Identifies "dead" codes, frequently-used codes, and code distribution patterns | LOW-MEDIUM | Post-training analysis of codebook indices across dataset. t-SNE or UMAP projection of codebook embeddings. Extends `training.metrics` | Directly addresses the small-dataset codebook collapse problem. Users can see if their model is healthy (high utilization) or sick (few codes dominating) |
| **Top-k / nucleus sampling controls** | Expose nucleus (top-p) and top-k sampling parameters for the autoregressive prior. Lets users control generation diversity more precisely than temperature alone. Standard in language model interfaces | LOW | Parameters on the sampling function. UI: two additional sliders alongside temperature. No model changes needed | Users familiar with LLM interfaces (ChatGPT, etc.) will recognize these controls. Bridges the mental model between text generation and audio generation |
| **Encode-Edit-Decode workflow as single operation** | One-click workflow: upload audio, see codes, edit in place, hear result. The "sound DNA editor" as an integrated experience rather than separate encode/view/edit/decode steps | MEDIUM | Orchestration layer tying together encode, code view, edit operations, and decode into a single Gradio tab with reactive updates. State management via `ui.state.app_state` | This is what makes the tool feel magical vs feeling like a debug interface. The integrated workflow is the product |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Real-time code editing with instant audio preview** | Users want to hear changes immediately as they edit codes, like a live instrument | VQ-VAE decode (mel spectrogram + Griffin-Lim/vocoder) is too slow for real-time at 48kHz. Attempting this leads to audio glitches, buffer underruns, and frustration. The non-real-time design decision from v1.0 still applies | "Preview" button that decodes a 1-2 second snippet around the edit point. Fast enough to feel interactive (100-500ms) without promising real-time. Show a progress indicator during decode |
| **Text-conditioned code generation** | "Generate codes that sound like a warm pad" -- text prompts are familiar from ChatGPT/Suno | Requires CLAP or similar text-audio embedding model. Massive additional dependency. Text concepts are unreliable with small personal datasets (what does "warm" mean for a dataset of industrial noise?) | Temperature + manual code manipulation gives more reliable control. Codebook entry browser + auditioner lets users find sounds by listening, not describing |
| **Automatic code labeling / semantic tagging** | Users want codes labeled with what they represent (e.g., "this code = attack transient", "this code = sustain body") | Semantic meaning of codes is entangled, especially in early layers. Automated labeling would be inaccurate and misleading. Research shows even purpose-trained codecs have substantial attribute entanglement | Let users add their own labels/notes to codebook entries. Build labeling into the codebook browser. User-generated labels are more reliable than automated ones for personal datasets |
| **Importing codebooks from other models (EnCodec, SoundStream)** | "Use Google's codebook with my decoder" -- leverage pre-trained audio understanding | Codebooks are tightly coupled to their encoder/decoder. Swapping codebooks between models produces garbage. Even within the same architecture, codebooks are not interchangeable | Train your own model on your own audio. This is the core value proposition. Consider transfer learning (init from pretrained, fine-tune) in a future milestone instead |
| **Continuous latent space alongside discrete codes** | "Keep the PCA sliders from v1.0 AND add code manipulation" | Two control paradigms create confusion. Which one is the "real" control? They would need to be synchronized (slider change = code change and vice versa). Massive UI complexity for unclear benefit | Clean break: v1.1 is discrete codes only. The code manipulation UI replaces PCA sliders entirely. If users miss continuous control, consider adding "code embedding space" sliders in v1.2 that move through codebook neighborhoods |
| **Arbitrary-length generation from autoregressive prior** | "Generate 5 minutes of audio from the prior" | Autoregressive generation on small-dataset priors degrades rapidly with length. Quality drops, repetition increases, coherence breaks down. The model has seen too few examples to maintain long-range structure | Cap generation length based on model training data duration. Suggest 5-30 second outputs. For longer durations, use code sequence templates or manual composition of shorter generated segments |
| **Multi-codebook VQ (different codebook sizes per layer)** | "Use 1024 for coarse, 256 for fine" -- variable codebook sizes seem more flexible | lucidrains ResidualVQ uses uniform codebook size across layers. Variable sizes add significant complexity to the library integration, prior model, and code manipulation UI without proven benefit for small datasets | Use uniform codebook size (128-256 for small datasets, 512 for larger). The RVQ residual structure already handles coarse-to-fine via the layer hierarchy, not via codebook size |

---

## Feature Dependencies

```
RVQ-VAE Training
    |-- requires --> Codebook Management (EMA, dead code reset, k-means init)
    |-- requires --> Training Progress + Codebook Health Metrics
    |-- produces --> Trained RVQ-VAE Model
    |                   |
    |                   |-- enables --> Encode Audio to Codes
    |                   |                   |-- enables --> Code Visualization
    |                   |                   |                   |-- enables --> Code Swapping
    |                   |                   |                   |-- enables --> Per-Layer Manipulation
    |                   |                   |                   |-- enables --> Code Sequence Templates
    |                   |                   |
    |                   |                   |-- enables --> Codebook Entry Browser
    |                   |                   |
    |                   |                   |-- enables --> Code Blending (requires embedding access)
    |                   |
    |                   |-- enables --> Decode Codes to Audio
    |                   |                   |-- required by --> ALL manipulation features
    |                   |                   |-- reuses --> existing export/spatial pipeline
    |                   |
    |                   |-- enables --> VQ-VAE Model Persistence
    |                   |                   |-- reuses --> library.catalog
    |                   |
    |                   |-- enables --> Autoregressive Prior Training
    |                                       |-- requires --> Encoded code sequences from training set
    |                                       |-- enables --> Code Sequence Generation
    |                                       |                   |-- requires --> Temperature Control
    |                                       |                   |-- enhanced by --> Top-k / Nucleus Sampling
    |                                       |                   |-- enhanced by --> Conditional Generation
    |                                       |
    |                                       |-- enables --> Codebook Usage Heatmap
    |
    |-- reuses --> existing training.runner, training.checkpoint
    |-- reuses --> existing data.dataset, audio.preprocessing

Encode-Edit-Decode Workflow
    |-- requires --> Encode + Visualize + Edit Operations + Decode
    |-- orchestration layer in UI
```

### Dependency Notes

- **RVQ-VAE Training must come first**: Everything else depends on having a trained model with populated codebooks. Cannot test encode/decode without a trained model.
- **Encode and Decode are paired**: Encode without decode is diagnostic-only. Decode without encode limits to prior-generated or manually constructed codes. Both are needed for the core workflow.
- **Code visualization gates all manipulation features**: Users cannot swap/blend/edit what they cannot see. Visualization is the prerequisite for all creative operations.
- **Autoregressive prior is independent of code manipulation**: The prior generates codes; manipulation edits codes. They enhance each other but neither requires the other. Can ship manipulation without prior, or prior without manipulation.
- **Code blending is harder than code swapping**: Swapping is index-level (array operations). Blending requires embedding-space interpolation and re-quantization. Ship swapping first, blending second.
- **Conditional generation requires both prior and encode**: Must encode a prefix, then feed to the prior. This is the most complex integration point.

---

## UI Paradigm Recommendation: The Sound DNA Editor

### Why Not PCA Sliders (v1.0 Approach)

The v1.0 PCA slider approach maps continuous latent dimensions to knobs. This breaks down for VQ-VAE because:
1. Discrete codes are integers, not continuous values. Sliders produce continuous values that must be quantized, losing the precision advantage of discrete codes.
2. PCA assumes a Gaussian latent space. VQ-VAE codes occupy a finite codebook, not a continuous manifold.
3. The interesting operations are combinatorial (swap code 47 for code 183), not continuous (slide from -10 to +10).

### Recommended UI: Code Grid + Timeline

**Primary view: Code Timeline**
- Horizontal axis = time (one column per time step, matching encoder frame rate)
- Vertical axis = RVQ layer (row 1 = coarsest/layer 1, row N = finest/layer N)
- Each cell shows a codebook index (integer) with color-coding by entry
- Clicking a cell opens a picker showing all codebook entries with audio preview
- Selecting a region enables batch operations (swap, copy, paste, fill with pattern)

**Secondary view: Codebook Browser**
- Grid of all codebook entries for a selected layer
- Each entry shows: index number, usage frequency, audio preview button
- Entries color-coded by usage (hot/cold heatmap)
- Search/filter by audio similarity

**Tertiary view: Dual-File Comparison**
- Two code timelines side by side (source A, source B)
- Drag-and-drop codes between files
- "Merge" operation that interleaves or selects codes from both sources

### Gradio Implementation Constraints

Gradio does not have a native "code grid editor" component. Practical approaches:
1. **gr.DataFrame** for the code grid: editable cells, color via CSS. Functional but not visually rich.
2. **gr.Plot (Plotly heatmap)** for visualization + separate edit controls: pretty but indirect editing.
3. **Custom Gradio component** via gr.HTML + JavaScript: most flexible but highest development cost.
4. **Hybrid approach** (recommended): Plotly heatmap for visualization + row of dropdowns/number inputs for editing selected time steps + action buttons for batch operations. Achievable within standard Gradio.

---

## MVP Definition (v1.1 Milestone)

### Must Have (Core VQ-VAE Value)

- [ ] **RVQ-VAE Training** -- Cannot do anything without a trained model. Replaces v1.0 training loop entirely. Must include codebook health monitoring.
- [ ] **Encode audio to discrete codes** -- The bridge from audio to the code domain. Without this, discrete codes are invisible.
- [ ] **Decode codes to audio** -- The bridge back. Without this, code manipulation is silent.
- [ ] **Code visualization (timeline grid)** -- Users must see the codes to understand and trust the system. The "aha moment" of the product.
- [ ] **Basic code editing (select cell, change code index)** -- Minimum viable manipulation. Even changing one code and hearing the result proves the concept.
- [ ] **Code swapping between two files** -- The headline feature: "mix the DNA of two sounds." Most accessible creative operation.
- [ ] **Autoregressive prior (basic generation)** -- VQ-VAE without a prior cannot generate new material, only reconstruct/edit existing audio. Generation is expected.
- [ ] **Temperature control for generation** -- Minimum generation control. Users need "more random" vs "more conservative."
- [ ] **Updated model persistence** -- Users must save and load VQ-VAE models. Reuses library catalog pattern.

### Add After Core Works (v1.1.x)

- [ ] **Per-layer code manipulation** -- Trigger: users want "change the texture but keep the rhythm." Requires labeling layers with what they affect.
- [ ] **Codebook entry browser with audio preview** -- Trigger: users ask "what does code 47 sound like?" Builds intuition about the codebook.
- [ ] **Codebook usage heatmap** -- Trigger: users report "my model sounds weird." Diagnostic tool for codebook health post-training.
- [ ] **Code blending in embedding space** -- Trigger: swapping feels too abrupt. Blending enables smoother transitions between sounds.
- [ ] **Top-k / nucleus sampling** -- Trigger: temperature alone is too coarse. Power users want finer generation control.
- [ ] **Code sequence templates/patterns** -- Trigger: users want rhythmic/looping structures. Manual composition of code sequences.

### Future Consideration (v1.2+)

- [ ] **Conditional generation (audio continuation)** -- Why defer: requires robust prior + encode integration. Prior may not be coherent enough on small datasets initially.
- [ ] **Encode-Edit-Decode as single integrated workflow** -- Why defer: orchestration layer is polish. Core pieces must work independently first.
- [ ] **Code embedding space sliders** -- Why defer: this is "PCA sliders but for codes." Only add if users miss continuous control after living with discrete editing.
- [ ] **Batch code operations across multiple files** -- Why defer: power user feature. Single-file editing must be solid first.

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Phase Order |
|---------|------------|---------------------|----------|-------------|
| RVQ-VAE Training + Codebook Management | HIGH | HIGH | P1 | 1st (everything depends on this) |
| Encode Audio to Codes | HIGH | MEDIUM | P1 | 2nd (needs trained model) |
| Decode Codes to Audio | HIGH | MEDIUM | P1 | 2nd (paired with encode) |
| Code Visualization (timeline grid) | HIGH | MEDIUM | P1 | 3rd (needs encode output) |
| Basic Code Editing (cell-level) | HIGH | LOW-MEDIUM | P1 | 3rd (needs visualization) |
| Code Swapping Between Files | HIGH | LOW-MEDIUM | P1 | 3rd (needs encode + edit) |
| Updated Model Persistence | HIGH | MEDIUM | P1 | 2nd (save trained models) |
| Training Progress + Codebook Health | MEDIUM-HIGH | MEDIUM | P1 | 1st (part of training) |
| Autoregressive Prior Training | HIGH | HIGH | P1 | 4th (needs encoded training data) |
| Temperature Control | MEDIUM-HIGH | LOW | P1 | 4th (part of generation) |
| Per-Layer Code Manipulation | MEDIUM | MEDIUM | P2 | After core works |
| Codebook Entry Browser | MEDIUM | MEDIUM | P2 | After core works |
| Codebook Usage Heatmap | MEDIUM | LOW-MEDIUM | P2 | After core works |
| Code Blending (embedding space) | MEDIUM | HIGH | P2 | After swapping works |
| Top-k / Nucleus Sampling | LOW-MEDIUM | LOW | P2 | After basic generation |
| Code Sequence Templates | MEDIUM | MEDIUM | P2 | After visualization |
| Conditional Generation | MEDIUM | HIGH | P3 | Requires mature prior |
| Integrated Workflow Polish | MEDIUM | MEDIUM | P3 | Polish phase |

**Priority key:**
- P1: Must have for v1.1 milestone (proves VQ-VAE + code manipulation value prop)
- P2: Should have, add when P1 is working (elevates from functional to compelling)
- P3: Future consideration (requires P1+P2 maturity)

---

## Competitor/Prior Art Feature Analysis

| Feature | Jukebox (OpenAI) | RAVE (IRCAM) | EnCodec/MusicGen (Meta) | SoundStream (Google) | Our Approach |
|---------|------------------|--------------|-------------------------|----------------------|--------------|
| **Architecture** | 3-level VQ-VAE, 2048 codebook, Transformer priors | Continuous VAE (or discrete config like SoundStream) | RVQ with 8 codebooks, 1024 entries each | RVQ with up to 80 layers, quantizer dropout | RVQ with 4-8 layers, 128-512 entries (dataset-adaptive) |
| **Training data** | 1.2M songs | Hours of audio | Large corpora | Large corpora | **5-500 files (minutes to hours)** |
| **User training** | No | Yes (CLI) | No | No | **Yes (GUI + CLI)** |
| **Code visualization** | None (research tool) | Latent dims visible in Max/MSP | None (library API) | None (research tool) | **Code timeline grid with per-layer view** |
| **Code manipulation** | None | Slider manipulation of continuous latents | None | None | **Swap, edit, blend, template codes** |
| **Generation control** | Artist/genre/lyrics conditioning | Latent space LFOs in Max/MSP | Text + melody conditioning | N/A (codec, not generator) | **Temperature + top-k + code seeding** |
| **Codebook introspection** | None exposed | None exposed | None exposed | None exposed | **Codebook browser + usage heatmap** |
| **Small dataset support** | No | Partial (needs hours) | No | No | **Primary design target** |

**Key differentiation**: No existing tool exposes discrete audio codes for direct user manipulation. Jukebox/EnCodec/SoundStream are research/production codecs with no creative editing interface. RAVE offers latent manipulation but in continuous space, not discrete codes. Our "sound DNA editor" is genuinely novel.

---

## Domain-Specific Insights

### What RVQ Layers Actually Capture (Research Findings)

Research on interpretable neural audio codecs (Interspeech 2025) reveals:

- **Layer 1**: Captures coarse structure -- linguistic content for speech, rhythmic/harmonic structure for music. Highest alignment with semantic features. Modifying layer 1 produces the most audible and structural changes.
- **Layers 2-4**: Encode speaker identity, timbre characteristics, and broad spectral shape. Changes here alter "who/what it sounds like" while preserving structure.
- **Deeper layers (5+)**: Fine acoustic detail -- precise spectral shape, noise characteristics, subtle timing. Changes are audible but subtle. Increasingly "diffuse" with less interpretable structure.
- **Key caveat**: Information is substantially **entangled** across layers, not cleanly separated. Modifying one layer can unexpectedly affect attributes that "belong" to another layer. Users should be warned: "Layer operations affect multiple aspects of the sound."

**Implication for our UI**: Label layers with approximate descriptions ("Structure", "Timbre", "Detail") but include a disclaimer. Per-layer editing is powerful but not surgically precise.

### Codebook Sizing for Small Datasets

Research on codebook collapse and small data suggests:

- **128 entries**: Viable minimum for 5-20 files. Low risk of collapse. Coarse representation but meaningful.
- **256 entries**: Good balance for 20-100 files. May see 10-20% dead codes without mitigation.
- **512 entries**: Appropriate for 100-500 files. Requires EMA + dead code reset.
- **1024+ entries**: Risky for small datasets. High collapse probability. Only viable with aggressive mitigation (k-means init + EMA + reset + warmup).

**Recommendation**: Auto-scale codebook size based on dataset size. Default mapping: `codebook_size = min(max(len(dataset) * 8, 128), 512)`. Expose as an advanced setting.

### Autoregressive Prior Design for Small Datasets

- **Model size**: 2-4 transformer layers, 128-256 hidden dim. Larger models overfit immediately on small datasets.
- **Training**: Prior trains on encoded sequences from the training set. With 20 files, that may be only 20 code sequences. High overfitting risk.
- **Practical expectation**: The prior will memorize small datasets rather than generalize. This is partially acceptable: "remixed versions of your sounds" rather than "entirely new sounds." Set user expectations accordingly.
- **Sampling diversity**: Temperature > 1.0 may be necessary to get novel output from an overfit prior. Top-k sampling prevents degenerate tokens.
- **Alternative for very small datasets (5-10 files)**: Skip the prior entirely. Offer shuffle-based and template-based code generation instead. Users compose new code sequences manually from the codes they can see in the visualization.

### Gradio UI Patterns for Code Manipulation

Based on Gradio component capabilities (2025-2026):
- **gr.Dataframe**: Best for editable code grid. Supports colored cells via CSS. Integer-only cells for code indices.
- **gr.Plot (Plotly)**: Best for code visualization heatmaps. Click events supported. Not directly editable.
- **gr.Audio**: Instant preview of decode results. Pair with every edit operation.
- **gr.Dropdown per cell**: Feasible for small time ranges but not scalable to full sequences. Use for "edit selected region" pattern.
- **State management**: `gr.State` for current code sequence, edit history (undo/redo). The `ui.state.app_state` singleton can hold the active code buffer.

---

## Sources

**RVQ-VAE Architecture and Neural Audio Codecs:**
- [SoundStream: An End-to-End Neural Audio Codec](https://research.google/blog/soundstream-an-end-to-end-neural-audio-codec/)
- [EnCodec: High Fidelity Neural Audio Compression](https://audiocraft.metademolab.com/encodec.html)
- [Jukebox: A Generative Model for Music](https://openai.com/index/jukebox/)
- [EuleroDec: Complex-Valued RVQ-VAE for Efficient Audio Coding](https://arxiv.org/abs/2601.17517)
- [ERVQ: Enhanced Residual Vector Quantization](https://arxiv.org/html/2410.12359)
- [Neural audio codecs: how to get audio into LLMs](https://kyutai.org/codec-explainer)

**VQ-VAE Fundamentals and Autoregressive Priors:**
- [Neural Discrete Representation Learning (VQ-VAE paper)](https://arxiv.org/abs/1711.00937)
- [VQ-VAE-2 Explained](https://paperswithcode.com/method/vq-vae-2)
- [Understanding VQ-VAE (ML Berkeley)](https://mlberkeley.substack.com/p/vq-vae)
- [VQ-VAE-2 Implementation with Autoregressive Prior](https://github.com/mattiasxu/VQVAE-2)

**Codebook Interpretability and Layer Analysis:**
- [Bringing Interpretability to Neural Audio Codecs (Interspeech 2025)](https://arxiv.org/html/2506.04492v1)
- [Analysing the Language of Neural Audio Codecs](https://arxiv.org/html/2509.01390)
- [Discrete Audio Tokens: More Than a Survey](https://hal.science/hal-05424376v1/file/5055_Discrete_Audio_Tokens_Mor.pdf)

**Codebook Collapse and Small Dataset Techniques:**
- [Addressing Index Collapse of Large-Codebook Speech Tokenizer](https://arxiv.org/html/2406.02940v1)
- [Finite Scalar Quantization: VQ-VAE Made Simple (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf)
- [Addressing Representation Collapse in VQ Models](https://openreview.net/pdf/91a14185eeff83559b178556728653853d8a8803.pdf)
- [Is Hierarchical Quantization Essential for Optimal Reconstruction?](https://arxiv.org/html/2601.22244)

**Creative Audio Manipulation and Style Transfer:**
- [Self-Supervised VQ-VAE for One-Shot Music Style Transfer](https://github.com/cifkao/ss-vq-vae)
- [Latent Space Interpolation of Synthesizer Parameters](https://gwendal-lv.github.io/spinvae2/)
- [TokenSynth: Token-based Neural Synthesizer](https://arxiv.org/html/2502.08939v1)
- [Audio Conditioning for Music Generation via Discrete Bottleneck Features](https://musicgenstyle.github.io/)

**Autoregressive Generation and Sampling:**
- [Generation configurations: temperature, top-k, top-p](https://huyenchip.com/2024/01/16/sampling.html)
- [Top-k and Top-p Decoding](https://www.aussieai.com/research/top-k-decoding)
- [Continuous Autoregressive Modeling (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/7d90c28e7820709792d969211815a2b3-Paper-Conference.pdf)

**RAVE and Latent Space Exploration:**
- [RAVE Official Implementation](https://github.com/acids-ircam/RAVE)
- [Tutorial: Neural Synthesis in Max 8 with RAVE](https://forum.ircam.fr/article/detail/tutorial-neural-synthesis-in-max-8-with-rave/)
- [Latent Terrain: Dissecting the Latent Space of Neural Audio Autoencoders](https://forum.ircam.fr/article/detail/latent-terrain-dissecting-the-latent-space-of-neural-audio-autoencoder-by-shuoyang-jasper-zheng/)

**Library (lucidrains vector-quantize-pytorch):**
- [vector-quantize-pytorch GitHub](https://github.com/lucidrains/vector-quantize-pytorch)
- [Residual Vector Quantization Explained (Scott Hawley)](https://drscotthawley.github.io/blog/posts/2023-06-12-RVQ.html)

**Gradio UI Components:**
- [Gradio Audio Component Docs](https://www.gradio.app/docs/gradio/audio)
- [Gradio Audio Waveform Visualization Request](https://github.com/gradio-app/gradio/issues/9740)

---
*Feature research for: VQ-VAE Audio Code Manipulation (v1.1 milestone)*
*Researched: 2026-02-21*
*Confidence: MEDIUM-HIGH (RVQ-VAE architecture patterns well-established in literature; code manipulation UI paradigm is novel with limited prior art; small-dataset implications extrapolated from codebook collapse research)*
