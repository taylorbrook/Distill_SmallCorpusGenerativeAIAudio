# Milestones

## v1.0 MVP (Shipped: 2026-02-15)

**Delivered:** A generative audio application that trains on small personal datasets (5-500 files) and generates high-fidelity audio with musically meaningful slider controls, full GUI and CLI, multi-format export, and spatial audio.

**Phases:** 1-11 (11 phases, 34 plans, ~70 tasks)
**Stats:** 17,520 LOC Python | 149 commits | 186 files | 3 days (2026-02-12 to 2026-02-14)
**Git range:** initial commit → c88be10

**Key accomplishments:**
1. Complete AI audio generation from small datasets (5-500 files) with VAE training and overfitting prevention
2. Musical parameter control via PCA-based latent space analysis with safe/warning zones
3. Professional audio export in WAV/MP3/FLAC/OGG with spatial processing (stereo/binaural)
4. Full-featured Gradio web UI with live training, slider controls, model library, presets, A/B comparison
5. Scriptable CLI for batch generation, training, and model management on headless servers
6. Model persistence as .sda files with library management and instant slider restoration on load

**Tech debt carried forward:**
- 16 human listening/verification tests pending (Phases 4, 5, 10)
- HRTF SOFA file not bundled for binaural mode (user must download)

---

## v1.1 VQ-VAE (In Progress)

**Goal:** Replace continuous VAE with RVQ-VAE architecture, add autoregressive prior for generation, build code manipulation UI for exploring discrete audio representations.

**Phases:** 12-18 (7 phases, 34 requirements)
**Started:** 2026-02-21

**Key deliverables:**
1. RVQ-VAE encoder/decoder with stacked codebooks and dataset-adaptive sizing
2. Codebook health monitoring (utilization, perplexity, dead code detection)
3. Autoregressive prior model for generating new code sequences
4. Code manipulation UI: encode audio to codes, view/edit/swap/blend, decode back
5. Updated training pipeline, model persistence (v2 format), and CLI
6. Post-training diagnostics (usage heatmaps, health indicators)

**Research flags:**
- Empirical codebook sizing defaults need validation on 5/50/500 file datasets
- LSTM vs. Transformer prior decision pending for small datasets
- Gradio code grid editor has no established prior art

---
