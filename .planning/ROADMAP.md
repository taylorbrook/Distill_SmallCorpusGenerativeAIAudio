# Roadmap: Small DataSet Audio

## Milestones

- ✅ **v1.0 MVP** -- Phases 1-11 (shipped 2026-02-15)
- 🚧 **v1.1 VQ-VAE** -- Phases 12-18 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-11) -- SHIPPED 2026-02-15</summary>

- [x] Phase 1: Project Setup (3/3 plans) -- completed 2026-02-12
- [x] Phase 2: Data Pipeline Foundation (3/3 plans) -- completed 2026-02-12
- [x] Phase 3: Core Training Engine (4/4 plans) -- completed 2026-02-12
- [x] Phase 4: Audio Quality & Export (3/3 plans) -- completed 2026-02-12
- [x] Phase 5: Musically Meaningful Controls (2/2 plans) -- completed 2026-02-13
- [x] Phase 6: Model Persistence & Management (1/1 plan) -- completed 2026-02-13
- [x] Phase 7: Presets & Generation History (3/3 plans) -- completed 2026-02-13
- [x] Phase 8: Gradio UI (5/5 plans) -- completed 2026-02-13
- [x] Phase 9: CLI Interface (3/3 plans) -- completed 2026-02-14
- [x] Phase 10: Multi-Format Export & Spatial Audio (5/5 plans) -- completed 2026-02-15
- [x] Phase 11: Wire Latent Space Analysis (2/2 plans) -- completed 2026-02-14

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### 🚧 v1.1 VQ-VAE (In Progress)

**Milestone Goal:** Replace continuous VAE with RVQ-VAE, add autoregressive prior for generation, build code manipulation UI for exploring discrete audio representations.

- [x] **Phase 12: RVQ-VAE Core Architecture** - Build the ConvVQVAE model with residual vector quantization, codebook management, and commitment loss (completed 2026-02-22)
- [x] **Phase 13: VQ-VAE Training Pipeline** - Wire VQ-VAE into training loop with codebook health monitoring, model persistence, and training UI/CLI (completed 2026-02-22)
- [x] **Phase 14: Autoregressive Prior** - Build and train the autoregressive prior model on frozen VQ-VAE code sequences (completed 2026-02-22)
- [x] **Phase 15: Generation Pipeline** - Wire prior-based generation with sampling controls, multi-chunk stitching, and generate UI/CLI (completed 2026-02-27)
- [ ] **Phase 16: Encode/Decode + Code Visualization** - Encode audio to codes, decode back, visualize as timeline grid, preview codebook entries
- [ ] **Phase 17: Code Editing** - Interactive code manipulation: cell editing, region swapping, embedding blending, undo/redo
- [ ] **Phase 18: Diagnostics + Library Integration** - Codebook health indicators, usage heatmaps, model library v2 support, CLI encode/decode

## Phase Details

### Phase 12: RVQ-VAE Core Architecture
**Goal**: A working RVQ-VAE model exists that can encode mel spectrograms to discrete codes and decode them back, with dataset-adaptive codebook sizing and stable quantization
**Depends on**: Phase 11 (v1.0 codebase)
**Requirements**: VQVAE-01, VQVAE-02, VQVAE-03, VQVAE-05, VQVAE-06
**Success Criteria** (what must be TRUE):
  1. User can instantiate a ConvVQVAE model with configurable RVQ levels (2-4) and codebook dimension, and run a forward pass that returns reconstructed mel, code indices, and commitment loss
  2. Codebook size automatically scales based on dataset size (64 for 5-20 files, 128 for 20-100, 256 for 100-500) without user intervention
  3. Codebooks initialize via k-means on first batch, update via EMA, and automatically reset dead codes during training
  4. Training uses commitment loss (single weight parameter) with no KL divergence, free bits, or annealing logic
**Plans:** 2/2 plans complete
Plans:
- [ ] 12-01-PLAN.md -- Install vector-quantize-pytorch, create VQVAEConfig with adaptive sizing, build ConvVQVAE model architecture
- [ ] 12-02-PLAN.md -- Create vqvae_loss with multi-scale spectral loss, wire VQ-VAE exports through models __init__.py

### Phase 13: VQ-VAE Training Pipeline
**Goal**: Users can train an RVQ-VAE model end-to-end through the UI or CLI, see codebook health during training, and save/load trained models in v2 format
**Depends on**: Phase 12
**Requirements**: VQVAE-04, VQVAE-07, PERS-01, UI-03, CLI-01
**Success Criteria** (what must be TRUE):
  1. User can start VQ-VAE training from the training tab with configurable codebook size, RVQ levels, and commitment weight
  2. Per-level codebook utilization, perplexity, and dead code count are displayed during training in the UI
  3. Training warns the user when codebook utilization drops below 30% on any level
  4. Trained model saves as v2 format (.sda file) containing codebook state and VQ-specific metadata, and loads back identically
  5. User can train from CLI with `--codebook-size`, `--rvq-levels`, and `--commitment-weight` flags
**Plans:** 3/3 plans complete
Plans:
- [ ] 13-01-PLAN.md -- VQ-VAE training loop, VQ metrics dataclasses, codebook health monitoring, v2 model persistence
- [ ] 13-02-PLAN.md -- Gradio train tab VQ-VAE controls, codebook health display, loss chart with commitment loss
- [ ] 13-03-PLAN.md -- CLI VQ-VAE training flags, per-level health output, end-of-training summary

### Phase 14: Autoregressive Prior
**Goal**: An autoregressive prior model can be trained on frozen VQ-VAE code sequences, with memorization detection, and bundled into the saved model file
**Depends on**: Phase 13 (needs a trained, saved VQ-VAE model)
**Requirements**: GEN-01, GEN-05, GEN-06, PERS-02, CLI-02
**Success Criteria** (what must be TRUE):
  1. User can train a prior model on a frozen VQ-VAE model's code sequences, and see training progress (loss, validation perplexity)
  2. Training detects and warns when validation perplexity drops below a memorization threshold
  3. Prior model state is bundled into the same .sda model file alongside the VQ-VAE weights
  4. User can train the prior from CLI by pointing at a trained VQ-VAE model file
**Plans:** 3/3 plans complete
Plans:
- [ ] 14-01-PLAN.md -- CodePrior transformer model, PriorConfig with adaptive sizing, code extraction pipeline
- [ ] 14-02-PLAN.md -- Prior training loop with cross-entropy loss, memorization detection, best-checkpoint tracking
- [ ] 14-03-PLAN.md -- Persistence bundling (prior into .sda), CLI train-prior command, public API exports

### Phase 15: Generation Pipeline
**Goal**: Users can generate new audio from a trained prior with temperature, top-k, and top-p controls, producing multi-chunk output through the existing pipeline
**Depends on**: Phase 14 (needs a trained prior)
**Requirements**: GEN-02, GEN-03, GEN-04, UI-04, CLI-04
**Success Criteria** (what must be TRUE):
  1. User can generate new audio from the generate tab using a model that has a trained prior, and hear output that is recognizably derived from training data but not identical
  2. User can control generation diversity via temperature, top-k, and top-p sliders in the generate tab
  3. Generated audio longer than one chunk uses overlap-add stitching with no audible seams at chunk boundaries
  4. User can generate from CLI with `--temperature`, `--top-k`, and `--top-p` flags
**Plans:** 3/3 plans complete
Plans:
- [ ] 15-01-PLAN.md -- Core sampling engine (sample_code_sequence) and generate_audio_from_prior() with multi-chunk stitching
- [ ] 15-02-PLAN.md -- Generate tab UI rework with prior-based sampling controls (temperature, top-k, top-p, duration)
- [ ] 15-03-PLAN.md -- CLI generate extension with VQ-VAE model detection and sampling flags

### Phase 16: Encode/Decode + Code Visualization
**Goal**: Users can encode audio files into discrete code representations, view them as a labeled timeline grid, preview individual codebook entries as audio, and decode codes back to audio
**Depends on**: Phase 13 (needs a trained VQ-VAE model; does NOT depend on prior)
**Requirements**: CODE-01, CODE-02, CODE-03, CODE-07, CODE-09
**Success Criteria** (what must be TRUE):
  1. User can load an audio file and encode it into a grid of discrete codes, displayed as a timeline (columns = time positions, rows = quantizer levels)
  2. Quantizer levels are labeled with semantic roles (e.g., Structure / Timbre / Detail) rather than raw level numbers
  3. User can click any codebook entry in the grid to hear a short audio preview of what that code sounds like
  4. User can decode the code grid back to audio and play it back, hearing a reconstruction of the original
**Plans**: TBD

### Phase 17: Code Editing
**Goal**: Users can interactively edit discrete codes -- changing individual cells, swapping regions between audio files, blending in embedding space -- with full undo/redo, in a dedicated Codes tab
**Depends on**: Phase 16 (needs encode/decode and visualization)
**Requirements**: CODE-04, CODE-05, CODE-06, CODE-08, UI-01
**Success Criteria** (what must be TRUE):
  1. User can change individual code cells in the grid (select a cell, pick a new codebook index) and decode to hear the result
  2. User can encode two audio files and swap selected code regions between them, creating hybrid audio
  3. User can blend codes between two files in embedding space, producing smoother interpolations than index swapping
  4. All code edits support undo and redo, so the user can step backward through changes
  5. A dedicated Codes tab in the Gradio UI hosts all code visualization and editing operations
**Plans**: TBD

### Phase 18: Diagnostics + Library Integration
**Goal**: Users see codebook health at a glance with plain-language indicators, can browse codebook usage heatmaps, and the model library fully supports v2 format including CLI encode/decode
**Depends on**: Phase 13 (persistence), Phase 16 (encode/decode for CLI)
**Requirements**: UI-02, UI-05, PERS-03, CLI-03
**Success Criteria** (what must be TRUE):
  1. After training, a codebook usage heatmap is displayed showing which codes are heavily used, lightly used, or dead
  2. Codebook health is shown as a green/yellow/red indicator with plain-language labels (e.g., "Healthy", "Some codes underused", "Codebook collapse detected") rather than raw numbers
  3. The model library (save/load/delete/search) works with v2 format models, and v2 models appear in search results with VQ-specific metadata
  4. User can encode and decode audio files from the CLI (e.g., `encode --model X --input audio.wav --output codes.json`)
**Plans**: TBD

## Progress

**Execution Order:**
Phases 12-18 execute sequentially, except: Phase 16 can begin after Phase 13 (parallel with 14-15).

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Project Setup | v1.0 | 3/3 | Complete | 2026-02-12 |
| 2. Data Pipeline Foundation | v1.0 | 3/3 | Complete | 2026-02-12 |
| 3. Core Training Engine | v1.0 | 4/4 | Complete | 2026-02-12 |
| 4. Audio Quality & Export | v1.0 | 3/3 | Complete | 2026-02-12 |
| 5. Musically Meaningful Controls | v1.0 | 2/2 | Complete | 2026-02-13 |
| 6. Model Persistence & Management | v1.0 | 1/1 | Complete | 2026-02-13 |
| 7. Presets & Generation History | v1.0 | 3/3 | Complete | 2026-02-13 |
| 8. Gradio UI | v1.0 | 5/5 | Complete | 2026-02-13 |
| 9. CLI Interface | v1.0 | 3/3 | Complete | 2026-02-14 |
| 10. Multi-Format Export & Spatial Audio | v1.0 | 5/5 | Complete | 2026-02-15 |
| 11. Wire Latent Space Analysis | v1.0 | 2/2 | Complete | 2026-02-14 |
| 12. RVQ-VAE Core Architecture | 2/2 | Complete    | 2026-02-22 | - |
| 13. VQ-VAE Training Pipeline | 3/3 | Complete    | 2026-02-22 | - |
| 14. Autoregressive Prior | 3/3 | Complete    | 2026-02-22 | - |
| 15. Generation Pipeline | 3/3 | Complete    | 2026-02-27 | - |
| 16. Encode/Decode + Code Visualization | v1.1 | 0/0 | Not started | - |
| 17. Code Editing | v1.1 | 0/0 | Not started | - |
| 18. Diagnostics + Library Integration | v1.1 | 0/0 | Not started | - |

---
*Roadmap created: 2026-02-12*
*Last updated: 2026-02-21 after Phase 12 planning complete*
