# Requirements: Small DataSet Audio

**Defined:** 2026-02-21
**Core Value:** Controllable exploration -- users can reliably navigate between sound worlds using discrete audio codes and generative priors

## v1.1 Requirements

Requirements for VQ-VAE milestone. Each maps to roadmap phases.

### VQ-VAE Architecture & Training

- [x] **VQVAE-01**: User can train an RVQ-VAE model on a small audio dataset (5-500 files) using stacked residual codebooks
- [x] **VQVAE-02**: Codebook size automatically scales based on dataset size (64 for 5-20 files, 128 for 20-100, 256 for 100-500)
- [x] **VQVAE-03**: Training uses EMA codebook updates with k-means initialization and dead code reset
- [x] **VQVAE-04**: Per-level codebook utilization, perplexity, and dead code count are displayed during training
- [x] **VQVAE-05**: Training uses commitment loss (single weight parameter) instead of KL divergence
- [x] **VQVAE-06**: User can configure number of RVQ levels (2-4) and codebook dimension
- [x] **VQVAE-07**: Training detects and warns when codebook utilization drops below 30%

### Generation & Prior

- [x] **GEN-01**: User can train an autoregressive prior model on frozen VQ-VAE code sequences
- [ ] **GEN-02**: User can generate new audio from the trained prior with temperature control
- [ ] **GEN-03**: User can control generation with top-k and nucleus (top-p) sampling
- [ ] **GEN-04**: Prior generates multi-chunk audio with overlap-add stitching (existing pipeline)
- [x] **GEN-05**: Prior model is bundled in the saved model file alongside the VQ-VAE
- [x] **GEN-06**: Prior training detects memorization (validation perplexity monitoring)

### Code Manipulation

- [ ] **CODE-01**: User can encode any audio file into its discrete code representation
- [ ] **CODE-02**: User can decode a code grid back to audio with playback preview
- [ ] **CODE-03**: User can view codes as a timeline grid (rows = quantizer levels, columns = time positions)
- [ ] **CODE-04**: User can edit individual code cells (change codebook index)
- [ ] **CODE-05**: User can swap code regions between two encoded audio files
- [ ] **CODE-06**: User can blend codes in embedding space (smoother than index swapping)
- [ ] **CODE-07**: User can preview individual codebook entries as audio (click a code, hear it)
- [ ] **CODE-08**: Code edits support undo/redo
- [ ] **CODE-09**: Per-layer manipulation is labeled (Structure/Timbre/Detail)

### Model Persistence

- [x] **PERS-01**: VQ-VAE models save as v2 format with codebook state and VQ-specific metadata
- [x] **PERS-02**: Prior model state is bundled in the same model file
- [ ] **PERS-03**: Model library (save/load/delete/search) works with v2 format

### User Interface

- [ ] **UI-01**: New Codes tab in Gradio UI for code visualization and editing
- [ ] **UI-02**: Codebook usage heatmap displayed as post-training diagnostic
- [x] **UI-03**: Training tab updated for VQ-VAE config (codebook size, RVQ levels, commitment weight)
- [ ] **UI-04**: Generate tab updated for prior-based generation (temperature, top-k, top-p controls)
- [ ] **UI-05**: Codebook health shown as green/yellow/red indicator with plain-language labels

### Command Line

- [x] **CLI-01**: CLI supports VQ-VAE training with configurable codebook parameters
- [x] **CLI-02**: CLI supports prior training on a trained VQ-VAE model
- [ ] **CLI-03**: CLI supports encode/decode operations for code manipulation
- [ ] **CLI-04**: CLI supports generation from trained prior with sampling controls

## Future Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Deferred from v1.0

- **DEFER-01**: Incrementally add more training audio to an existing model
- **DEFER-02**: Feed generated outputs back into training data for iterative refinement
- **DEFER-03**: Bundle HRTF SOFA file for binaural mode

### Deferred from v1.1 Research

- **DEFER-04**: Conditional generation / audio continuation (requires mature prior)
- **DEFER-05**: Encode-Edit-Decode as single integrated workflow (polish requiring solid P1+P2)
- **DEFER-06**: Code embedding space sliders (only if users miss continuous control from v1.0)
- **DEFER-07**: Code sequence templates and patterns

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real-time code editing | Griffin-Lim decode is too slow for interactive feedback |
| Text-conditioned generation | Massive dependency, unreliable on personal small datasets |
| Importing codebooks from other models | Codebooks are architecture-coupled and not interchangeable |
| Arbitrary-length prior generation | Degrades rapidly on small datasets |
| Continuous PCA sliders alongside codes | Two paradigms create confusion; clean break to discrete |
| v1.0 model backward compatibility | Clean break; VQ-VAE is fundamentally different architecture |
| Multi-model blending | v1.0 feature; needs complete redesign for VQ-VAE codes |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| VQVAE-01 | Phase 12 | Complete |
| VQVAE-02 | Phase 12 | Complete |
| VQVAE-03 | Phase 12 | Complete |
| VQVAE-04 | Phase 13 | Complete |
| VQVAE-05 | Phase 12 | Complete |
| VQVAE-06 | Phase 12 | Complete |
| VQVAE-07 | Phase 13 | Complete |
| GEN-01 | Phase 14 | Complete |
| GEN-02 | Phase 15 | Pending |
| GEN-03 | Phase 15 | Pending |
| GEN-04 | Phase 15 | Pending |
| GEN-05 | Phase 14 | Complete |
| GEN-06 | Phase 14 | Complete |
| CODE-01 | Phase 16 | Pending |
| CODE-02 | Phase 16 | Pending |
| CODE-03 | Phase 16 | Pending |
| CODE-04 | Phase 17 | Pending |
| CODE-05 | Phase 17 | Pending |
| CODE-06 | Phase 17 | Pending |
| CODE-07 | Phase 16 | Pending |
| CODE-08 | Phase 17 | Pending |
| CODE-09 | Phase 16 | Pending |
| PERS-01 | Phase 13 | Complete |
| PERS-02 | Phase 14 | Complete |
| PERS-03 | Phase 18 | Pending |
| UI-01 | Phase 17 | Pending |
| UI-02 | Phase 18 | Pending |
| UI-03 | Phase 13 | Complete |
| UI-04 | Phase 15 | Pending |
| UI-05 | Phase 18 | Pending |
| CLI-01 | Phase 13 | Complete |
| CLI-02 | Phase 14 | Complete |
| CLI-03 | Phase 18 | Pending |
| CLI-04 | Phase 15 | Pending |

**Coverage:**
- v1.1 requirements: 34 total
- Mapped to phases: 34
- Unmapped: 0

---
*Requirements defined: 2026-02-21*
*Last updated: 2026-02-21 after roadmap creation (traceability updated)*
