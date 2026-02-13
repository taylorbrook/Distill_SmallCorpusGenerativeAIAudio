# Roadmap: Small DataSet Audio

## Overview

This roadmap follows a risk-first progression: establish a robust small-dataset training pipeline with overfitting prevention, validate 48kHz audio quality, build the core innovation (musically meaningful parameter controls via latent space disentanglement), then wrap in usable interfaces. The 10-phase structure reflects comprehensive depth, delivering a research-grade generative audio tool that learns from personal datasets as small as 5-500 files and generates high-fidelity output with controllable exploration.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Project Setup** - Environment, dependencies, hardware abstraction *(completed 2026-02-12)*
- [x] **Phase 2: Data Pipeline Foundation** - Import, validation, preprocessing, augmentation *(completed 2026-02-12)*
- [x] **Phase 3: Core Training Engine** - VAE training with overfitting prevention and checkpointing *(completed 2026-02-12)*
- [ ] **Phase 4: Audio Quality & Export** - 48kHz fidelity, anti-aliasing, basic generation, WAV export
- [ ] **Phase 5: Musically Meaningful Controls** - Latent space disentanglement and PCA mapping to musical parameters
- [ ] **Phase 6: Model Persistence & Management** - Save/load models with metadata, model library
- [ ] **Phase 7: Presets & Generation History** - User workflow support for exploration
- [ ] **Phase 8: Gradio UI** - Interactive interface with sliders, playback, file management
- [ ] **Phase 9: CLI Interface** - Batch generation and scripting for power users
- [ ] **Phase 10: Multi-Format Export & Spatial Audio** - Advanced output options (MP3/FLAC/OGG, stereo field, binaural)

## Phase Details

### Phase 1: Project Setup
**Goal**: Establish development environment with PyTorch, hardware abstraction (MPS/CUDA/CPU), and foundational project structure.
**Depends on**: Nothing (first phase)
**Requirements**: UI-03, UI-04, UI-05
**Success Criteria** (what must be TRUE):
  1. Application initializes on Apple Silicon (MPS), NVIDIA (CUDA), and CPU-only hardware
  2. PyTorch 2.10.0 and TorchAudio load successfully with correct device selection
  3. Device falls back gracefully to CPU when no GPU is available
  4. Project structure exists with directories for models, datasets, and generated outputs
**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md — Project scaffolding, packaging, and configuration system
- [x] 01-02-PLAN.md — Hardware detection, memory management, and GPU benchmarking
- [x] 01-03-PLAN.md — Environment validation, startup sequence, first-run experience, and entry points

### Phase 2: Data Pipeline Foundation
**Goal**: Users can import audio files, view dataset summaries, and the system validates data integrity before training with aggressive augmentation ready.
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. User can drag-and-drop audio files (WAV, AIFF, MP3, FLAC) or import via file browser
  2. User can import a batch of files from a folder as a dataset
  3. User can view dataset summary showing file count, total duration, sample rate consistency, and waveform thumbnails
  4. System validates dataset integrity and warns about corrupt files, sample rate mismatches, or insufficient file count
  5. System applies data augmentation automatically (pitch shift, time stretch, noise injection, loudness variation) to expand training data
**Plans:** 3 plans

Plans:
- [x] 02-01-PLAN.md — Audio I/O abstraction layer and dataset validation
- [x] 02-02-PLAN.md — Augmentation pipeline and preprocessing with caching
- [x] 02-03-PLAN.md — Dataset class, summary computation, and waveform thumbnails

### Phase 3: Core Training Engine
**Goal**: Users can train a generative VAE model on small datasets (5-500 files) with overfitting prevention, progress monitoring, and checkpoint recovery.
**Depends on**: Phase 2
**Requirements**: TRAIN-01, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06
**Success Criteria** (what must be TRUE):
  1. User can train a generative model on a dataset of 5-500 audio files
  2. User can monitor training progress with loss curves, epoch/step count, and estimated time remaining
  3. User can hear sample previews generated during training at regular intervals
  4. User can cancel training and resume from the last checkpoint
  5. System saves checkpoints during training for recovery and resumption
  6. Validation loss tracks within 20% of training loss (overfitting detection working)
  7. KL divergence remains above 0.5 (posterior collapse prevented)
**Plans:** 4 plans

Plans:
- [x] 03-01-PLAN.md — Mel spectrogram representation layer, convolutional VAE model, and loss function
- [x] 03-02-PLAN.md — Training configuration with overfitting presets, PyTorch dataset, and metrics collection
- [x] 03-03-PLAN.md — Checkpoint save/load/retention and audio preview generation
- [x] 03-04-PLAN.md — Training loop, threaded runner with cancel/resume, and public API exports

### Phase 4: Audio Quality & Export
**Goal**: Users can generate high-fidelity 48kHz/24-bit audio from trained models without aliasing artifacts and export as professional-quality WAV files.
**Depends on**: Phase 3
**Requirements**: GEN-01, OUT-01, OUT-02, OUT-03, OUT-04, OUT-05
**Success Criteria** (what must be TRUE):
  1. User can generate audio from a trained model with configurable duration
  2. User can preview generated audio with waveform display and transport controls (play/pause/stop/scrub)
  3. Generated audio is 48kHz/24-bit quality with no audible aliasing artifacts
  4. User can export audio as WAV at 48kHz/24-bit (baseline)
  5. User can configure export sample rate (44.1kHz, 48kHz, 96kHz) and bit depth (16-bit, 24-bit, 32-bit float)
  6. User can configure mono or stereo output per generation
  7. Spectral analysis shows no aliasing above 20kHz
**Plans:** 3 plans

Plans:
- [ ] 04-01-PLAN.md — Anti-aliasing filter and chunk-based generation (crossfade + latent interpolation)
- [ ] 04-02-PLAN.md — Stereo processing (mid-side + dual-seed) and quality metrics (SNR + clipping)
- [ ] 04-03-PLAN.md — GenerationPipeline orchestrator, WAV export with sidecar JSON, and public API

### Phase 5: Musically Meaningful Controls
**Goal**: Users can control generation via sliders mapped to musically meaningful parameters (timbre, harmony, temporal, spatial) instead of opaque latent dimensions.
**Depends on**: Phase 4
**Requirements**: GEN-02, GEN-03, GEN-04, GEN-05, GEN-06, GEN-07, GEN-08, GEN-09
**Success Criteria** (what must be TRUE):
  1. User can control generation density (sparse ↔ dense) via slider
  2. User can control timbral parameters via sliders (brightness, warmth, roughness)
  3. User can control harmonic tension via slider (consonance ↔ dissonance)
  4. User can control temporal character via sliders (rhythmic ↔ ambient, pulse strength)
  5. User can control spatial/textural parameters via sliders (sparse ↔ dense, dry ↔ reverberant)
  6. Each slider maps to a single perceptual attribute (validated through listening tests)
  7. Parameter sliders have range limits and visual indicators to prevent broken output
  8. User can set a random seed for reproducible generation
  9. Latent space dimensions are derived from PCA/feature extraction after training
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

### Phase 6: Model Persistence & Management
**Goal**: Users can save trained models with metadata, load them for generation, and browse a model library with search and filtering.
**Depends on**: Phase 5
**Requirements**: MOD-01, MOD-02, MOD-03, MOD-05
**Success Criteria** (what must be TRUE):
  1. User can save a trained model with metadata (dataset info, training date, sample count, training parameters)
  2. User can load a previously trained model and immediately begin generating
  3. User can browse a model library with metadata and search/filter capabilities
  4. User can delete models from the library
  5. Loading a model restores both weights and latent space mappings (parameter sliders work immediately)
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

### Phase 7: Presets & Generation History
**Goal**: Users can save slider configurations as presets, recall them, and view a history of past generations with parameter snapshots and A/B comparison.
**Depends on**: Phase 6
**Requirements**: PRES-01, PRES-02, PRES-03, OUT-08, OUT-09
**Success Criteria** (what must be TRUE):
  1. User can save current slider configuration as a named preset
  2. User can recall a saved preset to restore slider positions
  3. User can browse, rename, and delete saved presets
  4. User can view a history of past generations with waveform thumbnails and parameter snapshots
  5. User can A/B compare two generations from history (play side-by-side)
**Plans**: TBD

Plans:
- [ ] 07-01: TBD

### Phase 8: Gradio UI
**Goal**: Application provides a complete Gradio-based GUI with sliders, audio playback, file management, and all core features accessible through the interface.
**Depends on**: Phase 7
**Requirements**: UI-01
**Success Criteria** (what must be TRUE):
  1. Application provides a Gradio-based GUI accessible through web browser
  2. GUI includes file upload/import for datasets
  3. GUI includes sliders for all musically meaningful parameters
  4. GUI includes audio playback with waveform display and transport controls
  5. GUI includes model management (save, load, browse library)
  6. GUI includes preset management (save, recall, browse)
  7. GUI includes generation history with thumbnails
  8. GUI provides training progress monitoring with loss curves
  9. All features from Phases 1-7 are accessible through the interface
**Plans**: TBD

Plans:
- [ ] 08-01: TBD

### Phase 9: CLI Interface
**Goal**: Application provides a command-line interface for batch generation, scripting, and headless operation.
**Depends on**: Phase 8
**Requirements**: UI-02
**Success Criteria** (what must be TRUE):
  1. Application provides a CLI for batch generation with parameter specification via arguments
  2. CLI can load models, presets, and generate audio without GUI
  3. CLI supports scripting workflows (generate multiple variations with different seeds)
  4. CLI provides progress output suitable for logging
  5. CLI can run headless on remote machines or servers
**Plans**: TBD

Plans:
- [ ] 09-01: TBD

### Phase 10: Multi-Format Export & Spatial Audio
**Goal**: Users can export audio in multiple formats (MP3, FLAC, OGG) and generate spatial audio output (stereo field, binaural).
**Depends on**: Phase 9
**Requirements**: OUT-06, OUT-07, MOD-04
**Success Criteria** (what must be TRUE):
  1. User can export audio as MP3, FLAC, or OGG in addition to WAV
  2. User can generate spatial audio output with configurable stereo field width
  3. User can generate binaural audio output for headphone listening
  4. User can load multiple models simultaneously and blend their outputs with configurable ratios
  5. Exported files maintain metadata (model name, preset name, parameters, seed)
**Plans**: TBD

Plans:
- [ ] 10-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Project Setup | 3/3 | ✓ Complete | 2026-02-12 |
| 2. Data Pipeline Foundation | 3/3 | ✓ Complete | 2026-02-12 |
| 3. Core Training Engine | 4/4 | ✓ Complete | 2026-02-12 |
| 4. Audio Quality & Export | 0/TBD | Not started | - |
| 5. Musically Meaningful Controls | 0/TBD | Not started | - |
| 6. Model Persistence & Management | 0/TBD | Not started | - |
| 7. Presets & Generation History | 0/TBD | Not started | - |
| 8. Gradio UI | 0/TBD | Not started | - |
| 9. CLI Interface | 0/TBD | Not started | - |
| 10. Multi-Format Export & Spatial Audio | 0/TBD | Not started | - |

---
*Roadmap created: 2026-02-12*
*Last updated: 2026-02-12*
