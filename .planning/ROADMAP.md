# Roadmap: Small DataSet Audio

## Milestones

- ✅ **v1.0 MVP** — Phases 1-11 (shipped 2026-02-15)
- 🚧 **v1.1 HiFi-GAN Vocoder** — Phases 12-16 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-11) — SHIPPED 2026-02-15</summary>

- [x] Phase 1: Project Setup (3/3 plans) — completed 2026-02-12
- [x] Phase 2: Data Pipeline Foundation (3/3 plans) — completed 2026-02-12
- [x] Phase 3: Core Training Engine (4/4 plans) — completed 2026-02-12
- [x] Phase 4: Audio Quality & Export (3/3 plans) — completed 2026-02-12
- [x] Phase 5: Musically Meaningful Controls (2/2 plans) — completed 2026-02-13
- [x] Phase 6: Model Persistence & Management (1/1 plan) — completed 2026-02-13
- [x] Phase 7: Presets & Generation History (3/3 plans) — completed 2026-02-13
- [x] Phase 8: Gradio UI (5/5 plans) — completed 2026-02-13
- [x] Phase 9: CLI Interface (3/3 plans) — completed 2026-02-14
- [x] Phase 10: Multi-Format Export & Spatial Audio (5/5 plans) — completed 2026-02-15
- [x] Phase 11: Wire Latent Space Analysis (2/2 plans) — completed 2026-02-14

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

### 🚧 v1.1 HiFi-GAN Vocoder (In Progress)

**Milestone Goal:** Replace Griffin-Lim reconstruction with neural vocoders (BigVGAN-v2 universal + optional per-model HiFi-GAN V2) for dramatically improved audio quality, then fully remove Griffin-Lim.

- [x] **Phase 12: Vocoder Interface & BigVGAN Integration** - Abstract vocoder layer, mel adapter, BigVGAN-v2 as default reconstruction
- [x] **Phase 13: Model Persistence v2** - Model format update with optional vocoder state and backward compatibility (completed 2026-02-22)
- [x] **Phase 14: Generation Pipeline Integration** - Wire neural vocoder through all generation paths with sample rate handling (completed 2026-02-27)
- [ ] **Phase 15: UI & CLI Vocoder Controls** - Vocoder selection dropdown, download progress, and CLI flags
- [ ] **Phase 16: Per-Model HiFi-GAN Training & Griffin-Lim Removal** - Adversarial vocoder training, auto-selection, training UI/CLI, and full Griffin-Lim removal

## Phase Details

### Phase 12: Vocoder Interface & BigVGAN Integration
**Goal**: Users get dramatically better audio from every existing model with zero additional training — BigVGAN-v2 replaces Griffin-Lim as the default mel-to-waveform reconstruction
**Depends on**: Phase 11 (v1.0 complete)
**Requirements**: VOC-01, VOC-02, VOC-03, VOC-04, VOC-06
**Success Criteria** (what must be TRUE):
  1. Calling the vocoder on a VAE-produced mel spectrogram returns a waveform that sounds clearly better than Griffin-Lim output
  2. BigVGAN model weights download automatically on first use with visible progress, and are cached for subsequent runs
  3. Vocoder inference produces audio on CUDA, MPS (Apple Silicon), and CPU without error
  4. BigVGAN source code is vendored in the repository with a pinned version (not installed via pip)
  5. The mel adapter correctly converts VAE's log1p-normalized mels to BigVGAN's log-clamp format (no muffled or distorted output)
**Plans**: 3 plans
Plans:
- [x] 12-01-PLAN.md — Vendor BigVGAN source code and create vocoder interface
- [x] 12-02-PLAN.md — Implement BigVGAN vocoder wrapper and weight manager
- [x] 12-03-PLAN.md — Implement mel adapter and verify audio quality

### Phase 13: Model Persistence v2
**Goal**: The new .distillgan model format replaces .distill entirely, supports optional per-model vocoder state bundling, and v1 models are cleanly rejected with a retrain message
**Depends on**: Phase 12
**Requirements**: PERS-01, PERS-02, PERS-03
**Success Criteria** (what must be TRUE):
  1. A .distillgan model file saved with vocoder state can be loaded and the vocoder state is restored
  2. Attempting to load a v1 .distill file raises a clear error telling the user to retrain
  3. The model catalog (library) shows vocoder training stats (epochs, loss) when a model has a trained per-model vocoder
**Plans**: 2 plans
Plans:
- [ ] 13-01-PLAN.md — Update core persistence layer: constants, save/load with vocoder support, v1 rejection
- [ ] 13-02-PLAN.md — Sweep catalog, CLI, UI, and training references to .distillgan with vocoder display

### Phase 14: Generation Pipeline Integration
**Goal**: Every generation path in the application (single chunk, crossfade, latent interpolation, preview, reconstruction) uses the neural vocoder, defaults to 44.1kHz native output, and optionally resamples to 48kHz at the export boundary
**Depends on**: Phase 12, Phase 13
**Requirements**: GEN-01, GEN-02, GEN-03
**Success Criteria** (what must be TRUE):
  1. All five generation code paths (single chunk, crossfade, latent interpolation, preview, reconstruction) produce audio through the neural vocoder
  2. BigVGAN's 44.1kHz output is transparently resampled to 48kHz with no pitch shift or timing error
  3. Export pipeline (WAV/MP3/FLAC/OGG), metadata embedding, and spatial audio processing work identically with vocoder output as they did with Griffin-Lim output
**Plans**: 2 plans
Plans:
- [ ] 14-01-PLAN.md — Wire vocoder through core generation pipeline and chunking functions
- [ ] 14-02-PLAN.md — Update callers, training previews, and sample rate defaults

### Phase 15: UI & CLI Vocoder Controls
**Goal**: Users can select their vocoder and see download progress in both the Gradio web UI and the CLI
**Depends on**: Phase 14
**Requirements**: UI-01, UI-02, CLI-01, CLI-03
**Success Criteria** (what must be TRUE):
  1. Generate tab shows a vocoder selection control with options: Auto, BigVGAN Universal, Per-model HiFi-GAN
  2. When BigVGAN downloads for the first time, the UI shows download progress (not a frozen interface)
  3. Running the CLI generate command with `--vocoder bigvgan` or `--vocoder auto` selects the specified vocoder
  4. BigVGAN download progress appears as a Rich progress bar in the CLI terminal
**Plans**: TBD

### Phase 16: Per-Model HiFi-GAN Training & Griffin-Lim Removal
**Goal**: Users who want maximum fidelity can train a small per-model HiFi-GAN V2 vocoder on their specific audio, the system auto-selects the best available vocoder, and the legacy Griffin-Lim path is fully removed
**Depends on**: Phase 14, Phase 15
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, VOC-05, UI-03, UI-04, CLI-02, GEN-04
**Success Criteria** (what must be TRUE):
  1. User can train a HiFi-GAN V2 vocoder on any model's training audio via both the Train tab and the `train-vocoder` CLI command, with loss curve and progress visible
  2. Trained per-model vocoder weights are bundled into the .distill model file and persist across save/load cycles
  3. When a model has a trained per-model vocoder, the system auto-selects it over BigVGAN universal (per-model HiFi-GAN > BigVGAN)
  4. Griffin-Lim reconstruction code is fully removed — no fallback, no legacy path, neural vocoder is the only reconstruction method
  5. Training supports cancellation with checkpoint save and can resume from where it left off
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 12 → 13 → 14 → 15 → 16

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
| 12. Vocoder Interface & BigVGAN Integration | v1.1 | Complete    | 2026-02-22 | 2026-02-22 |
| 13. Model Persistence v2 | 2/2 | Complete    | 2026-02-22 | - |
| 14. Generation Pipeline Integration | 2/2 | Complete    | 2026-02-27 | - |
| 15. UI & CLI Vocoder Controls | v1.1 | 0/TBD | Not started | - |
| 16. Per-Model HiFi-GAN Training & Griffin-Lim Removal | v1.1 | 0/TBD | Not started | - |

---
*Roadmap created: 2026-02-12*
*Last updated: 2026-02-21 after Phase 13 planning*
