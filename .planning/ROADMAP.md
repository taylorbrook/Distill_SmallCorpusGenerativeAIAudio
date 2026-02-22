# Roadmap: Small DataSet Audio

## Milestones

- ✅ **v1.0 MVP** — Phases 1-11 (shipped 2026-02-15)
- 🚧 **v2.0 Complex Spectrogram** — Phases 12-16 (in progress)

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

### 🚧 v2.0 Complex Spectrogram (In Progress)

**Milestone Goal:** Replace magnitude-only mel spectrogram pipeline with 2-channel magnitude + instantaneous frequency representation, eliminating Griffin-Lim reconstruction entirely via exact ISTFT.

- [x] **Phase 12: 2-Channel Data Pipeline** - Compute magnitude + IF spectrograms with normalization, IF masking, and caching (completed 2026-02-22)
- [ ] **Phase 13: 2-Channel VAE Architecture** - Adapt encoder/decoder for 2-channel input/output with appropriate activations and 128-dim latent space
- [ ] **Phase 14: Multi-Resolution Loss** - Add auraloss multi-resolution STFT loss with magnitude-weighted IF loss and configurable balancing
- [ ] **Phase 15: ISTFT Reconstruction** - Reconstruct phase from IF via cumulative sum and produce waveforms via ISTFT, removing Griffin-Lim
- [ ] **Phase 16: Full Pipeline Integration** - Wire ISTFT generation through export, training previews, PCA analysis, UI, and CLI

## Phase Details

### Phase 12: 2-Channel Data Pipeline
**Goal**: Training data exists as 2-channel magnitude + instantaneous frequency spectrograms ready for the VAE
**Depends on**: Nothing (first v2.0 phase; builds on v1.0 data pipeline)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. Loading an audio file produces a 2-channel tensor where channel 0 is magnitude and channel 1 is instantaneous frequency, both computed in mel domain
  2. Magnitude and IF channels are independently normalized (zero mean, unit variance) with stored statistics for denormalization
  3. IF values in low-amplitude bins (where phase is meaningless noise) are masked to zero
  4. Preprocessed 2-channel spectrograms are cached to disk and reloaded without recomputation on subsequent training runs
**Plans**: 2 plans
Plans:
- [ ] 12-01-PLAN.md — Core 2-channel spectrogram computation (magnitude + IF in mel domain, normalization, IF masking, config)
- [ ] 12-02-PLAN.md — Cache pipeline and training integration (preprocessing, caching, manifest, change detection, training loop wiring)

### Phase 13: 2-Channel VAE Architecture
**Goal**: The VAE model accepts 2-channel input and produces 2-channel output with appropriate per-channel handling
**Depends on**: Phase 12
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Success Criteria** (what must be TRUE):
  1. The VAE encoder accepts 2-channel spectrograms (magnitude + IF) and encodes them to a 128-dimensional latent space
  2. The VAE decoder produces 2-channel output where magnitude channel is non-negative and IF channel is unbounded
  3. The default latent dimension is 128 (configurable), and a round-trip encode-decode of a 2-channel input produces a 2-channel output of matching shape
  4. Model instantiation with in_channels=2 and latent_dim=128 succeeds on all supported devices (CPU, CUDA, MPS)
**Plans**: TBD

### Phase 14: Multi-Resolution Loss
**Goal**: Training uses perceptually grounded loss combining multi-resolution STFT loss with per-channel reconstruction loss weighted by magnitude
**Depends on**: Phase 13
**Requirements**: LOSS-01, LOSS-02, LOSS-03, LOSS-04
**Success Criteria** (what must be TRUE):
  1. Training loss includes multi-resolution STFT loss (auraloss) computed at 3+ window sizes alongside per-channel reconstruction loss
  2. IF channel reconstruction loss is weighted by magnitude so that errors in low-energy (inaudible) regions contribute less
  3. Loss term weights (STFT vs. per-channel, magnitude vs. IF, KL) are configurable via training config without code changes
  4. Training on a small dataset (5-20 files) converges with the combined loss (loss decreases over epochs, no NaN/divergence)
**Plans**: TBD

### Phase 15: ISTFT Reconstruction
**Goal**: Generated spectrograms are converted to audio waveforms via ISTFT using phase reconstructed from instantaneous frequency, with no Griffin-Lim dependency
**Depends on**: Phase 12, Phase 14
**Requirements**: RECON-01, RECON-02, RECON-03
**Success Criteria** (what must be TRUE):
  1. Phase is reconstructed from the IF channel by cumulative sum along the time axis, producing a continuous phase signal
  2. A waveform is produced by combining denormalized magnitude with reconstructed phase and applying ISTFT -- no Griffin-Lim anywhere in the pipeline
  3. All Griffin-Lim code paths are removed or disabled; attempting to use Griffin-Lim reconstruction raises an error or is simply absent
**Plans**: TBD

### Phase 16: Full Pipeline Integration
**Goal**: All user-facing workflows (generation, export, training previews, slider controls, UI, CLI) work end-to-end with the new 2-channel ISTFT pipeline
**Depends on**: Phase 13, Phase 15
**Requirements**: INTEG-01, INTEG-02, INTEG-03, INTEG-04, INTEG-05
**Success Criteria** (what must be TRUE):
  1. Generating audio from a trained v2.0 model produces a waveform via ISTFT that can be played back in the UI and exported in all formats (WAV/MP3/FLAC/OGG)
  2. Training preview audio (generated during training for monitoring) uses ISTFT reconstruction, not Griffin-Lim
  3. PCA-based latent space analysis works correctly with the 128-dimensional latent space, producing musically meaningful slider controls
  4. The Gradio UI and CLI function identically to v1.0 from the user's perspective -- no changes to user-facing interfaces, controls, or workflows
  5. Export pipeline produces valid audio files in all supported formats with correct metadata
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
| 12. 2-Channel Data Pipeline | 2/2 | Complete    | 2026-02-22 | - |
| 13. 2-Channel VAE Architecture | v2.0 | 0/? | Not started | - |
| 14. Multi-Resolution Loss | v2.0 | 0/? | Not started | - |
| 15. ISTFT Reconstruction | v2.0 | 0/? | Not started | - |
| 16. Full Pipeline Integration | v2.0 | 0/? | Not started | - |

---
*Roadmap created: 2026-02-12*
*Last updated: 2026-02-21 after Phase 12 planning (2 plans)*
