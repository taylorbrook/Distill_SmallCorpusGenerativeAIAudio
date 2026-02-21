# Requirements: Small Dataset Audio — v1.1 HiFi-GAN Vocoder

**Defined:** 2026-02-21
**Core Value:** Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters, generating audio that clearly comes from their own material while discovering new territory within it.

## v1.1 Requirements

Requirements for the HiFi-GAN vocoder milestone. Each maps to roadmap phases.

### Vocoder Core

- [ ] **VOC-01**: BigVGAN-v2 universal vocoder converts mel spectrograms to waveforms as the default reconstruction method
- [ ] **VOC-02**: Mel adapter converts VAE's log1p-normalized mels to BigVGAN's log-clamp format
- [ ] **VOC-03**: BigVGAN model downloads automatically on first use with progress indication
- [ ] **VOC-04**: Vocoder inference runs on CUDA, MPS (Apple Silicon), and CPU
- [ ] **VOC-05**: Vocoder auto-selects best available: per-model HiFi-GAN > BigVGAN universal
- [ ] **VOC-06**: BigVGAN source code vendored with version pinning (not pip-installed)

### Generation Pipeline

- [ ] **GEN-01**: All generation paths (single chunk, crossfade, latent interpolation, preview, reconstruction) use neural vocoder
- [ ] **GEN-02**: BigVGAN's 44.1kHz output resampled to 48kHz transparently
- [ ] **GEN-03**: Export pipeline (WAV/MP3/FLAC/OGG), metadata, and spatial audio work unchanged with vocoder output
- [ ] **GEN-04**: Griffin-Lim reconstruction code fully removed

### Model Persistence

- [ ] **PERS-01**: Model format v2 stores optional per-model HiFi-GAN vocoder state
- [ ] **PERS-02**: Existing v1.0 .sda files load without error (backward compatible)
- [ ] **PERS-03**: Model catalog indicates whether a model has a trained per-model vocoder

### User Interface

- [ ] **UI-01**: Generate tab has vocoder selection (Auto / BigVGAN Universal / Per-model HiFi-GAN)
- [ ] **UI-02**: BigVGAN download progress shown in UI on first use
- [ ] **UI-03**: Train tab has "Train Vocoder" option for models with completed VAE training
- [ ] **UI-04**: Vocoder training shows loss curve and progress in UI

### CLI

- [ ] **CLI-01**: `--vocoder` flag on generate command selects vocoder (auto/bigvgan/hifigan)
- [ ] **CLI-02**: `train-vocoder` CLI command trains per-model HiFi-GAN vocoder
- [ ] **CLI-03**: BigVGAN download progress shown via Rich progress bar

### HiFi-GAN Training

- [ ] **TRAIN-01**: User can train HiFi-GAN V2 vocoder on a model's training audio
- [ ] **TRAIN-02**: Training uses adversarial loss (MPD+MSD discriminators) + mel reconstruction loss + feature matching loss
- [ ] **TRAIN-03**: Data augmentation applied during training to prevent discriminator overfitting on small datasets
- [ ] **TRAIN-04**: Trained vocoder weights bundled into .distill model file
- [ ] **TRAIN-05**: Training supports cancel with checkpoint save and resume

## v1.2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Quality & Comparison

- **QUAL-01**: Vocoder quality comparison metrics displayed in UI (SNR, spectral convergence)
- **QUAL-02**: A/B comparison between vocoder types for same generation

### Advanced Vocoder

- **AVOC-01**: CUDA kernel optimization for BigVGAN inference (1.5-3x speedup)
- **AVOC-02**: Multiple BigVGAN model variants selectable (different quality/speed tradeoffs)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Griffin-Lim fallback | User chose full removal — neural vocoder is the only path |
| BigVGAN fine-tuning per model | 122M params causes OOM on consumer GPUs; use HiFi-GAN V2 instead |
| CUDA kernel optimization | Breaks MPS/CPU compatibility; defer to v1.2 |
| Vocos vocoder | Unofficial alpha only, no stable 48kHz model available |
| Multiple BigVGAN model variants | Ship one best model (bigvgan_v2_44khz_128band_512x) |
| 48kHz BigVGAN retraining | Infeasible without massive dataset; 44.1kHz + resample is standard practice |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| VOC-01 | — | Pending |
| VOC-02 | — | Pending |
| VOC-03 | — | Pending |
| VOC-04 | — | Pending |
| VOC-05 | — | Pending |
| VOC-06 | — | Pending |
| GEN-01 | — | Pending |
| GEN-02 | — | Pending |
| GEN-03 | — | Pending |
| GEN-04 | — | Pending |
| PERS-01 | — | Pending |
| PERS-02 | — | Pending |
| PERS-03 | — | Pending |
| UI-01 | — | Pending |
| UI-02 | — | Pending |
| UI-03 | — | Pending |
| UI-04 | — | Pending |
| CLI-01 | — | Pending |
| CLI-02 | — | Pending |
| CLI-03 | — | Pending |
| TRAIN-01 | — | Pending |
| TRAIN-02 | — | Pending |
| TRAIN-03 | — | Pending |
| TRAIN-04 | — | Pending |
| TRAIN-05 | — | Pending |

**Coverage:**
- v1.1 requirements: 25 total
- Mapped to phases: 0
- Unmapped: 25

---
*Requirements defined: 2026-02-21*
*Last updated: 2026-02-21 after initial definition*
