# Requirements: Small DataSet Audio

**Defined:** 2026-02-21
**Core Value:** Controllable exploration -- users can reliably navigate between sound worlds using musically meaningful parameters

## v2.0 Requirements

Requirements for complex spectrogram milestone. Each maps to roadmap phases.

### Data Pipeline

- [x] **DATA-01**: System computes magnitude + instantaneous frequency from STFT as 2-channel representation
- [x] **DATA-02**: System normalizes magnitude and IF channels independently (zero mean, unit variance)
- [x] **DATA-03**: System computes IF in mel domain to preserve existing mel-scale pipeline
- [x] **DATA-04**: System masks IF values in low-amplitude regions where phase is meaningless noise
- [x] **DATA-05**: System preprocesses and caches 2-channel spectrograms for training

### Model Architecture

- [x] **ARCH-01**: VAE encoder accepts 2-channel input (magnitude + IF)
- [x] **ARCH-02**: VAE decoder outputs 2-channel reconstruction (magnitude + IF)
- [x] **ARCH-03**: Default latent dimension is 128 (configurable)
- [x] **ARCH-04**: Decoder activation handles both magnitude (non-negative) and IF (unbounded) channels appropriately

### Loss Function

- [x] **LOSS-01**: Training uses multi-resolution STFT loss (auraloss) at multiple window sizes
- [x] **LOSS-02**: Training uses per-channel reconstruction loss (magnitude + IF)
- [x] **LOSS-03**: IF channel loss is weighted by magnitude to focus on perceptually relevant regions
- [x] **LOSS-04**: Loss terms are balanced with configurable weights

### Reconstruction

- [x] **RECON-01**: Phase is reconstructed from IF via cumulative sum
- [x] **RECON-02**: Waveform is reconstructed via ISTFT from magnitude + reconstructed phase
- [x] **RECON-03**: Griffin-Lim code is removed from the generation pipeline

### Integration

- [ ] **INTEG-01**: Generation pipeline produces audio via ISTFT (not Griffin-Lim)
- [ ] **INTEG-02**: Export pipeline works with new reconstruction (all formats: WAV/MP3/FLAC/OGG)
- [ ] **INTEG-03**: Training previews use ISTFT reconstruction
- [ ] **INTEG-04**: PCA-based latent space analysis works with 128-dim latent space
- [ ] **INTEG-05**: UI and CLI function without changes to user-facing interfaces

## Future Requirements

Deferred from active requirements. Tracked but not in current roadmap.

### Incremental Training

- **INCR-01**: User can add more training audio to an existing model
- **INCR-02**: User can feed generated outputs back into training data

### Bundled Assets

- **ASSET-01**: HRTF SOFA file bundled for binaural mode (currently requires user download)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real + imaginary representation | Requires moving from mel to linear STFT (more freq bins, bigger architecture change). Mag+IF preserves mel pipeline. |
| VQ-VAE / discrete latent space | Breaks PCA-based slider UI. Different generation paradigm. |
| Neural vocoder (HiFi-GAN/BigVGAN) | Separate model adds training complexity. Complex spectrogram eliminates Griffin-Lim without it. |
| Raw phase prediction | Notoriously hard for neural networks (wrapping, quasi-random at low energy). IF is the proven alternative. |
| v1.0 model backward compatibility | New representation is fundamentally different. Clean break. |
| Real-time generation | Quality over latency -- unchanged from v1.0 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 12 | Complete |
| DATA-02 | Phase 12 | Complete |
| DATA-03 | Phase 12 | Complete |
| DATA-04 | Phase 12 | Complete |
| DATA-05 | Phase 12 | Complete |
| ARCH-01 | Phase 13 | Complete |
| ARCH-02 | Phase 13 | Complete |
| ARCH-03 | Phase 13 | Complete |
| ARCH-04 | Phase 13 | Complete |
| LOSS-01 | Phase 14 | Complete |
| LOSS-02 | Phase 14 | Complete |
| LOSS-03 | Phase 14 | Complete |
| LOSS-04 | Phase 14 | Complete |
| RECON-01 | Phase 15 | Complete |
| RECON-02 | Phase 15 | Complete |
| RECON-03 | Phase 15 | Complete |
| INTEG-01 | Phase 16 | Pending |
| INTEG-02 | Phase 16 | Pending |
| INTEG-03 | Phase 16 | Pending |
| INTEG-04 | Phase 16 | Pending |
| INTEG-05 | Phase 16 | Pending |

**Coverage:**
- v2.0 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-02-21*
*Last updated: 2026-02-21 after roadmap creation (traceability populated)*
