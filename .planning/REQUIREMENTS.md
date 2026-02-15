# Requirements: Small DataSet Audio

**Defined:** 2026-02-12
**Core Value:** Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Dataset Management

- [ ] **DATA-01**: User can import audio files via drag-and-drop or file browser (WAV, AIFF, MP3, FLAC)
- [ ] **DATA-02**: User can import a batch of files from a folder as a dataset
- [ ] **DATA-03**: User can view dataset summary: file count, total duration, sample rate consistency, and waveform thumbnails
- [ ] **DATA-04**: System validates dataset integrity before training (corrupt files, sample rate mismatches, minimum file count)

### Training

- [ ] **TRAIN-01**: User can train a generative model on a dataset of 5–500 audio files
- [ ] **TRAIN-02**: System applies data augmentation automatically (pitch shift, time stretch, noise injection, loudness variation) to expand small datasets
- [ ] **TRAIN-03**: User can monitor training progress: loss curves, epoch/step count, estimated time remaining
- [ ] **TRAIN-04**: User can hear sample previews generated during training at regular intervals
- [ ] **TRAIN-05**: User can cancel training and resume from the last checkpoint
- [ ] **TRAIN-06**: System saves checkpoints during training for recovery and resumption

### Generation

- [ ] **GEN-01**: User can generate audio from a trained model with configurable duration
- [ ] **GEN-02**: User can control generation density (sparse ↔ dense)
- [ ] **GEN-03**: User can control timbral parameters via sliders (brightness, warmth, roughness)
- [ ] **GEN-04**: User can control harmonic tension via slider (consonance ↔ dissonance)
- [ ] **GEN-05**: User can control temporal character via sliders (rhythmic ↔ ambient, pulse strength)
- [ ] **GEN-06**: User can control spatial/textural parameters via sliders (sparse ↔ dense, dry ↔ reverberant)
- [ ] **GEN-07**: System maps latent space dimensions to musically meaningful parameters using PCA/feature extraction after training
- [ ] **GEN-08**: Parameter sliders have range limits and visual indicators to prevent broken output
- [ ] **GEN-09**: User can set a random seed for reproducible generation

### Audio Output

- [ ] **OUT-01**: User can preview generated audio with waveform display and transport controls (play/pause/stop/scrub)
- [ ] **OUT-02**: User can export audio as WAV at 48kHz/24bit (baseline)
- [ ] **OUT-03**: User can configure export sample rate (44.1kHz, 48kHz, 96kHz)
- [ ] **OUT-04**: User can configure export bit depth (16-bit, 24-bit, 32-bit float)
- [ ] **OUT-05**: User can configure mono or stereo output per generation
- [ ] **OUT-06**: User can export as MP3, FLAC, or OGG in addition to WAV
- [ ] **OUT-07**: User can generate spatial audio output (stereo field, binaural)
- [ ] **OUT-08**: User can view a history of past generations with waveform thumbnails and parameter snapshots
- [ ] **OUT-09**: User can A/B compare two generations from history

### Presets

- [ ] **PRES-01**: User can save current slider configuration as a named preset
- [ ] **PRES-02**: User can recall a saved preset to restore slider positions
- [ ] **PRES-03**: User can browse, rename, and delete saved presets

### Model Management

- [ ] **MOD-01**: User can save a trained model with metadata (dataset info, training date, sample count, training parameters)
- [ ] **MOD-02**: User can load a previously trained model and immediately begin generating
- [ ] **MOD-03**: User can browse a model library with metadata and search/filter
- [ ] **MOD-04**: User can load multiple models simultaneously and blend their outputs with configurable ratios
- [ ] **MOD-05**: User can delete models from the library

### Interface

- [ ] **UI-01**: Application provides a Gradio-based GUI with sliders, audio playback, and file management
- [ ] **UI-02**: Application provides a CLI for batch generation and scripting
- [ ] **UI-03**: Application runs locally without internet connection
- [ ] **UI-04**: Application supports Apple Silicon (MPS) and NVIDIA (CUDA) GPU acceleration
- [ ] **UI-05**: Application falls back gracefully to CPU when no GPU is available

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Incremental Training

- **INCR-01**: User can add new audio files to an existing trained model without full retraining
- **INCR-02**: System mitigates catastrophic forgetting when incrementally training

### Feedback Loop

- **FEED-01**: User can select generated outputs and feed them back into a model's training data
- **FEED-02**: System supports iterative refinement cycles (generate → select → retrain → generate)

### Performance Integration

- **PERF-01**: User can map OSC/MIDI controllers to parameter sliders
- **PERF-02**: User can receive OSC/MIDI messages from a DAW for parameter automation
- **PERF-03**: Application supports MIDI learn (assign hardware knob to slider by twisting)

### Advanced Generation

- **ADV-01**: User can batch-generate multiple variations with different seeds in one operation
- **ADV-02**: User can control granular parameters (grain count, grain size, grain pitch, grain shape)
- **ADV-03**: User can generate multi-channel spatial audio (5.1, 7.1, ambisonic)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Text-to-music generation | Wrong domain — Suno/Udio territory. Requires massive datasets. Not aligned with personal small datasets |
| Built-in DAW/multi-track editor | Users have DAWs already. Tool should export to them, not replace them |
| Cloud model training | Privacy concerns with personal audio. Local-first training |
| Real-time generation (<10ms) | Quality over latency. Non-real-time allows deeper processing and higher fidelity |
| Vocal/lyric generation | Legal/ethical minefield. Outside textures/soundscapes/building blocks scope |
| Social/sharing features | Premature complexity. Users share via SoundCloud/Bandcamp |
| Mobile app | Training requires GPU. Desktop-first |
| DAW plugin (VST/AU) | Standalone tool first. Plugin integration is a separate product |
| Automatic genre classification | Personal datasets don't fit genres. Freeform metadata instead |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 2 | Pending |
| TRAIN-01 | Phase 3 | Pending |
| TRAIN-02 | Phase 2 | Pending |
| TRAIN-03 | Phase 3 | Pending |
| TRAIN-04 | Phase 3 | Pending |
| TRAIN-05 | Phase 3 | Pending |
| TRAIN-06 | Phase 3 | Pending |
| GEN-01 | Phase 4 | Pending |
| GEN-02 | Phase 5, 11 | Pending |
| GEN-03 | Phase 5, 11 | Pending |
| GEN-04 | Phase 5, 11 | Pending |
| GEN-05 | Phase 5, 11 | Pending |
| GEN-06 | Phase 5, 11 | Pending |
| GEN-07 | Phase 5, 11 | Pending |
| GEN-08 | Phase 5, 11 | Pending |
| GEN-09 | Phase 5 | Pending |
| OUT-01 | Phase 4 | Pending |
| OUT-02 | Phase 4 | Pending |
| OUT-03 | Phase 4 | Pending |
| OUT-04 | Phase 4 | Pending |
| OUT-05 | Phase 4 | Pending |
| OUT-06 | Phase 10 | Pending |
| OUT-07 | Phase 10 | Pending |
| OUT-08 | Phase 7 | Pending |
| OUT-09 | Phase 7 | Pending |
| PRES-01 | Phase 7 | Pending |
| PRES-02 | Phase 7 | Pending |
| PRES-03 | Phase 7 | Pending |
| MOD-01 | Phase 6 | Pending |
| MOD-02 | Phase 6 | Pending |
| MOD-03 | Phase 6 | Pending |
| MOD-04 | Phase 10 | Pending |
| MOD-05 | Phase 6 | Pending |
| UI-01 | Phase 8 | Pending |
| UI-02 | Phase 9 | Pending |
| UI-03 | Phase 1 | Pending |
| UI-04 | Phase 1 | Pending |
| UI-05 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 40 total
- Mapped to phases: 40
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-12*
*Last updated: 2026-02-12 after roadmap creation*
