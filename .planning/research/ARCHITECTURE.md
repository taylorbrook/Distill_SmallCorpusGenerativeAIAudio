# Architecture Research

**Domain:** Generative Audio from Small Datasets
**Researched:** 2026-02-12
**Confidence:** MEDIUM-HIGH

## Standard Architecture

### System Overview

Small-dataset generative audio systems follow a multi-stage pipeline architecture, combining data preprocessing, neural model training, latent space manipulation, and high-fidelity synthesis. The dominant pattern is a **Variational Autoencoder (VAE) or hybrid VAE-GAN architecture** with differentiable DSP components.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          UI LAYER (Gradio)                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐   │
│  │ File Upload      │  │ Parameter Sliders │  │ Audio Playback     │   │
│  │ Management       │  │ (Timbre/Harmonic) │  │ Export Controls    │   │
│  └────────┬─────────┘  └─────────┬─────────┘  └─────────┬──────────┘   │
├───────────┴────────────────┬─────┴────────────────┬──────┴──────────────┤
│                     GENERATION ENGINE                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Latent Space Navigator                                          │   │
│  │  - Maps UI sliders → latent dimensions                           │   │
│  │  - Preset save/load                                              │   │
│  │  - Interpolation & exploration                                   │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
├────────────────────────────────┴─────────────────────────────────────────┤
│                        TRAINED MODEL CORE                                │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────────┐     │
│  │   Encoder    │  │ Latent Space  │  │      Decoder             │     │
│  │  (compress   │→ │ (bottleneck   │→ │  (reconstruct/generate   │     │
│  │   to latent) │  │  disentangled)│  │   audio waveform)        │     │
│  └──────────────┘  └───────────────┘  └──────────────────────────┘     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  DSP Synthesis Components (DDSP-style, optional)                 │   │
│  │  - Harmonic Additive Synthesizer                                 │   │
│  │  - Filtered Noise Synthesizer                                    │   │
│  │  - Differentiable Reverb                                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────────┤
│                        TRAINING PIPELINE                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────┐   │
│  │ Data Loader │→ │ Feature      │→ │ Augmentation  │→ │ Training │   │
│  │             │  │ Extraction   │  │ (pitch/time/  │  │ Loop     │   │
│  │             │  │ (mel/MFCC)   │  │  noise/spec)  │  │ (VAE+GAN)│   │
│  └─────────────┘  └──────────────┘  └───────────────┘  └──────────┘   │
├──────────────────────────────────────────────────────────────────────────┤
│                        MODEL MANAGEMENT                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Checkpointing│  │ Versioning   │  │ Model Blend  │                  │
│  │ (PyTorch)    │  │ (Hub/local)  │  │ (averaging)  │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
├──────────────────────────────────────────────────────────────────────────┤
│                        DATA PIPELINE                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐   │
│  │ Audio Ingest │→ │ Normalization│→ │ Resampling   │→ │ Chunking │   │
│  │ (5-500 files)│  │ Peak/RMS     │  │ 48kHz/24bit  │  │ (segments│   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Data Pipeline** | Ingest, validate, normalize, chunk raw audio | torchaudio or librosa for I/O, preprocessing, augmentation |
| **Feature Extraction** | Extract spectral/temporal features for model input | Mel-spectrograms, MFCC, CQT via torchaudio.transforms |
| **Augmentation Module** | Expand small datasets with pitch shift, time stretch, noise injection, SpecAugment | audiomentations, torchaudio transforms, custom |
| **Training Engine** | Orchestrate model training, loss computation, optimization | PyTorch training loop, PyTorch Lightning for complex workflows |
| **Encoder** | Compress audio to low-dimensional latent representation | Convolutional or recurrent neural network layers |
| **Latent Space** | Bottleneck representation with disentangled dimensions | VAE with regularization (beta-VAE, disentanglement constraints) |
| **Decoder** | Reconstruct/generate audio from latent codes | Transposed convolutions, upsampling layers, or DSP synthesizers |
| **DSP Synthesis** | Interpretable signal generation (harmonic + noise) | DDSP modules: additive synth, subtractive filters, reverb |
| **Discriminator (GAN)** | Adversarial refinement for perceptual quality | Multi-scale or multi-period discriminators on waveform/spectrogram |
| **Latent Space Mapper** | Map user-friendly controls (timbre, harmony) to latent dims | Learned or heuristic mapping from interpretable params to z-space |
| **Generation Engine** | Non-realtime inference, parameter control, preset management | PyTorch inference mode, parameter interpolation logic |
| **Model Manager** | Checkpoint saving, versioning, incremental training, blending | PyTorch checkpointing, PyTorch Lightning callbacks, Hugging Face Hub |
| **Export Pipeline** | Format conversion, sample rate/bit depth adjustment, dithering | soundfile, torchaudio for writing WAV/FLAC with target specs |
| **UI Layer** | Sliders, file upload, playback, export controls | Gradio interfaces with Audio, Slider, File components |
| **CLI Interface** | Batch generation, scripting, automation | argparse-based Python scripts wrapping generation engine |

## Recommended Project Structure

```
src/
├── data/                  # Data pipeline
│   ├── dataset.py         # PyTorch Dataset for audio files
│   ├── augmentation.py    # Augmentation transforms for small datasets
│   ├── preprocessing.py   # Normalization, resampling, chunking
│   └── features.py        # Feature extraction (mel, MFCC, CQT)
├── models/                # Neural architecture
│   ├── encoder.py         # Encoder network (audio → latent)
│   ├── decoder.py         # Decoder network (latent → audio or DSP params)
│   ├── vae.py             # VAE wrapper combining encoder/decoder
│   ├── discriminator.py   # GAN discriminator (optional adversarial loss)
│   ├── dsp.py             # DDSP synthesis modules (harmonic, noise, reverb)
│   └── losses.py          # Reconstruction, KL, adversarial, perceptual losses
├── training/              # Training orchestration
│   ├── train.py           # Main training script
│   ├── trainer.py         # Training loop logic (or PyTorch Lightning module)
│   ├── config.py          # Hyperparameters, architecture configs
│   └── callbacks.py       # Checkpointing, early stopping, logging
├── inference/             # Generation engine
│   ├── generator.py       # Non-realtime generation from latent codes
│   ├── latent_mapper.py   # Map sliders (timbre, harmony) → latent dims
│   ├── presets.py         # Save/load slider configurations
│   └── interpolation.py   # Latent space exploration (interpolation, random walk)
├── export/                # Audio export pipeline
│   ├── converter.py       # Sample rate conversion, bit depth, dithering
│   └── formats.py         # WAV, FLAC, MP3 export
├── ui/                    # Gradio interface
│   ├── gradio_app.py      # Main Gradio app with sliders, playback, export
│   └── components.py      # Reusable UI components
├── cli/                   # Command-line interface
│   ├── train_cli.py       # CLI for training
│   ├── generate_cli.py    # CLI for batch generation
│   └── export_cli.py      # CLI for export operations
├── checkpoints/           # Model storage (gitignored)
│   └── .gitkeep
└── utils/                 # Shared utilities
    ├── audio_io.py        # Load/save audio files
    ├── device.py          # MPS/CUDA/CPU device selection
    └── logging.py         # Logging setup
```

### Structure Rationale

- **data/:** Groups all preprocessing, augmentation, and feature extraction. Small-dataset systems rely heavily on augmentation, so this is a first-class concern.
- **models/:** Modular components (encoder, decoder, VAE, discriminator, DSP) allow mix-and-match architectures. DDSP components are optional but improve interpretability.
- **training/:** Training is complex (multi-stage VAE + GAN, checkpointing, hyperparameter tuning), so isolate it from inference.
- **inference/:** Generation is a separate workflow from training. Latent mapping is the key innovation layer.
- **export/:** High-fidelity export (48kHz/24bit+) requires careful sample rate conversion and dithering, so isolate this.
- **ui/ and cli/:** Separate interfaces for interactive (Gradio) and batch (CLI) use cases.

## Architectural Patterns

### Pattern 1: Two-Stage Training (Representation + Adversarial)

**What:** Train the VAE first for reconstruction, then add adversarial loss (GAN) for perceptual refinement.

**When to use:** Always for high-quality audio synthesis. VAE alone produces blurry audio; adversarial loss sharpens it.

**Trade-offs:**
- **Pros:** Higher perceptual quality, better high-frequency detail.
- **Cons:** More complex training (GAN instability), longer training time, hyperparameter sensitivity.

**Example:**
```python
# Stage 1: Train VAE (reconstruction + KL divergence)
for epoch in range(vae_epochs):
    recon_loss = F.mse_loss(recon_audio, target_audio)
    kl_loss = compute_kl_divergence(mu, logvar)
    vae_loss = recon_loss + beta * kl_loss
    vae_loss.backward()

# Stage 2: Add GAN (adversarial loss on top of VAE)
for epoch in range(gan_epochs):
    # Generator (VAE decoder)
    recon_audio = vae.decode(latent)
    adv_loss = discriminator_loss(recon_audio, fake=False)
    total_loss = recon_loss + beta * kl_loss + lambda_adv * adv_loss
    total_loss.backward()

    # Discriminator
    d_real = discriminator(target_audio)
    d_fake = discriminator(recon_audio.detach())
    d_loss = gan_loss(d_real, d_fake)
    d_loss.backward()
```

### Pattern 2: Multi-Band Decomposition (PQMF)

**What:** Use a pseudo-quadrature mirror filter (PQMF) bank to decompose audio into 8-16 frequency bands, process in parallel, then recompose.

**When to use:** For high sample rates (48kHz) and real-time targets. RAVE uses 16-band PQMF to run 20x faster than real-time.

**Trade-offs:**
- **Pros:** Dramatically reduces computational cost (16x reduction in sequence length), enables realtime synthesis, near-perfect reconstruction.
- **Cons:** Adds complexity, requires careful filter design, slightly less interpretable than full waveform.

**Example:**
```python
from rave.pqmf import PQMF

# Decompose 48kHz waveform into 16 bands
pqmf = PQMF(attenuation=100, n_band=16)
multiband = pqmf.forward(waveform)  # [batch, 16, time/16]

# Process each band with encoder/decoder
latent = encoder(multiband)
recon_multiband = decoder(latent)

# Recompose to full waveform
recon_waveform = pqmf.inverse(recon_multiband)  # [batch, 1, time]
```

### Pattern 3: Disentangled Latent Space with Explicit Regularization

**What:** Use beta-VAE, semi-supervised constraints, or orthogonalization to ensure latent dimensions correspond to interpretable audio properties (pitch, timbre, etc.).

**When to use:** When user-controllable parameters are a core feature. Essential for musically meaningful sliders.

**Trade-offs:**
- **Pros:** Interpretable controls, predictable behavior, easier to map sliders to latent dims.
- **Cons:** Harder to train, may reduce reconstruction quality, requires domain knowledge to define disentanglement targets.

**Example:**
```python
# Beta-VAE: Increase beta to encourage disentanglement
kl_loss = compute_kl_divergence(mu, logvar)
vae_loss = recon_loss + beta * kl_loss  # beta > 1 (e.g., 4-10)

# Semi-supervised: Explicitly disentangle pitch and timbre
# Assume we have pitch labels f0 for a subset of data
pitch_pred = pitch_predictor(latent[:, 0:2])  # First 2 dims = pitch
pitch_loss = F.mse_loss(pitch_pred, f0)

timbre_latent = latent[:, 2:]  # Remaining dims = timbre
# Ensure timbre is independent of pitch via orthogonality constraint
orth_loss = orthogonality_penalty(latent[:, 0:2], latent[:, 2:])

total_loss = recon_loss + beta * kl_loss + alpha * pitch_loss + gamma * orth_loss
```

### Pattern 4: DDSP Hybrid (Neural Encoder + DSP Decoder)

**What:** Use a neural encoder to extract features/parameters, then feed them to differentiable DSP modules (harmonic synth, filtered noise, reverb) instead of a neural decoder.

**When to use:** When interpretability and computational efficiency are priorities. DDSP achieves high quality with less data (10 min per instrument).

**Trade-offs:**
- **Pros:** Highly interpretable, efficient inference, strong inductive bias (good for small datasets), fewer parameters.
- **Cons:** Less flexible than pure neural decoders, may not capture all timbral nuances, requires DSP expertise.

**Example:**
```python
from ddsp import harmonic, filtered_noise, reverb

# Encoder extracts f0, loudness, and timbre features
f0, loudness, timbre = encoder(audio)

# Map to DSP parameters
harmonic_amplitudes = timbre_to_harmonics(timbre)  # Neural net
filter_coeffs = timbre_to_filter(timbre)

# Differentiable DSP synthesis
harmonic_audio = harmonic.synthesize(f0, harmonic_amplitudes)
noise_audio = filtered_noise.synthesize(loudness, filter_coeffs)
mixed = harmonic_audio + noise_audio
output = reverb.apply(mixed)

# Train end-to-end with spectral loss
loss = spectral_loss(output, target_audio)
```

### Pattern 5: Aggressive Augmentation for Small Datasets

**What:** Apply pitch shift, time stretch, noise injection, SpecAugment, and mixup to artificially expand the dataset.

**When to use:** Always when working with <500 files. Critical for preventing overfitting.

**Trade-offs:**
- **Pros:** Dramatically improves generalization, reduces overfitting, enables training with minimal data.
- **Cons:** Can introduce artifacts if too aggressive, increases training time, may dilute original dataset characteristics.

**Example:**
```python
import torchaudio.transforms as T
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise

# Waveform-level augmentation
wave_aug = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
])

# Spectrogram-level augmentation (SpecAugment)
spec_aug = T.FrequencyMasking(freq_mask_param=10)
time_aug = T.TimeMasking(time_mask_param=20)

# Apply during data loading
waveform = wave_aug(samples=waveform, sample_rate=48000)
spectrogram = mel_transform(waveform)
spectrogram = spec_aug(time_aug(spectrogram))
```

### Pattern 6: Transfer Learning from Pretrained Models

**What:** Start from a pretrained audio model (e.g., HuBERT, Wav2Vec2, pretrained RAVE) and fine-tune on the small dataset.

**When to use:** When <100 files or when training from scratch fails. Especially effective for similar audio domains (speech models → voice, music models → instruments).

**Trade-offs:**
- **Pros:** Requires less data, faster training, better generalization, leverages large-scale pretraining.
- **Cons:** Pretrained models may not match target domain perfectly, requires finding/adapting compatible architectures.

**Example:**
```python
# Load pretrained RAVE model
from rave import RAVE
pretrained = RAVE.from_pretrained("acids-ircam/rave-vintage")

# Freeze encoder, fine-tune decoder on new dataset
for param in pretrained.encoder.parameters():
    param.requires_grad = False

# Train only decoder + latent space with new data
optimizer = torch.optim.Adam(pretrained.decoder.parameters(), lr=1e-4)

for batch in small_dataset:
    latent = pretrained.encode(batch)  # Frozen encoder
    recon = pretrained.decode(latent)  # Trainable decoder
    loss = reconstruction_loss(recon, batch)
    loss.backward()
    optimizer.step()
```

## Data Flow

### Training Flow (Audio → Trained Model)

```
[Raw Audio Files (5-500)]
    ↓
[Preprocessing]
- Load with torchaudio/librosa (48kHz/24bit → float32 [-1,1])
- Normalize (peak or RMS)
- Chunk into fixed-length segments (e.g., 2-4 sec)
    ↓
[Augmentation]
- Pitch shift (±2 semitones)
- Time stretch (0.9-1.1x)
- Noise injection (SNR 40-60dB)
- SpecAugment (frequency/time masking)
    ↓
[Feature Extraction] (Optional for DDSP/hybrid)
- Mel-spectrogram (n_mels=128, hop_length=256)
- Fundamental frequency (f0) via CREPE/YIN
- Loudness (A-weighted RMS)
    ↓
[Training Loop]
- Forward pass: waveform → encoder → latent (mu, logvar) → decoder → recon
- Loss computation:
  * Reconstruction (spectral loss: multi-scale STFT + L1/L2)
  * KL divergence (for VAE regularization)
  * Adversarial loss (if GAN discriminator enabled)
  * Perceptual loss (optional: pretrained feature matching)
- Backward pass, optimizer step
- Checkpointing every N epochs
    ↓
[Trained Model Checkpoint]
- encoder.pt, decoder.pt, config.yaml
- Latent space metadata (dimension meanings, ranges)
```

### Generation Flow (Sliders → Audio)

```
[User Adjusts Sliders]
- Timbre (brightness, warmth, roughness)
- Harmonic (pitch class, harmonic content)
- Temporal (attack, decay, rhythm)
- Spatial (stereo width, reverb amount)
    ↓
[Latent Mapper]
- Map slider values to latent space coordinates
- Options:
  * Learned mapping (train small MLP: sliders → z)
  * Heuristic mapping (manually assign dims based on disentanglement analysis)
  * PCA/t-SNE projection (for 2D exploration)
    ↓
[Latent Code z]
- N-dimensional vector (e.g., 16-128 dims)
- Optionally: blend multiple presets, interpolate, add noise for variation
    ↓
[Decoder (Inference Mode)]
- z → decoder → audio waveform or DSP parameters
- If DDSP: decoder outputs (f0, amplitudes, filter coeffs) → DSP synth → waveform
- Non-realtime: no streaming, full context, high quality
    ↓
[Post-Processing]
- Normalize to target peak level (-1dBFS typical)
- Apply fade in/out if needed
- Stereo widening, final reverb (if not in model)
    ↓
[Export Pipeline]
- Convert to target sample rate (if ≠ 48kHz) using high-quality resampler
- Convert to target bit depth (16/24/32 bit) with dithering if reducing bit depth
- Save as WAV/FLAC/MP3
    ↓
[Audio File (48kHz/24bit WAV)]
```

### Key Data Flows

1. **Audio Ingestion → Training:** Raw files are preprocessed, augmented heavily (due to small dataset), then fed to VAE training loop. Checkpoints saved periodically.

2. **Sliders → Latent → Audio:** UI sliders map to latent dimensions (via learned or heuristic mapper), latent code is decoded to audio, then exported. This is the core user interaction loop.

3. **Incremental Training:** User adds new audio files → preprocess → continue training from last checkpoint → updated model. Model versioning tracks improvements.

4. **Preset System:** Current slider positions → save as JSON → load later → map to latent → generate. Enables reproducibility and exploration.

## Build Order and Risk Assessment

### Suggested Build Order (Lowest Risk → Highest Risk)

| Phase | Component | Why Build This Order | Risk Level |
|-------|-----------|---------------------|------------|
| 1 | **Data Pipeline** | Foundation. Can't train without it. Validates data quality early. | Low |
| 2 | **Feature Extraction** | Needed for both training and DDSP. Straightforward with torchaudio. | Low |
| 3 | **Basic VAE (No GAN)** | Simplest trainable model. Proves training loop works. May produce blurry audio but validates architecture. | Low-Medium |
| 4 | **Augmentation Module** | Essential for small datasets. Add after basic VAE works to improve generalization. | Low |
| 5 | **Checkpointing & Model Management** | Prevents losing training progress. Add before long training runs. | Low |
| 6 | **Gradio UI (Basic)** | Early user feedback. Start with simple file upload, play generated audio. | Low |
| 7 | **GAN Discriminator** | Adds perceptual quality. Riskier (training instability). Prototype first. | Medium-High |
| 8 | **Latent Space Mapping (Sliders)** | Core innovation. High risk if latent space isn't disentangled. Requires experimentation. | High |
| 9 | **DDSP Synthesis (Optional)** | Alternative to neural decoder. High reward (interpretability) but requires DSP expertise. | Medium |
| 10 | **Transfer Learning** | Fallback if training from scratch fails. Lower risk if pretrained model matches domain. | Medium |
| 11 | **Export Pipeline (High-Fidelity)** | Polish. Sample rate conversion, dithering are well-understood but finicky. | Low-Medium |
| 12 | **CLI & Batch Generation** | Convenience. Low risk, builds on working generation engine. | Low |

### Component Risk Assessment

| Component | Risk | Why Risky? | Mitigation Strategy |
|-----------|------|-----------|---------------------|
| **Latent Space Mapping** | HIGH | No guarantee latent dims will align with musical concepts. May require extensive trial/error. | Prototype beta-VAE and semi-supervised disentanglement early. Visualize latent space with t-SNE. Test with 2D latent first. |
| **GAN Training** | MEDIUM-HIGH | Mode collapse, instability, hyperparameter sensitivity. | Start with pretrained discriminator or simple architecture. Use spectral normalization, gradient penalties. Monitor carefully. |
| **Small Dataset Overfitting** | MEDIUM | 5-500 files is tiny. Model may memorize instead of generalize. | Aggressive augmentation, transfer learning, early stopping, validation split. |
| **DDSP Implementation** | MEDIUM | Requires DSP knowledge. Differentiable modules need careful gradient flow. | Use existing DDSP library (Magenta). Start with harmonic synth only, add noise/reverb later. |
| **High Sample Rate (48kHz)** | MEDIUM | Longer sequences, more memory, slower training. | Use PQMF multi-band decomposition (16 bands → 16x speedup). Start with 24kHz, scale up. |
| **Disentanglement** | MEDIUM | Hard to achieve without labels. Beta-VAE may hurt reconstruction. | Try multiple approaches: beta-VAE (beta=4-10), semi-supervised (label pitch manually for subset), group-supervised (timbre families). |
| **Inference Speed** | LOW-MEDIUM | Non-realtime is acceptable, but very slow generation hurts UX. | Use model.eval(), torch.no_grad(), optimize with torch.jit if needed. Consider smaller models. |
| **Apple Silicon (MPS)** | LOW | MPS backend is mature in PyTorch 2.x, but some ops may not be supported. | Test early on target hardware. Fallback to CPU if MPS fails. Use `device = torch.device("mps" if torch.has_mps else "cpu")`. |

### Critical Path (Must Work)

1. **Data Pipeline + Augmentation:** Can't train without data. Small dataset means augmentation is mandatory.
2. **VAE Training:** Core architecture. If this doesn't converge, nothing else matters.
3. **Latent Space Mapping:** The innovation. If latent space is uninterpretable, the entire premise fails.
4. **Gradio UI:** User interaction. If sliders don't control audio meaningfully, product has no value.

### Nice-to-Have (Can Defer)

- **GAN Refinement:** VAE alone may suffice for MVP. Add GAN if audio quality is insufficient.
- **DDSP Synthesis:** Neural decoder is simpler. DDSP is a performance/interpretability optimization.
- **Transfer Learning:** Only needed if training from scratch fails.
- **CLI:** Batch generation is useful but not essential for initial validation.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Training on Raw Waveforms at 48kHz Without Multi-Band Decomposition

**What people do:** Directly encode/decode full 48kHz waveforms, leading to massive sequence lengths (e.g., 4 sec = 192k samples).

**Why it's wrong:** Extremely slow training, massive memory usage, gradient vanishing over long sequences, unnecessary detail at high frequencies.

**Do this instead:** Use PQMF to decompose into 16 bands (12x speedup, near-perfect reconstruction). Or train at 24kHz, then upsample with a neural vocoder if needed.

**Example:**
```python
# BAD: Direct encoding of 48kHz waveform
waveform = load_audio("file.wav")  # [1, 192000] for 4 sec
latent = encoder(waveform)  # Extremely slow, huge memory

# GOOD: Multi-band decomposition
from rave.pqmf import PQMF
pqmf = PQMF(n_band=16)
multiband = pqmf.forward(waveform)  # [16, 12000] - 16x smaller
latent = encoder(multiband)  # Much faster
```

### Anti-Pattern 2: Using Only Reconstruction Loss (L1/L2) for Audio

**What people do:** Train VAE with only MSE or L1 loss on waveforms or spectrograms.

**Why it's wrong:** Produces blurry, muffled audio. Misses perceptual quality. L1/L2 optimize for average, not sharpness.

**Do this instead:** Use multi-scale spectral loss (multiple STFT window sizes) + adversarial loss (GAN) + perceptual loss (feature matching).

**Example:**
```python
# BAD: L1 waveform loss only
recon_loss = F.l1_loss(recon_audio, target_audio)

# GOOD: Multi-scale spectral + adversarial
spectral_loss = multi_scale_stft_loss(recon_audio, target_audio,
                                       fft_sizes=[512, 1024, 2048])
adv_loss = discriminator_loss(recon_audio)
perceptual_loss = feature_matching_loss(recon_audio, target_audio)
total_loss = spectral_loss + lambda_adv * adv_loss + lambda_perc * perceptual_loss
```

### Anti-Pattern 3: Assuming Latent Dimensions are Interpretable by Default

**What people do:** Train a standard VAE, then try to map sliders to random latent dimensions.

**Why it's wrong:** Latent dims are entangled by default. Changing one dim affects multiple audio properties unpredictably.

**Do this instead:** Use disentanglement regularization (beta-VAE, orthogonality constraints, semi-supervised labels). Analyze latent space post-training. Accept that some dims may not be interpretable.

**Example:**
```python
# BAD: Assume latent[0] = timbre, latent[1] = pitch (no basis)
z = torch.zeros(latent_dim)
z[0] = timbre_slider  # Hope it controls timbre (it won't)
z[1] = pitch_slider

# GOOD: Train with disentanglement, then analyze
# During training: beta-VAE with beta > 1
kl_loss = compute_kl_divergence(mu, logvar)
vae_loss = recon_loss + beta * kl_loss  # beta = 4-10

# Post-training: Analyze latent traversals
for dim in range(latent_dim):
    z = torch.zeros(latent_dim)
    z[dim] = torch.linspace(-3, 3, 10)  # Traverse dim
    audios = decoder(z)
    # Listen and label: "dim 3 controls brightness, dim 7 controls pitch"
```

### Anti-Pattern 4: No Augmentation or Minimal Augmentation with <100 Files

**What people do:** Train on 50 audio files with no augmentation or only pitch shift.

**Why it's wrong:** Model will memorize the dataset perfectly (overfitting) and fail to generalize. Validation loss diverges from training loss.

**Do this instead:** Apply aggressive augmentation: pitch shift, time stretch, noise injection, SpecAugment, mixup. Aim for 10-100x effective dataset expansion.

**Example:**
```python
# BAD: No augmentation
dataset = AudioDataset(files)  # 50 files, no transforms

# GOOD: Aggressive augmentation
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise
aug = Compose([
    PitchShift(min_semitones=-3, max_semitones=3, p=0.7),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.7),
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
])
dataset = AudioDataset(files, transform=aug)
# Also add SpecAugment during training
```

### Anti-Pattern 5: Ignoring Perceptual Quality (Only Optimizing for Loss Metrics)

**What people do:** Celebrate low reconstruction loss (MSE, spectral distance) without listening to generated audio.

**Why it's wrong:** Low loss ≠ good audio. Models can have great metrics but sound terrible (artifacts, muffled, unnatural).

**Do this instead:** Listen to outputs frequently. Use perceptual metrics (ViSQOL, PESQ) alongside loss metrics. Get human feedback. Prioritize subjective quality over loss numbers.

**Example:**
```python
# BAD: Only log numerical loss
print(f"Epoch {epoch}, Loss: {loss.item()}")

# GOOD: Save audio samples every N epochs and listen
if epoch % 10 == 0:
    with torch.no_grad():
        sample_audio = decoder(sample_latent)
        torchaudio.save(f"samples/epoch_{epoch}.wav", sample_audio, 48000)
    # Listen and assess: "Epoch 50 sounds better than Epoch 40, even though loss is similar"
```

### Anti-Pattern 6: Premature Real-Time Optimization

**What people do:** Focus on real-time inference before the model even generates good audio.

**Why it's wrong:** Real-time is unnecessary for this use case (non-realtime synthesis). Adds complexity, limits architecture choices, distracts from core goal (quality + controllability).

**Do this instead:** Focus on quality and controllability first. Optimize inference speed only if generation time becomes a UX issue (e.g., >30 sec per sample). Non-realtime is fine.

**Example:**
```python
# BAD: Limit model size and features to hit real-time targets
# "Model must run on CPU in <10ms" (unnecessary constraint)

# GOOD: Prioritize quality, use GPU, accept seconds of latency
with torch.no_grad():
    audio = decoder(latent)  # Takes 2 sec on GPU, that's fine for non-realtime use
```

## Integration Points

### External Libraries

| Library | Integration Pattern | Notes |
|---------|---------------------|-------|
| **torchaudio** | Audio I/O, feature extraction, augmentation | Use torchaudio.load, torchaudio.save, torchaudio.transforms. GPU-accelerated transforms. |
| **librosa** | Fallback for I/O, some feature extraction | Slower than torchaudio but more features (e.g., onset detection, tempo). |
| **audiomentations** | Waveform augmentation | Compose transforms. Faster than torchaudio for some augs (pitch shift, time stretch). |
| **DDSP (Magenta)** | Differentiable DSP modules | Import ddsp.core, ddsp.losses. Integrate as decoder replacement or parallel path. |
| **Gradio** | UI components | Use gr.Audio, gr.Slider, gr.File. Launch with app.launch(). |
| **Hugging Face Hub** | Model versioning and sharing | Use huggingface_hub.upload_file for checkpoints. Download with from_pretrained. |
| **PyTorch Lightning** | Training orchestration (optional) | Use LightningModule for complex training loops. Callbacks for checkpointing, logging. |
| **soundfile** | High-quality audio I/O | Alternative to torchaudio for FLAC, bit depth control. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **Data Pipeline ↔ Training** | PyTorch DataLoader yields batches of augmented spectrograms/waveforms | Prefetch with num_workers > 0. Pin memory for GPU efficiency. |
| **Training ↔ Model Management** | Callbacks or manual saving: `torch.save(model.state_dict(), checkpoint_path)` | Save every N epochs. Include optimizer state for resume. |
| **Model ↔ Generation Engine** | Load checkpoint: `model.load_state_dict(torch.load(path))`, then `model.eval()` | Use torch.no_grad() for inference. Device mapping (MPS/CUDA/CPU). |
| **Generation ↔ Latent Mapper** | Latent mapper outputs z, generator decodes z → audio | Mapper can be learned (MLP) or heuristic (manual assignment). |
| **Generation ↔ Export Pipeline** | Pass waveform tensor to export module → convert format/SR/bit depth | Ensure sample rate matches model output (e.g., 48kHz). Apply dithering if reducing bit depth. |
| **UI ↔ Generation** | Gradio calls generation function with slider values, receives audio file path | Use Gradio's `inputs=[slider1, slider2, ...]`, `outputs=gr.Audio()`. |
| **CLI ↔ Training/Generation** | argparse parses CLI args, calls training or generation functions | Shared config files (YAML) for reproducibility. |

## Scaling Considerations

| Concern | Small Scale (5-50 files) | Medium Scale (50-500 files) | Large Scale (500+ files) |
|---------|--------------------------|----------------------------|--------------------------|
| **Training Time** | Minutes to hours on laptop GPU (MPS/RTX 3060) | Hours on single GPU (A100, 4090) | May need multi-GPU or cloud (GCP, AWS) |
| **Overfitting Risk** | VERY HIGH - Aggressive augmentation mandatory, early stopping | HIGH - Augmentation + validation split + regularization | MODERATE - Standard techniques |
| **Model Complexity** | Small models (latent_dim=16-32, shallow encoder/decoder) | Medium models (latent_dim=64-128, deeper networks) | Can use larger models (latent_dim=128-256) |
| **Augmentation** | 100x expansion via augmentation (pitch, time, noise, mixup) | 10-50x expansion | 2-10x expansion |
| **Transfer Learning** | RECOMMENDED - Start from pretrained model (RAVE, HuBERT) | OPTIONAL - Helps but not mandatory | OPTIONAL |
| **Data Storage** | Local disk (<1GB) | Local disk (1-10GB) | Cloud storage (S3, GCS) if needed |
| **Inference** | Single-threaded, seconds per sample on CPU/MPS is fine | Single-threaded, GPU recommended | Batch inference, GPU required |

### Scaling Priorities

1. **First bottleneck (5-50 files):** Overfitting. Model memorizes dataset. Generalization fails.
   - **Fix:** Aggressive augmentation (10-100x), transfer learning, beta-VAE regularization, early stopping.

2. **Second bottleneck (50-500 files):** Training time. Hours to days on single GPU.
   - **Fix:** Use PQMF multi-band (16x speedup), smaller models, mixed precision (torch.cuda.amp), multi-GPU if available.

3. **Third bottleneck (500+ files):** Data management. Loading/augmenting hundreds of files is slow.
   - **Fix:** DataLoader with num_workers=4-8, prefetching, cache preprocessed features to disk.

## Sources

### High Confidence (Official Docs, GitHub Repos)

- [RAVE GitHub Repository](https://github.com/acids-ircam/RAVE) - Official RAVE implementation, architecture details
- [DDSP Official Documentation](https://magenta.tensorflow.org/ddsp) - DDSP architecture and components
- [PyTorch Audio Documentation](https://docs.pytorch.org/audio/stable/index.html) - Torchaudio official docs
- [PyTorch Lightning Checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html) - Model checkpointing best practices

### Medium Confidence (Research Papers, Web Search)

- [RAVE Paper (arXiv 2111.05011)](https://arxiv.org/abs/2111.05011) - RAVE architecture and training methodology
- [DDSP Paper (arXiv 2001.04643)](https://arxiv.org/abs/2001.04643) - DDSP system design and data flow
- [Pitch-Conditioned Instrument Sound Synthesis (arXiv 2510.04339)](https://arxiv.org/abs/2510.04339) - 2D latent space disentanglement for timbre
- [Learning Interpretable Features in Audio Latent Spaces (arXiv 2510.23802)](https://arxiv.org/abs/2510.23802) - Sparse autoencoders for interpretable audio features
- [Is Disentanglement Enough? (arXiv 2108.01450)](https://arxiv.org/abs/2108.01450) - Latent space disentanglement for controllable music generation
- [Automated Data Augmentation for Audio Classification (IEEE TASLP 2024)](https://dl.acm.org/doi/10.1109/TASLP.2024.3402049) - Audio augmentation techniques
- [Sample Rate & Bit Depth Guide](https://www.blackghostaudio.com/blog/sample-rate-bit-depth-explained) - Export pipeline best practices

### Low Confidence (WebSearch Only, Needs Verification)

- General audio synthesis architecture patterns - Inferred from multiple sources but not explicitly documented in a single authoritative reference
- Component boundary recommendations - Based on common patterns observed across projects, not standardized
- Build order prioritization - Derived from experience patterns, not empirically validated for this specific domain

---

*Architecture research for: Generative Audio from Small Datasets*
*Researched: 2026-02-12*
