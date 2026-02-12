# Technology Stack

**Project:** Small Dataset Generative Audio
**Researched:** 2026-02-12
**Confidence:** MEDIUM-HIGH

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch | 2.10.0 | Deep learning framework | Industry standard for research-grade audio ML. Native MPS support for Apple Silicon, excellent CUDA support, and flexible enough for novel architectures. Version 2.10.0 is latest stable as of Feb 2026. |
| TorchAudio | 2.10.0 | Audio I/O and transforms | Official PyTorch audio library. Provides efficient spectrograms, resampling, and integrates seamlessly with PyTorch tensors. Transitioned to maintenance phase, focusing on core audio processing. |
| PyTorch Lightning | 2.6.1 | Training orchestration | Reduces boilerplate for distributed training, multi-GPU support, experiment logging. Latest version (Jan 30, 2026) provides production-ready features for complex training workflows. |

### Generative Model Architecture

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| RAVE | 2.3.1 | VAE-based audio synthesis | Real-time Audio Variational autoEncoder specifically designed for audio. v2 architecture offers faster processing and higher quality than v1. Proven for small datasets (2-3 hours minimum). Use v2_small config (8GB GPU) for development, v2 for production (16GB GPU). **Caveat:** Latent controls are opaque - you'll need to build disentanglement on top. |
| Stable Audio Tools | Latest (requires PyTorch 2.5+) | Latent diffusion framework | Production-grade latent diffusion for audio. Includes pretrained VAEs, U-Net diffusion models, and supports fine-tuning. Open source (MIT), actively maintained by Stability AI. Enables conditioning on text/metadata and duration control. **Best for:** Fine-tuning pretrained models on small datasets rather than training from scratch. |
| DDSP | Latest (Magenta) | Differentiable DSP | Use for interpretable synthesis. Trains on <13 minutes of audio. Explicitly models harmonic and noise components, making it naturally more interpretable than pure neural approaches. **Trade-off:** Limited to monophonic or simple timbres, but parameters are inherently musical. |

**Recommendation:** Start with RAVE v2_small for rapid prototyping, then layer disentanglement techniques. Consider DDSP for interpretable baseline comparisons and Stable Audio Tools if you can leverage pretrained models via transfer learning.

### Neural Audio Codec

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| DAC (Descript Audio Codec) | 1.0.0 | High-fidelity audio compression | State-of-the-art neural codec with ~90x compression while preserving quality. Supports 44.1kHz (your target is 48kHz, this is close). Works across speech, music, environmental audio. MIT licensed. Operates at 8 kbps. **Use for:** Compressing training data or as a perceptual bottleneck in your architecture. |
| EnCodec | Latest | Alternative neural codec | Facebook's codec, supports stereo 48kHz (matches your requirement exactly). Use if you need exact 48kHz support. **Trade-off:** DAC has better quality per bitrate, but EnCodec has exact sample rate match. |

**Recommendation:** Use DAC for general work (44.1kHz is close enough for experimentation). Switch to EnCodec if 48kHz becomes a hard requirement or you need the perceptual features from Meta's ecosystem.

### Audio Processing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| librosa | 0.11.0+ | Audio analysis and feature extraction | De facto standard for music/audio feature extraction (MFCCs, spectral features, onset detection). Well-documented, widely used in research. |
| soundfile | Latest | Audio I/O | Efficient reading/writing of WAV, FLAC, OGG. Simpler interface than librosa for pure I/O. Recommended over librosa.load for production pipelines. |
| pedalboard | 0.9.22+ | Audio effects and augmentation | Spotify's audio effects library. 300x faster than pySoX, 4x faster file loading than librosa.load. Use for data augmentation (EQ, compression, reverb) to artificially expand small datasets. Supports VST3/Audio Unit plugins for creative effects. |

**Recommendation:** Use soundfile for I/O, librosa for analysis/features, pedalboard for augmentation. This combo gives you speed (pedalboard) + research features (librosa) + simplicity (soundfile).

### Training and Optimization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| auraloss | 0.2.0+ | Audio-specific loss functions | PyTorch package with time-domain (ESR, SI-SDR, SNR) and frequency-domain (multi-resolution STFT, Mel-STFT) losses designed for audio. Critical for perceptually meaningful training. Use multi-resolution STFT loss for high-fidelity reconstruction. |
| einops | 0.8.2 | Tensor manipulation | Readable tensor operations (rearrange, reduce, repeat). Essential for complex audio tensor shapes (batch, channels, time, frequency). Latest version (Jan 26, 2026) supports all major frameworks. |
| Hydra | 1.3+ | Configuration management | Facebook's hierarchical config system. Essential for managing experiments with multiple hyperparameters, model architectures, and dataset variations. Enables reproducible research and rapid iteration. |

### Latent Space Disentanglement

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Beta-VAE / Factor-VAE | Custom implementation | Disentangled representations | Your core innovation area. Beta-VAE pressures latent space toward factorial prior via weighted KL term. Factor-VAE explicitly penalizes total correlation. **Recent research (2025):** Shows beta-VAE can collapse mutual information under heavy regularization - you'll need careful tuning. Start with beta=4-10, monitor reconstruction vs. disentanglement trade-off. |
| SD-Codec approach | Research paper reference | Source-disentangled codec | 2025 ICASSP paper shows joint audio coding + source separation by assigning different domains to distinct codebooks. **Apply to your case:** Separate timbral, harmonic, temporal dimensions into different codebook groups for explicit control. |

**Recommendation:** Implement Beta-VAE first (simpler), then experiment with SD-Codec's multi-codebook approach if single-latent disentanglement fails. Monitor with qualitative latent traversals and quantitative disentanglement metrics.

### Experiment Tracking

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Weights & Biases | Latest | Experiment tracking and visualization | Industry standard for ML experiment tracking. Rich media support (audio, spectrograms). Real-time metric logging. Required by Stable Audio Tools. Free tier available. |
| MLflow | Latest | Alternative tracking | Open source alternative to W&B. Self-hosted option if you prefer local control. Less audio-specific features but more flexible for custom metrics. |

**Recommendation:** Start with W&B for rapid iteration and rich audio visualization. Switch to MLflow only if you need self-hosting or have privacy constraints.

### User Interface

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Gradio | 6.5.1 | Interactive UI | Latest version (Jan 29, 2026). Built-in audio components with waveform display, microphone input, streaming support. Perfect for rapid prototyping of audio interfaces. Supports real-time parameter manipulation via sliders. Integrates with Hugging Face Spaces for easy sharing. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Jupyter Lab | Interactive development | Standard for audio ML research. Use for exploratory analysis, feature visualization. |
| VSCode + Python extension | Code editor | Excellent PyTorch debugging, notebook support, Git integration. |
| FFmpeg | Audio conversion | Required by RAVE and many audio libraries. Install via conda or system package manager. |
| Git LFS | Large file storage | Essential for storing audio datasets and trained model checkpoints. Configure before committing audio files. |

## Installation

```bash
# Core PyTorch (Choose ONE based on your hardware)

# Apple Silicon (MPS)
pip3 install torch==2.10.0 torchaudio==2.10.0 torchvision

# NVIDIA GPU (CUDA 12.8 - recommended for RTX 30/40 series)
pip3 install torch==2.10.0 torchaudio==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU only (for testing)
pip3 install torch==2.10.0 torchaudio==2.10.0 torchvision

# Training framework
pip install pytorch-lightning==2.6.1

# Generative models
pip install acids-rave  # RAVE (install PyTorch first!)
pip install git+https://github.com/Stability-AI/stable-audio-tools  # Stable Audio

# Neural codecs
pip install descript-audio-codec  # DAC
pip install encodec  # EnCodec (if you need 48kHz exactly)

# Audio processing
pip install librosa>=0.11.0
pip install soundfile
pip install pedalboard>=0.9.22

# Training utilities
pip install auraloss
pip install einops>=0.8.2
pip install hydra-core

# Experiment tracking
pip install wandb  # Weights & Biases
# OR
pip install mlflow  # If you prefer MLflow

# UI
pip install gradio==6.5.1

# Development tools
pip install jupyter jupyterlab
pip install matplotlib seaborn  # Visualization
pip install tensorboard  # Alternative visualization

# System dependencies
conda install ffmpeg  # Required by RAVE and audio processing
```

## Alternatives Considered

| Category | Recommended | Alternative | When to Use Alternative |
|----------|-------------|-------------|-------------------------|
| Generative Model | RAVE v2 | Jukebox (OpenAI) | If you have massive compute (days on 256 V100s) and huge datasets (millions of songs). Not suitable for 5-500 files. |
| Generative Model | RAVE v2 / DDSP | AudioLM | If you want language model approach to audio. Requires large datasets and tokenized audio. Overkill for small datasets. |
| Neural Codec | DAC | SoundStream | Google's codec, but less accessible (no official implementation). DAC is "drop-in replacement for EnCodec" with better quality. |
| Audio I/O | soundfile | scipy.io.wavfile | Only if you're restricted to stdlib. soundfile supports more formats and is faster. |
| UI Framework | Gradio | Streamlit | If you need more complex app structure. Gradio is simpler and audio-focused. |
| Experiment Tracking | W&B | TensorBoard | If you want completely local/offline tracking. W&B has better audio support and collaboration features. |
| Config Management | Hydra | argparse + YAML | Only for very simple projects. Hydra's composition and override system is essential for complex experiments. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Magenta DDSP (TensorFlow version) | TensorFlow is declining in research community. Harder to customize. PyTorch ecosystem dominates audio ML. | DDSP concepts implemented in PyTorch (or contribute PyTorch port to community) |
| WaveNet / WaveGlow | Outdated (2016-2018). Extremely slow training and inference. Superseded by RAVE, diffusion models, and neural codecs. | RAVE for VAE approach, Stable Audio for diffusion |
| pySoX / SoxBindings | 300x slower than pedalboard. Limited effects. Bindings can be fragile. | pedalboard (modern, fast, maintained by Spotify) |
| Older PyTorch (<2.0) | Missing key features: torch.compile, better MPS support, Flash Attention. | PyTorch 2.10.0 (current stable) |
| MusicVAE (original) | Designed for symbolic music (MIDI), not audio waveforms. Small latent space (512-dim) limits expressiveness for audio. | RAVE (designed for audio from the ground up) |
| Real-time focused models for non-real-time use | RAVE has real-time variant but you want quality over speed. Disable real-time constraints in training config. | Use non-causal convolutions, larger models, prioritize quality in your fork/config |

## Stack Patterns by Use Case

**If training from scratch with 5-20 files:**
- Use DDSP (works with <13 min of audio)
- Aggressive data augmentation with pedalboard
- Simple architecture to avoid overfitting
- Multi-resolution STFT loss from auraloss

**If training with 50-500 files:**
- Use RAVE v2 or v2_small
- Beta-VAE for disentanglement (beta=4-10)
- Data augmentation with pedalboard (pitch shift, time stretch, EQ)
- Multi-resolution STFT + perceptual losses

**If fine-tuning pretrained models:**
- Use Stable Audio Tools
- Freeze encoder, train only decoder or specific layers
- Your small dataset becomes "style transfer" data
- Leverage W&B for comparing fine-tuning strategies

**For high-fidelity (48kHz/24bit+) generation:**
- EnCodec for exact 48kHz support, or upsample DAC output
- Multi-resolution STFT loss at multiple scales
- Avoid downsampling in preprocessing - work at native rate
- Use 24-bit WAV output (soundfile supports this)

**For musically meaningful controls:**
- Implement Beta-VAE with disentanglement metrics
- Create interpretable dimensions via controlled interventions
- Use DDSP components for explicit harmonic/noise control
- Build Gradio interface with labeled sliders per dimension

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| PyTorch 2.10.0 | TorchAudio 2.10.0 | Always use matching versions |
| PyTorch 2.10.0 | PyTorch Lightning 2.6.1 | Tested and verified |
| RAVE 2.3.1 | PyTorch 1.13+ | Now flexible - install PyTorch first, then RAVE |
| Stable Audio Tools | PyTorch 2.5+ | Requires Flash Attention support, introduced in 2.5 |
| Gradio 6.5.1 | PyTorch 2.10.0 | Independent packages, no conflicts |
| librosa 0.11.0 | soundfile latest | librosa uses soundfile as backend |
| pedalboard 0.9.22 | All Python 3.8+ | Native extension, framework-agnostic |

## Apple Silicon (MPS) Specific Notes

**What works:**
- PyTorch 2.10.0 has mature MPS support
- TorchAudio operations (spectrograms, resampling)
- RAVE training (tested in Jan 2026)
- Most audio processing (librosa, pedalboard, soundfile)

**What to watch:**
- Some PyTorch ops fall back to CPU (warnings will appear)
- Float precision matters - MPS can be sensitive to fp16
- For RAVE: use float32 for training on MPS (voice clone requires it)
- Test on MPS first, but have cloud GPU backup for large-scale training

**Workarounds:**
- Use `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable
- Monitor MPS memory usage (Activity Monitor > GPU History)
- For operations that fail on MPS, use `.cpu()` selectively

## Cloud GPU Recommendations

**For experimentation (RAVE v2_small):**
- 1x RTX 3080 (10GB) or better
- 16GB system RAM
- Cost: ~$0.50/hour on AWS/Lambda Labs

**For production (RAVE v2):**
- 1x RTX 3090 / 4090 (24GB) or A100 (40GB)
- 32GB system RAM
- Cost: ~$1-2/hour

**For Stable Audio fine-tuning:**
- Multi-GPU setup (2-4x A100) recommended
- DeepSpeed ZeRO Stage 2 for memory efficiency
- Cost: ~$4-8/hour for 4x A100

**Providers:**
- Lambda Labs: Good GPU availability, hourly billing
- RunPod: Competitive pricing, Spot instances
- Google Colab Pro+: Good for prototyping, A100 access
- AWS/GCP/Azure: Enterprise option, more expensive

## Small Dataset Strategy

Your core challenge is making generative audio work with 5-500 files. Here's the stack strategy:

**Data augmentation is critical:**
1. Use pedalboard for musical augmentations (EQ, compression, reverb)
2. Pitch shifting ±2 semitones without timestretching
3. Time stretching ±10% without pitch shifting
4. Random EQ curves, subtle distortion
5. Mix with environmental noise at low levels

**Architecture choices:**
1. Smaller models to prevent overfitting (RAVE v2_small over v2)
2. Strong regularization (Beta-VAE with beta=6-10)
3. Early stopping with validation set (20% of your data)
4. Multi-resolution STFT loss to learn features at multiple scales

**Transfer learning when possible:**
1. Stable Audio Tools: Fine-tune pretrained models
2. Neural codecs: Use pretrained DAC/EnCodec, train only on top
3. RAVE: If community releases pretrained models, start from those

**Validation strategy:**
1. Log audio to W&B every N epochs
2. Qualitative listening tests are as important as metrics
3. Latent space interpolations to check smoothness
4. Reconstruction quality on held-out files

## Sources

**High Confidence (Official Docs & Releases):**
- [PyTorch 2.10.0 Installation](https://pytorch.org/get-started/locally/)
- [PyTorch Lightning 2.6.1 Changelog](https://lightning.ai/docs/pytorch/stable/generated/CHANGELOG.html)
- [RAVE GitHub Repository](https://github.com/acids-ircam/RAVE)
- [Stable Audio Tools GitHub](https://github.com/Stability-AI/stable-audio-tools)
- [DAC GitHub Repository](https://github.com/descriptinc/descript-audio-codec)
- [Gradio 6.5.1 Documentation](https://www.gradio.app/docs/gradio/audio)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index)

**Medium Confidence (Recent Research & Community):**
- [Source-Disentangled Neural Audio Codec (SD-Codec)](https://arxiv.org/html/2409.11228v1) - ICASSP 2025
- [MiMo-Audio: Few-Shot Audio Learning](https://arxiv.org/abs/2512.23808) - 2025
- [Stable Audio Open Paper](https://arxiv.org/html/2407.14358v1)
- [DDSP Original Paper](https://magenta.tensorflow.org/ddsp)
- [Beta-VAE Disentanglement Research 2025](https://arxiv.org/html/2602.09277)
- [Pedalboard by Spotify](https://github.com/spotify/pedalboard)
- [auraloss GitHub](https://github.com/csteinmetz1/auraloss)

**Medium Confidence (Web Search - Multiple Sources):**
- [RAVE vs DDSP comparison](https://neurorave.github.io/neurorave/)
- [EnCodec Facebook Research](https://github.com/facebookresearch/encodec)
- [Neural Audio Codecs Overview 2025](https://www.abyssmedia.com/audioconverter/neural-audio-codecs-overview.shtml)
- [PyTorch MPS Apple Silicon Support](https://developer.apple.com/metal/pytorch/)
- [Einops 0.8.2 Release](https://github.com/arogozhnikov/einops)

**Note on Confidence Levels:**
- Core framework versions (PyTorch, Lightning, Gradio): HIGH - verified from official sources
- Generative models (RAVE, Stable Audio, DDSP): HIGH - verified from official repos
- Small dataset performance claims: MEDIUM - based on research papers and community reports
- Disentanglement techniques: MEDIUM - active research area, techniques evolving
- MPS support specifics: MEDIUM - based on community testing and GitHub issues, still maturing

---
*Stack research for: Small Dataset Generative Audio*
*Researched: 2026-02-12*
