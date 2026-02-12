# Project Research Summary

**Project:** Small Dataset Generative Audio
**Domain:** High-fidelity generative audio synthesis from small personal datasets
**Researched:** 2026-02-12
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project aims to build a generative audio tool that trains on extremely small datasets (5-500 audio files) and produces high-fidelity output (48kHz/24-bit) with musically meaningful parameter controls. Research shows this fills a gap between consumer text-to-music tools (Suno/Udio) and technical neural synthesis frameworks (RAVE). The recommended approach is a **VAE-based architecture with aggressive data augmentation, disentangled latent space, and optional DDSP components for interpretability**.

The core technical challenge is preventing overfitting with minimal data while maintaining quality and control. Successful implementations use: (1) PyTorch 2.10.0 with RAVE v2 or DDSP as the generative model backbone, (2) 10-100x data augmentation via pedalboard and audiomentations, (3) beta-VAE with disentanglement regularization for interpretable controls, and (4) multi-resolution spectral losses combined with adversarial training for perceptual quality. Training on 50-500 files typically requires 8GB+ GPU, hours-to-days of training time, and careful hyperparameter tuning.

Key risks include overfitting (requires validation-first design and aggressive augmentation), latent space uninterpretability (requires disentanglement from the start), and audio artifacts at 48kHz (requires anti-aliased architecture or PQMF multi-band decomposition). The product differentiates through musically meaningful parameter controls (timbre, harmony, temporal, spatial) derived from PCA-based feature extraction, not opaque latent dimensions. Success depends on building the training pipeline with small-dataset considerations from day one, not bolting them on later.

## Key Findings

### Recommended Stack

The stack centers on PyTorch 2.10.0 as the deep learning framework (mature MPS support for Apple Silicon, excellent CUDA support, research-grade flexibility) with PyTorch Lightning 2.6.1 for training orchestration. For the generative model, research recommends **starting with RAVE v2_small** for rapid prototyping on 8GB GPU, optionally layering DDSP components for interpretable synthesis. Stable Audio Tools provides an alternative for fine-tuning pretrained models if training from scratch fails.

**Core technologies:**
- **PyTorch 2.10.0 + TorchAudio 2.10.0**: Deep learning framework with native audio support — industry standard for research-grade audio ML with MPS/CUDA flexibility
- **RAVE v2_small/v2**: VAE-based real-time audio synthesis — proven for small datasets (2-3 hours minimum), fast inference, but opaque latent controls requiring disentanglement
- **DDSP (optional)**: Differentiable DSP synthesis — works with <13 minutes of audio, provides inherently interpretable parameters (harmonic/noise components), trade-off is limited to simpler timbres
- **Pedalboard 0.9.22+**: Audio effects and augmentation — 300x faster than alternatives, critical for expanding small datasets via pitch shift, time stretch, EQ variations
- **auraloss 0.2.0+**: Audio-specific loss functions — multi-resolution STFT loss essential for perceptual quality, avoids blurry audio from L1/L2 losses alone
- **Gradio 6.5.1**: Interactive UI — built-in audio components with waveform display, perfect for rapid prototyping of parameter controls and real-time manipulation
- **Weights & Biases**: Experiment tracking — rich media support for audio/spectrograms, real-time metric logging, essential for monitoring long training runs

**Version-critical notes:** RAVE 2.3.1 now works with PyTorch 2.10.0 (install PyTorch first). Stable Audio Tools requires PyTorch 2.5+ for Flash Attention. DAC supports 44.1kHz (use EnCodec for exact 48kHz if needed).

### Expected Features

Research into AI music generation tools, VST synthesis patterns, and audio ML workflows reveals clear feature expectations.

**Must have (table stakes):**
- **Audio file import with drag & drop** — standard across all audio tools, multi-file batch import critical for dataset building
- **Model training with progress monitoring** — long processes (hours) require visibility: loss curves, epoch counters, sample previews, time estimates
- **Real-time parameter controls during generation** — creative tools demand immediate feedback, responsive sliders while audio generates (synth-style interaction)
- **Audio preview/playback** — cannot evaluate without listening, requires waveform display and transport controls
- **Preset save/recall** — musicians expect this from VSTs/DAWs, enables exploration without losing good results
- **Audio export (WAV, 48kHz/24-bit)** — getting audio into DAW is table stakes, uncompressed format critical for professional use
- **Multiple model management** — users will train on different datasets (voice, drums, textures), need to switch between models with metadata
- **Dataset visualization** — confidence-building before hours-long training: file count, duration, sample rate consistency checks

**Should have (competitive differentiators):**
- **Musically meaningful parameter controls** — core differentiator vs RAVE's opaque z1-z16; map latent space to parameters musicians understand (timbre, harmony, temporal envelope, spatial width)
- **Small dataset optimization (5-500 files)** — core competitive moat vs Suno/Udio; specialization in personal datasets through data augmentation, transfer learning, architecture choices
- **PCA-based feature extraction** — automatic discovery of interpretable control dimensions from trained model, bridges gap between latent space and musical parameters
- **Incremental training** — add files to existing model without full retrain, supports iterative dataset curation, addresses catastrophic forgetting
- **Output-to-training feedback loop** — evolutionary sound design: generate → select favorites → retrain on them, creates refinement spiral
- **OSC/MIDI parameter control** — live performance use case, map hardware controllers or receive from DAW/Ableton
- **Generation history with visual diff** — see/hear what changed between parameter adjustments, supports exploration without getting lost

**Defer (v2+):**
- **Multi-model layering/blending** — generate from multiple models simultaneously, blend outputs (60% model A + 40% model B)
- **Spatial audio output** — beyond stereo: binaural, 5.1/7.1, ambisonic for soundscape/texture generation
- **Granular density/size controls** — texture generation sweet spot, grain count/size/pitch controls familiar from granular synths

**Anti-features (explicitly avoid):**
- **Text-to-music generation** — Suno/Udio's domain, not aligned with personal small dataset value prop
- **Built-in DAW/multi-track editor** — scope creep, users have DAWs, focus on being best-in-class generator that integrates
- **Cloud model training** — privacy concerns with personal audio, latency, infrastructure costs
- **Real-time (<10ms) generation** — RAVE achieves this but compromises quality, accept 100ms-1s for 48kHz/24-bit quality
- **Vocal/lyric generation** — legal/ethical concerns, outside "textures/soundscapes/building blocks" scope

### Architecture Approach

The standard architecture is a **multi-stage VAE pipeline with optional GAN refinement and DDSP components**. The system follows: Data Pipeline (ingest, normalize, chunk, augment) → Feature Extraction (mel-spectrograms, optional f0/loudness) → Encoder (compress to latent) → Latent Space (disentangled bottleneck) → Decoder (reconstruct waveform or DSP parameters) → Post-Processing (normalize, export). Training uses two-stage approach: first VAE with reconstruction + KL loss, then add GAN discriminator for perceptual refinement.

**Major components:**
1. **Data Pipeline** — Handles 5-500 file ingestion with torchaudio/soundfile for I/O, aggressive augmentation via pedalboard (pitch shift ±2 semitones, time stretch 0.9-1.1x, noise injection), SpecAugment for frequency/time masking; responsible for 10-100x dataset expansion critical for small-data scenarios
2. **VAE Core (Encoder + Latent + Decoder)** — Encoder compresses audio to 16-128 dimensional latent space, latent space uses beta-VAE regularization (beta=4-10) for disentanglement, decoder reconstructs via neural layers or DDSP synthesis modules; follows RAVE v2 pattern with optional PQMF multi-band decomposition for 48kHz efficiency
3. **Training Engine** — PyTorch Lightning orchestration with multi-resolution STFT loss (auraloss), KL divergence with beta-scheduling, optional adversarial loss from multi-scale discriminators, checkpointing every N epochs with validation split; monitors for overfitting and posterior collapse
4. **Latent Space Mapper** — Post-training PCA/sparse autoencoder analysis to identify interpretable dimensions, maps UI sliders (timbre, harmony, temporal, spatial) to latent coordinates, enables preset save/load and interpolation; the innovation layer distinguishing from raw RAVE
5. **Generation Engine** — Non-realtime inference with parameter controls, latent code generation from sliders, decoder forward pass (100ms-1s latency acceptable), post-processing for normalization/fades; integrates with Gradio UI for interactive exploration
6. **Export Pipeline** — Sample rate conversion if needed (high-quality resampling), bit depth adjustment with dithering, WAV/FLAC output with metadata; ensures DAW-ready audio

**Critical architectural patterns:**
- **PQMF multi-band decomposition** — Decomposes 48kHz into 16 frequency bands, reduces sequence length 16x, enables realtime synthesis without quality loss (RAVE approach)
- **Two-stage training (VAE → VAE+GAN)** — Train VAE first for reconstruction, add adversarial loss for perceptual quality; prevents GAN instability early in training
- **Aggressive augmentation for small datasets** — Time/pitch/noise transforms + SpecAugment applied probabilistically during data loading; critical pattern for <500 files
- **Disentangled latent space** — Beta-VAE with beta > 1, semi-supervised constraints, or orthogonalization to map latent dims to interpretable properties; without this, sliders are unusable

### Critical Pitfalls

Research into small-dataset audio ML, VAE/GAN training, and neural audio synthesis reveals recurring failure modes.

1. **Overfitting without detection** — With 5-50 files, models memorize training data instead of learning generalizable features; becomes lookup table that can only reproduce training samples. **Avoid by:** validation set from day one (even with 10 files, hold out 2), track perceptual diversity metrics (not just reconstruction loss), use dropout 0.2-0.3 + weight decay + VAE beta-scheduling, implement 10-100x data augmentation, apply DPO for output diversity. Warning signs: validation loss diverges after 10-20 epochs, generated samples sound identical to training, latent interpolations produce abrupt transitions, KL divergence approaches zero.

2. **Audio preprocessing pipeline incompatibilities** — Loading/preprocessing with librosa during training but torchaudio during inference causes distribution shift and quality degradation; torchaudio MelScale uses non-default librosa options creating different filter banks. **Avoid by:** single library consistency (choose librosa OR torchaudio for entire pipeline), explicit parameter documentation (save n_fft, hop_length, n_mels, window, norm, htk in config), preprocessing module with frozen parameters, validation test asserting train/inference outputs are identical. Warning signs: model works in training but produces noisy output in inference, switching CPU/GPU changes quality, spectrograms look visually different.

3. **Aliasing artifacts at 48kHz** — Unconstrained nonlinear activations generate infinite harmonics exceeding Nyquist frequency (24kHz), causing "folded-back" aliasing; ConvTranspose upsampling creates "mirrored" high-frequency content. Audio sounds metallic with unnatural shimmer. **Avoid by:** anti-aliased activations with oversampling, replace ConvTranspose with interpolation + Conv1d, PQMF multi-band decomposition (RAVE approach), explicit low-pass filtering before upsampling, multi-resolution STFT loss to penalize high-frequency artifacts. Warning signs: high-frequency "sparkle" in output, metallic ringing, spectrograms show mirrored patterns, constant-frequency tones appear.

4. **Uninterpretable latent space controls** — Standard VAE doesn't enforce semantic meaning on latent dimensions; sliders produce unpredictable or correlated changes, users can't explore sound space intentionally. **Avoid by:** latent space regularization forcing dims to map to target attributes (pitch, spectral centroid, RMS), beta-VAE with beta > 1 for dimension independence, sparse autoencoders on top of learned latents, post-training systematic analysis varying each dim to measure acoustic property changes. Warning signs: one slider changes multiple unrelated characteristics, latent interpolations produce incoherent transitions, extreme values produce noise/silence.

5. **Posterior collapse in VAE training** — Latent space becomes uninformative, KL divergence approaches zero, decoder ignores latent codes and generates same output regardless of input. **Avoid by:** beta-VAE scheduling (start beta=0, gradually increase to 1.0), KL floor ensuring minimum divergence (e.g., KL >= 0.5), "free bits" allowing first N nats of KL without penalty, monitor KL per dimension, limit decoder capacity to force latent dependence. Warning signs: KL < 0.1, sampling from prior produces coherent audio (decoder ignores latents), different latent codes produce identical outputs.

6. **GAN training instability** — Discriminator becomes too strong (generator gradients vanish) or generator mode collapses (same outputs repeatedly); training oscillates or diverges. **Avoid by:** Wasserstein loss with gradient penalty (WGAN-GP), spectral normalization on discriminator, balanced update schedule (update discriminator less frequently), progressive training (start low-resolution, increase to 48kHz), multi-scale discriminators, monitor gradient norms. Warning signs: discriminator accuracy > 95%, generator loss increases continuously, mode collapse, loss oscillations amplitude > 2x.

## Implications for Roadmap

Based on combined research, the roadmap should follow a **risk-first progression**: establish small-dataset training pipeline with validation/augmentation before building differentiators, validate architecture quality before adding interpretability layer, polish UX only after core value prop proven.

### Suggested Phase Structure

**Phase 1: Core Training Pipeline (Foundation)**
- **Rationale:** Cannot build anything without ability to train on small datasets; overfitting is the primary risk, must address immediately with validation split and augmentation
- **Delivers:** Data pipeline (ingest, normalize, chunk), augmentation (pedalboard transforms, SpecAugment), basic VAE training loop, checkpointing, validation framework
- **Features addressed:** Audio file import, dataset visualization, model training, training progress monitoring
- **Pitfalls avoided:** Overfitting without detection (#1), preprocessing incompatibilities (#2), posterior collapse (#5)
- **Stack used:** PyTorch 2.10.0, TorchAudio, pedalboard, auraloss (multi-resolution STFT), soundfile
- **Architecture components:** Data Pipeline, Feature Extraction, Encoder/Decoder skeleton, Training Engine
- **Validation criteria:** Model trains without posterior collapse (KL > 0.5), validation loss tracks within 20% of training loss, spectrograms visually consistent between train/inference
- **Research flag:** Standard patterns well-documented in RAVE/PyTorch Lightning docs — skip phase-specific research

**Phase 2: Quality & Audio Fidelity (Prove Quality)**
- **Rationale:** After basic training works, validate output quality at 48kHz before investing in interpretability; aliasing and perceptual quality are architecture-level decisions hard to fix later
- **Delivers:** Anti-aliased architecture (PQMF or interpolation-based upsampling), GAN discriminator for perceptual refinement, perceptual evaluation metrics (ViSQOL, DPAM), multi-resolution STFT loss
- **Features addressed:** Audio generation, audio preview, audio export (48kHz/24-bit WAV)
- **Pitfalls avoided:** Aliasing artifacts at 48kHz (#3), evaluating only with reconstruction metrics (#9), ignoring phase information (#10)
- **Stack used:** RAVE PQMF decomposition or custom anti-aliased layers, GAN discriminator (spectral norm), ViSQOL/DPAM metrics
- **Architecture components:** VAE Core (full implementation), GAN Discriminator, Export Pipeline
- **Validation criteria:** Spectral analysis shows no aliasing above 20kHz, perceptual metrics correlate with listening tests, generated audio passes blind quality test vs training samples
- **Research flag:** May need phase research for anti-aliasing techniques if custom architecture required (PQMF is RAVE standard)

**Phase 3: Latent Space Exploration (Core Innovation)**
- **Rationale:** Once quality proven, tackle the differentiator — musically meaningful controls; this is high-risk innovation area requiring experimentation
- **Delivers:** Beta-VAE disentanglement, post-training PCA/sparse autoencoder analysis, latent space mapper, interpretable slider labels (timbre, harmony, temporal, spatial), preset save/load
- **Features addressed:** Parameter controls (musically meaningful), preset save/recall
- **Pitfalls avoided:** Uninterpretable latent space controls (#4)
- **Stack used:** Beta-VAE implementation, sparse autoencoders, librosa for acoustic feature extraction, custom latent analysis tools
- **Architecture components:** Latent Space (with disentanglement), Latent Space Mapper
- **Validation criteria:** Each slider maps to single perceptual attribute in listening tests, latent traversals produce smooth transitions, extreme values produce valid (not noisy) audio
- **Research flag:** NEEDS PHASE RESEARCH — disentanglement techniques are research-grade, limited production examples; study beta-VAE papers, sparse autoencoder implementations, audio-specific disentanglement

**Phase 4: UI & Model Management (Usability)**
- **Rationale:** With working model + interpretable controls, make it usable; Gradio interface validates workflow before complex features
- **Delivers:** Gradio interface (file upload, sliders, playback, export), model library/browser, load/unload models, metadata display, generation history
- **Features addressed:** Multiple model management, generation history, parameter controls (UI layer)
- **Pitfalls avoided:** MPS vs CUDA differences (#7), UX pitfalls (no progress indication, unlabeled sliders, no undo)
- **Stack used:** Gradio 6.5.1, device abstraction (MPS/CUDA/CPU), model checkpointing
- **Architecture components:** UI Layer, Generation Engine, Model Manager
- **Validation criteria:** UI works on both MPS and CUDA, memory doesn't leak over 10+ generations, sliders labeled with perceptual meaning, history allows revisiting previous results
- **Research flag:** Standard patterns — Gradio official docs sufficient, skip phase research

**Phase 5: Advanced Features (Competitive Edge)**
- **Rationale:** Core validated, now add differentiators; incremental training is high-complexity, defer until users explicitly need it
- **Delivers:** Incremental training with catastrophic forgetting mitigation, output-to-training feedback loop, OSC/MIDI control, batch generation CLI
- **Features addressed:** Incremental training, output-to-training loop, OSC/MIDI control
- **Pitfalls avoided:** Catastrophic forgetting (#13), training without compute budgeting (#12)
- **Stack used:** Elastic Weight Consolidation or replay buffers, OSC/MIDI libraries, argparse CLI
- **Architecture components:** Training Engine (incremental mode), CLI Interface
- **Validation criteria:** Adding new data maintains <5% loss increase on old validation set, OSC/MIDI controls respond in <100ms, batch generation completes overnight
- **Research flag:** NEEDS PHASE RESEARCH — incremental learning for audio is active research area, study continual learning papers, EWC implementations

### Phase Ordering Rationale

**Why this order:**
1. **Foundation first (Phase 1):** Small-dataset training is the hardest problem; no point building UI for a model that overfits immediately. Validation framework and augmentation must be architectural, not bolted on.
2. **Quality before features (Phase 2):** 48kHz aliasing and GAN instability are architecture-level issues. Fixing them later means re-training from scratch. Prove audio quality before investing in interpretability.
3. **Innovation when stable (Phase 3):** Latent space disentanglement is the core differentiator but also high-risk experimentation. Only attempt after training pipeline and quality are proven.
4. **UX when validated (Phase 4):** Gradio UI is low-risk but useless without good model. Build UI to validate workflow, not as first deliverable.
5. **Advanced features last (Phase 5):** Incremental training is complex and users may not need it (can retrain from scratch early on). Add when pain point emerges.

**Dependency-driven grouping:**
- Phases 1-2 are tightly coupled: training pipeline → quality validation → export. Must be sequential.
- Phase 3 depends on Phase 2 output: need trained model to analyze latent space.
- Phase 4 depends on Phase 3: UI controls need interpretable latent mapping.
- Phase 5 is largely independent: incremental training builds on Phase 1, OSC/MIDI builds on Phase 4, but both can be deferred.

**Risk mitigation through ordering:**
- Front-loads highest risks (overfitting, quality) when changes are cheap.
- Defers low-probability risks (catastrophic forgetting) until value proven.
- Enables early validation: Phase 2 ends with working audio generation, proving core concept before investing in advanced features.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 3 (Latent Space Exploration):** Disentanglement techniques are research-grade with limited production examples. Need to study beta-VAE implementation details, sparse autoencoder architectures, audio-specific disentanglement metrics (MIG, SAP, DCI). Community examples (NSynth, Autolume) provide patterns but not drop-in solutions.
- **Phase 5 (Incremental Training):** Catastrophic forgetting mitigation is active research area. Need to study Elastic Weight Consolidation (EWC), replay buffers, continual learning for audio (recent papers from 2025). No established pattern for <500 file scenarios.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Core Training Pipeline):** Well-documented in PyTorch Lightning docs, RAVE repository, auraloss examples. Data augmentation patterns established (pedalboard, audiomentations). Validation split and checkpointing are standard ML practices.
- **Phase 2 (Quality & Audio Fidelity):** RAVE PQMF decomposition is reference implementation. GAN training with spectral normalization and WGAN-GP is well-documented (PyTorch examples). Perceptual metrics (ViSQOL, PESQ) have official implementations.
- **Phase 4 (UI & Model Management):** Gradio official docs cover audio components, file upload, sliders. Model checkpointing is standard PyTorch. MPS/CUDA device handling documented in PyTorch 2.x migration guides.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core technologies (PyTorch 2.10.0, RAVE, pedalboard, Gradio) verified from official repos and releases. Version compatibility tested by community. DDSP and neural codecs well-documented. |
| Features | MEDIUM | Table stakes features verified across multiple AI music tools and VST synthesis patterns. Differentiators (PCA feature extraction, incremental training) inferred from research papers and Autolume case study, not production-validated. |
| Architecture | MEDIUM-HIGH | Standard VAE pipeline and RAVE patterns documented in official implementations. PQMF decomposition, two-stage training, disentanglement techniques have research papers and code examples. Build order prioritization based on common patterns, not empirically validated for this specific project. |
| Pitfalls | MEDIUM-HIGH | Critical pitfalls (overfitting, aliasing, posterior collapse, GAN instability) well-documented in research literature and GitHub issues. MPS-specific issues verified in PyTorch issue tracker. Recovery strategies based on community reports and best practices. |

**Overall confidence:** MEDIUM-HIGH

Research is grounded in official documentation for core technologies and well-studied patterns for small-dataset audio ML. Uncertainty exists around novel aspects (musically meaningful controls, incremental training with 5-500 files) that lack production examples. Mitigation: validate assumptions early in Phase 3 when tackling interpretability, defer high-uncertainty features (incremental training) to Phase 5 when they can be cut if unsuccessful.

### Gaps to Address

**Gaps identified during research:**

1. **Musically meaningful parameter mapping** — Research shows beta-VAE and sparse autoencoders can find interpretable dimensions, but no clear methodology for mapping them to specific musical attributes (timbre vs harmony vs temporal). Plan to address: Implement multiple disentanglement techniques in Phase 3, use qualitative latent traversals + librosa feature analysis to empirically determine which dims control which properties, accept some trial-and-error.

2. **5-50 file extreme small-data regime** — Most research shows "small dataset" as 1000-10000 samples; 5-50 files is unusually small even for transfer learning. DDSP paper claims <13 minutes of audio, but that's monophonic instruments. Plan to address: Start Phase 1 with 50-500 file assumption, add transfer learning as fallback if training fails, consider DDSP baseline for comparison, may need to raise minimum dataset size after validation.

3. **Incremental training without catastrophic forgetting** — Research shows EWC and replay buffers work for larger datasets, but no validation for <500 file scenarios where new data can be significant fraction of total. Plan to address: Defer to Phase 5, implement simplest approach first (retrain with all data old+new), add EWC only if users report pain, may cut feature entirely if retrain-from-scratch is fast enough.

4. **MPS (Apple Silicon) production readiness** — PyTorch 2.10.0 has "mature" MPS support but community reports memory leaks and numerical differences vs CUDA. Plan to address: Test early on MPS hardware in Phase 1, implement explicit memory management and .contiguous() calls, document batch size limitations, provide cloud GPU training instructions as backup.

5. **48kHz quality at small dataset sizes** — No clear examples of 48kHz/24-bit generative audio trained on <100 files. RAVE examples are mostly 16-24kHz or use hours of data. Plan to address: Start Phase 2 with 24kHz validation, progressively scale to 48kHz, PQMF decomposition may be essential (not optional), prepare to compromise on either sample rate or dataset size if quality suffers.

## Sources

### Primary (HIGH confidence)

**Official Documentation & Repositories:**
- [PyTorch 2.10.0 Installation](https://pytorch.org/get-started/locally/) — Framework versions, hardware compatibility
- [PyTorch Lightning 2.6.1 Changelog](https://lightning.ai/docs/pytorch/stable/generated/CHANGELOG.html) — Training orchestration features
- [RAVE GitHub Repository](https://github.com/acids-ircam/RAVE) — Architecture details, PQMF decomposition, training configs
- [Stable Audio Tools GitHub](https://github.com/Stability-AI/stable-audio-tools) — Latent diffusion framework, transfer learning patterns
- [DAC GitHub Repository](https://github.com/descriptinc/descript-audio-codec) — Neural codec implementation
- [Gradio 6.5.1 Documentation](https://www.gradio.app/docs/gradio/audio) — UI components, audio handling
- [Pedalboard GitHub](https://github.com/spotify/pedalboard) — Audio augmentation performance claims
- [auraloss GitHub](https://github.com/csteinmetz1/auraloss) — Loss function implementations

**Research Papers (Verified):**
- [RAVE: A variational autoencoder for fast and high-quality neural audio synthesis (arXiv 2111.05011)](https://arxiv.org/abs/2111.05011) — Core architecture, training methodology
- [DDSP: Differentiable Digital Signal Processing (arXiv 2001.04643)](https://arxiv.org/abs/2001.04643) — Interpretable synthesis approach
- [Source-Disentangled Neural Audio Codec (arXiv 2409.11228)](https://arxiv.org/html/2409.11228v1) — Multi-codebook disentanglement, ICASSP 2025
- [Stable Audio Open Paper (arXiv 2407.14358)](https://arxiv.org/html/2407.14358v1) — Latent diffusion for audio

### Secondary (MEDIUM confidence)

**Small Dataset & Overfitting:**
- [Synthio: Augmenting Small-Scale Audio Classification (arXiv 2410.02056)](https://arxiv.org/html/2410.02056v2) — DPO for small-dataset audio, preference optimization
- [A survey of deep learning audio generation methods (arXiv 2406.00146)](https://arxiv.org/pdf/2406.00146) — Current landscape, overfitting challenges
- [Make the Most of Limited Datasets Using Audio Data Augmentation](https://www.edgeimpulse.com/blog/make-the-most-of-limited-datasets-using-audio-data-augmentation/) — Augmentation strategies

**Audio Quality & Artifacts:**
- [Aliasing-Free Neural Audio Synthesis (arXiv 2512.20211)](https://arxiv.org/html/2512.20211) — Anti-aliasing techniques for 48kHz
- [Upsampling Artifacts in Neural Audio Synthesis](https://www.researchgate.net/publication/352171371_Upsampling_Artifacts_in_Neural_Audio_Synthesis) — ConvTranspose problems
- [Objective Measures of Perceptual Audio Quality Reviewed (arXiv 2110.11438)](https://arxiv.org/pdf/2110.11438) — ViSQOL, PESQ, DPAM metrics

**Latent Space & Disentanglement:**
- [Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders (arXiv 2510.23802)](https://arxiv.org/html/2510.23802) — SAE for interpretability
- [Latent Space Regularization for Explicit Control of Musical Attributes](https://musicinformatics.gatech.edu/wp-content_nondefault/uploads/2019/06/Pati-and-Lerch-Latent-Space-Regularization-for-Explicit-Control-o.pdf) — Beta-VAE for music
- [Is Disentanglement Enough? (arXiv 2108.01450)](https://arxiv.org/abs/2108.01450) — Controllable music generation

**VAE/GAN Training Stability:**
- [Common GAN Problems - Google ML Guide](https://developers.google.com/machine-learning/gan/problems) — Mode collapse, vanishing gradients
- [Common VAE Training Difficulties](https://apxml.com/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive/vae-training-difficulties) — Posterior collapse prevention
- [Preventing Posterior Collapse with delta-VAEs (OpenReview)](https://openreview.net/forum?id=BJe0Gn0cY7) — KL floor technique

**Hardware & Implementation:**
- [PyTorch MPS Apple Silicon Support](https://developer.apple.com/metal/pytorch/) — MPS capabilities
- [MPS Memory Leak - PyTorch Issue #154329](https://github.com/pytorch/pytorch/issues/154329) — Known MPS issues
- [Comparing Librosa, Soundfile and Torchaudio](https://nasseredd.github.io/blog/speech-and-language-processing/comparing-audio-libraries) — Preprocessing compatibility
- [MelSpectrogram inconsistency with librosa - PyTorch Issue #1058](https://github.com/pytorch/audio/issues/1058) — Parameter alignment issues

**Incremental Learning:**
- [Online incremental learning for audio classification (arXiv 2508.20732)](https://arxiv.org/html/2508.20732) — Continual learning patterns
- [Continual Learning and Catastrophic Forgetting (arXiv 2403.05175)](https://arxiv.org/html/2403.05175v1) — EWC and mitigation strategies

**Feature Landscape:**
- [Top-13 AI Tools for Audio Creation & Editing in 2026](https://dataforest.ai/blog/best-ai-tools-for-audio-editing) — Current tool landscape
- [Best AI Music Generators 2026](https://wavespeed.ai/blog/posts/best-ai-music-generators-2026/) — Competitive analysis
- [Autolume 2.0: GAN-based No-Coding Small Data](https://creativity-ai.github.io/assets/papers/49.pdf) — PCA feature extraction pattern
- [NSynth Instrument Interface](https://magenta.tensorflow.org/nsynth-instrument) — Latent space exploration UX

### Tertiary (LOW confidence, needs validation)

**Architecture Patterns:**
- General audio synthesis architecture patterns — Inferred from multiple sources, not explicitly documented in single authoritative reference
- Component boundary recommendations — Based on common patterns across projects, not standardized
- Build order prioritization — Derived from experience patterns, not empirically validated for this specific domain

**UX & Workflow:**
- Creative version control patterns — Limited research on generative audio history/undo systems
- Preset management expectations — Inferred from VST/DAW patterns, not validated for generative audio context
- Generation history with visual diff — Novel feature with no established implementation examples

---
*Research completed: 2026-02-12*
*Ready for roadmap: yes*
