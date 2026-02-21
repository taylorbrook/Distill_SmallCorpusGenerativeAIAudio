# Small DataSet Audio

## What This Is

A generative audio application that trains VQ-VAE models (Residual Vector Quantization) on small, personal audio datasets (5-500 files) and lets users explore discrete audio codes through a "sound DNA editor" — encoding existing audio into codebook entries, swapping/blending/editing codes between files, and decoding back. Includes an autoregressive prior for generating new code sequences. Features a Gradio web UI and CLI, multi-format export (WAV/MP3/FLAC/OGG), spatial audio (stereo/binaural), and model library management. The musical equivalent of Autolume — but for audio instead of visuals.

## Core Value

Controllable exploration — users can reliably navigate between sound worlds using discrete audio codes and generative priors, generating audio that clearly comes from their own material while discovering new territory within it.

## Current Milestone: v1.1 VQ-VAE

**Goal:** Replace the continuous VAE with a Residual Vector Quantization architecture, add an autoregressive prior for generation, and build a code-manipulation UI for exploring discrete audio representations.

**Target features:**
- RVQ-VAE encoder/decoder with stacked codebooks (coarse-to-fine)
- Codebook management: EMA updates, dead code resets, k-means init, dataset-scaled sizing
- Autoregressive prior model for generating new code sequences
- Code manipulation UI: encode audio → view/edit/swap/blend codes → decode
- Updated training pipeline for VQ-VAE (replaces continuous VAE entirely)
- Updated model persistence and library for VQ-VAE format

## Requirements

### Validated

- ✓ Train a generative model on a small personal audio dataset (5-500 files) — v1.0
- ✓ Generate non-real-time audio (textures, soundscapes, building blocks) from a trained model — v1.0
- ✓ Control generation via musically meaningful sliders: timbral (brightness, warmth, roughness), harmonic tension, temporal character, spatial/textural — v1.0
- ✓ Set generation duration per output — v1.0
- ✓ Set generation density per output — v1.0
- ✓ Export audio at 48kHz/24bit baseline with configurable sample rate and bit depth — v1.0
- ✓ Configurable mono/stereo output — v1.0
- ✓ Save and recall slider presets — v1.0
- ✓ Load and explore previously trained models — v1.0
- ✓ Train and manage multiple models, blending between them — v1.0
- ✓ GUI with sliders, audio playback, file management (Gradio-based) — v1.0
- ✓ CLI access for batch generation and scripting — v1.0
- ✓ Multi-format export (MP3, FLAC, OGG) with metadata embedding — v1.0
- ✓ Spatial audio output (stereo field, binaural) — v1.0
- ✓ Generation history with A/B comparison — v1.0
- ✓ Model library with save/load/delete/search — v1.0

### Active

- [ ] RVQ-VAE architecture replacing continuous VAE
- [ ] Autoregressive prior model for code sequence generation
- [ ] Code manipulation UI (encode, swap, blend, decode)
- [ ] Updated training loop for VQ-VAE
- [ ] Updated model persistence for VQ-VAE format

### Deferred

- Incrementally add more training audio to an existing model — deferred to future milestone
- Feed generated outputs back into training data for iterative refinement — deferred to future milestone
- Bundle HRTF SOFA file for binaural mode — deferred to future milestone

### Out of Scope

- Real-time audio generation — quality over latency; non-real-time allows deeper processing and higher fidelity
- Text/semantic prompt control — small datasets make semantic concepts unreliable; sliders are more grounded
- Full musical compositions with structure — v1 focuses on textures and building blocks
- Mobile app — desktop/browser-first
- DAW plugin (VST/AU) — standalone tool first, plugin integration later
- Cloud-hosted training service — runs locally on user's hardware
- OSC/MIDI controller mapping — deferred to future
- Multi-channel spatial audio (5.1, 7.1, ambisonic) — stereo/binaural sufficient
- Continuous latent space exploration (PCA sliders) — replaced by discrete code manipulation in v1.1
- Multi-model blending — v1.0 feature, needs redesign for VQ-VAE codes
- v1.0 model backward compatibility — clean break, VQ-VAE is a new architecture

## Context

Shipped v1.0 with 17,520 LOC Python across 186 files.
Tech stack: Python 3.13, PyTorch 2.10.0, TorchAudio, Gradio, Typer, Rich, soundfile, mutagen, sofar.
Hardware: Apple Silicon (MPS), NVIDIA (CUDA), CPU fallback.
Architecture: v1.0 used Convolutional VAE on mel spectrograms with PCA-based latent space analysis. v1.1 replaces this with RVQ-VAE (Residual Vector Quantization) for discrete code representations plus an autoregressive prior for generation.

**Inspiration:** Autolume (Metacreation Lab) — a no-code system for training StyleGAN on personal image datasets with interactive latent space exploration. This project applies the same philosophy to audio.

**Prior art and frustrations:**
- **RAVE**: Right approach (train on your own audio, explore latent space) but parameters are opaque — latent dimensions don't map to musical concepts — and the quasi-real-time focus compromises output quality
- **Suno/commercial tools**: Too polished, rely on massive datasets of popular styles, output sounds derivative. No personal voice.

**The gap this fills:** Between RAVE's "train your own model" ethos and commercial tools' polish. Nothing currently offers controllable exploration of your own sound with musically meaningful parameters.

**User profile:** Musicians, sound artists, experimental composers working across genres — ambient/textural, electronic, experimental/noise, acoustic/instrumental. The tool is genre-agnostic, learning from whatever audio it's fed.

**Output usage:** Generated audio goes into DAW production, standalone listening, and live performance contexts. Export pipeline and format flexibility matter.

**Known tech debt:**
- 16 human listening/verification tests pending (Phases 4, 5, 10)
- HRTF SOFA file not bundled for binaural mode

## Constraints

- **Audio quality**: 48kHz/24bit minimum baseline — lower is not acceptable for professional audio work
- **Hardware flexibility**: Must support Apple Silicon (MPS), NVIDIA (CUDA), and CPU fallback. Cannot be locked to one platform
- **Dataset ethics**: Only user's own recordings or ethically/legally sourced material. No scraping, no copyright infringement
- **Small data**: The architecture must produce meaningful results with as few as 5-20 audio files, not just 500+
- **Tech stack**: Python-based (PyTorch ecosystem) — non-negotiable given the ML/audio library landscape

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Non-real-time generation | Quality over latency; allows deeper processing, higher fidelity, more complex models | ✓ Good — enables chunk-based generation with crossfade for arbitrary duration |
| Parameter sliders over text prompts for v1 | Small datasets make semantic concepts unreliable; sliders map directly to latent space; musicians think in knobs | ✓ Good — PCA-based mapping gives musically meaningful control |
| Python + Gradio for v1 | Fastest path to usable creative tool; Gradio gives sliders/audio/upload natively; can upgrade UI later | ✓ Good — full 4-tab UI shipped in 5 plans |
| 48kHz/24bit baseline | Professional audio production standard; below this is not credible for serious audio work | ✓ Good — anti-aliasing filter prevents artifacts above 20kHz |
| Genre-agnostic architecture | User works across ambient, electronic, experimental, acoustic — model must learn from any audio | ✓ Good — VAE learns from any input audio |
| Novel approach emphasis | User wants research-grade innovation, not just wrapping existing models in a GUI | ✓ Good — PCA-based latent space analysis is core innovation over RAVE |
| soundfile over TorchCodec for audio I/O | TorchCodec/FFmpeg broken on macOS; soundfile is reliable and lightweight | ✓ Good — zero issues across all phases |
| JSON index over SQLite for model catalog | Sufficient for <1000 models, human-readable, zero-dependency | ✓ Good — simple and reliable |
| Atomic write pattern for all JSON indexes | Crash-safe writes with temp file + os.replace + .bak backup | ✓ Good — prevents data corruption |
| PCA with 2% variance threshold | Conservative filter removes noise dimensions while preserving meaningful ones | ✓ Good — gives 4-8 meaningful sliders per model |
| sofar library for HRTF/binaural | Standard SOFA format for head-related transfer functions | ⚠️ Revisit — requires user to download SOFA file |
| Replace continuous VAE with RVQ-VAE | Discrete codes produce sharper reconstructions; eliminates KL-balancing complexity; enables code-level manipulation | — Pending |
| Autoregressive prior over codes | Needed for generation from VQ-VAE (can't sample from N(0,1) like continuous VAE) | — Pending |
| Clean break from v1.0 models | VQ-VAE is fundamentally different architecture; backward compat adds complexity for no benefit | — Pending |
| lucidrains/vector-quantize-pytorch | Drop-in VQ/RVQ layers for PyTorch; well-maintained, widely used | — Pending |

---
*Last updated: 2026-02-21 after v1.1 milestone start*
