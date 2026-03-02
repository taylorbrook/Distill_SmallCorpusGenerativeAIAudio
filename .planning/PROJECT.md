# Small DataSet Audio

## What This Is

A generative audio application that trains VAE models on small, personal audio datasets (5-500 files) and lets users explore the learned sound space through musically meaningful parameter controls mapped via PCA-based latent space analysis. Uses neural vocoder reconstruction (BigVGAN-v2 universal or per-model HiFi-GAN V2) for high-fidelity mel-to-waveform synthesis. Features a Gradio web UI and CLI, multi-format export (WAV/MP3/FLAC/OGG), spatial audio (stereo/binaural), multi-model blending, presets, generation history with A/B comparison, and model library management. The musical equivalent of Autolume — but for audio instead of visuals.

## Core Value

Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters, generating audio that clearly comes from their own material while discovering new territory within it.

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
- ✓ BigVGAN-v2 universal neural vocoder as default reconstruction — v1.1
- ✓ Optional per-model HiFi-GAN V2 training for maximum fidelity — v1.1
- ✓ Griffin-Lim fully removed, neural vocoder is the only reconstruction path — v1.1
- ✓ UI vocoder selection and per-model vocoder training controls — v1.1
- ✓ CLI vocoder selection (--vocoder flag) and train-vocoder command — v1.1
- ✓ Model format v2 (.distillgan) with vocoder state bundling — v1.1

### Active

- [ ] Incrementally add more training audio to an existing model
- [ ] Feed generated outputs back into training data for iterative refinement
- [ ] Bundle HRTF SOFA file for binaural mode (currently requires user download)

### Out of Scope

- Real-time audio generation — quality over latency; non-real-time allows deeper processing and higher fidelity
- Text/semantic prompt control — small datasets make semantic concepts unreliable; sliders are more grounded
- Full musical compositions with structure — focuses on textures and building blocks
- Mobile app — desktop/browser-first
- DAW plugin (VST/AU) — standalone tool first, plugin integration later
- Cloud-hosted training service — runs locally on user's hardware
- OSC/MIDI controller mapping — deferred to v2
- Multi-channel spatial audio (5.1, 7.1, ambisonic) — stereo/binaural sufficient
- Griffin-Lim fallback — fully removed in v1.1; neural vocoder is the only path
- BigVGAN fine-tuning per model — 122M params causes OOM on consumer GPUs; use HiFi-GAN V2 instead
- CUDA kernel optimization for BigVGAN — breaks MPS/CPU compatibility

## Context

Shipped v1.1 with 21,762 LOC Python.
Tech stack: Python 3.13, PyTorch 2.10.0, TorchAudio, Gradio, Typer, Rich, soundfile, mutagen, sofar, huggingface_hub, librosa.
Hardware: Apple Silicon (MPS), NVIDIA (CUDA), CPU fallback.
Architecture: Convolutional VAE operating on mel spectrograms with PCA-based latent space analysis for musically meaningful parameter mapping. Neural vocoder reconstruction via BigVGAN-v2 (122M params, universal) or per-model HiFi-GAN V2 (0.92M params, trained on user's audio).

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
- MPS compatibility for BigVGAN Snake activations unverified
- HiFi-GAN V2 training convergence on 5-50 file datasets unvalidated with real user data
- BlendEngine creates fresh BigVGANVocoder (~489MB) per blend_generate() call — should cache

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
| BigVGAN-v2 as universal vocoder | 122M params, trained on diverse audio, works on any mel input without per-model training | ✓ Good — dramatic quality improvement over Griffin-Lim for all models |
| HiFi-GAN V2 for per-model training | 0.92M params, trainable on small datasets, learns user's specific audio distribution | ✓ Good — eliminates BigVGAN OOD and mel conversion artifacts |
| Vendor BigVGAN source (not pip) | Not on PyPI, MIT license, ~50KB source; git submodule adds complexity | ✓ Good — VENDOR_PIN.txt with commit hash for version pinning |
| Griffin-Lim full removal (no fallback) | Neural vocoder proven reliable across all generation paths; fallback adds complexity and testing burden | ✓ Good — cleaner codebase, single reconstruction path |
| Direct mel-domain filterbank transfer | Replaces Griffin-Lim waveform round-trip for mel conversion; Tikhonov regularization for stability | ✓ Good — eliminates Griffin-Lim distortion artifacts |
| 44.1kHz internal sample rate (BigVGAN native) | BigVGAN outputs at 44.1kHz; Kaiser resampler for 48kHz export | ✓ Good — no pitch shift or timing errors |
| Deferred pipeline creation | Pipeline created at generate time with resolved vocoder, not at model load | ✓ Good — enables lazy BigVGAN download and per-model vocoder selection |

---
*Last updated: 2026-03-01 after v1.1 milestone*
