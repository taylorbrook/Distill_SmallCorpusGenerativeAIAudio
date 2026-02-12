# Small DataSet Audio

## What This Is

A generative audio application that trains models on small, personal audio datasets (5–500 files) and lets users explore the learned sound space through musically meaningful parameter controls. The musical equivalent of Autolume — but for audio instead of visuals. It generates high-fidelity textures, soundscapes, and building blocks (not real-time) that reflect the character of the user's own recordings.

## Core Value

Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters, generating audio that clearly comes from their own material while discovering new territory within it.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Train a generative model on a small personal audio dataset (5–500 files)
- [ ] Generate non-real-time audio (textures, soundscapes, building blocks) from a trained model
- [ ] Control generation via musically meaningful sliders: timbral (brightness, warmth, roughness), harmonic tension (consonance ↔ dissonance), temporal character (rhythmic ↔ ambient, pulse strength), spatial/textural (sparse ↔ dense, dry ↔ reverberant)
- [ ] Set generation duration per output
- [ ] Set generation density per output
- [ ] Export audio at 48kHz/24bit baseline with configurable sample rate and bit depth (up to 96kHz/32bit)
- [ ] Configurable mono/stereo output
- [ ] Save and recall slider presets
- [ ] Load and explore previously trained models
- [ ] Incrementally add more training audio to an existing model
- [ ] Feed generated outputs back into training data for iterative refinement
- [ ] Train and manage multiple models, potentially blending between them
- [ ] GUI with sliders, audio playback, file management (Gradio-based for v1)
- [ ] CLI access for batch generation and scripting

### Out of Scope

- Real-time audio generation — quality over latency; non-real-time allows deeper processing and higher fidelity
- Text/semantic prompt control — small datasets make semantic concepts unreliable; sliders are more grounded. Future expansion
- Full musical compositions with structure (intro/development/ending) — v1 focuses on textures and building blocks. Future expansion
- Mobile app — desktop/browser-first
- DAW plugin (VST/AU) — standalone tool first, plugin integration later
- Cloud-hosted training service — runs locally on user's hardware

## Context

**Inspiration:** Autolume (Metacreation Lab) — a no-code system for training StyleGAN on personal image datasets with interactive latent space exploration. This project applies the same philosophy to audio.

**Prior art and frustrations:**
- **RAVE**: Right approach (train on your own audio, explore latent space) but parameters are opaque — latent dimensions don't map to musical concepts — and the quasi-real-time focus compromises output quality
- **Suno/commercial tools**: Too polished, rely on massive datasets of popular styles, output sounds derivative. No personal voice.

**The gap this fills:** Between RAVE's "train your own model" ethos and commercial tools' polish. Nothing currently offers controllable exploration of your own sound with musically meaningful parameters.

**Key technical challenges:**
- Making generative audio models work well with very small datasets (5–20 files is fundamentally different from 500)
- Mapping latent space dimensions to musically meaningful controls (the core innovation over RAVE)
- Achieving high-fidelity output (48kHz/24bit+) when many current models operate at 16–22kHz
- Supporting incremental training and model blending

**User profile:** Musicians, sound artists, experimental composers working across genres — ambient/textural, electronic, experimental/noise, acoustic/instrumental. The tool must be genre-agnostic, learning from whatever audio it's fed.

**Output usage:** Generated audio goes into DAW production, standalone listening, and live performance contexts. Export pipeline and format flexibility matter.

## Constraints

- **Audio quality**: 48kHz/24bit minimum baseline — lower is not acceptable for professional audio work
- **Hardware flexibility**: Must support Apple Silicon (MPS), NVIDIA (CUDA), and potentially cloud GPU. Cannot be locked to one platform
- **Dataset ethics**: Only user's own recordings or ethically/legally sourced material. No scraping, no copyright infringement
- **Small data**: The architecture must produce meaningful results with as few as 5–20 audio files, not just 500+
- **Tech stack**: Python-based (PyTorch ecosystem) — non-negotiable given the ML/audio library landscape

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Non-real-time generation | Quality over latency; allows deeper processing, higher fidelity, more complex models | — Pending |
| Parameter sliders over text prompts for v1 | Small datasets make semantic concepts unreliable; sliders map directly to latent space; musicians think in knobs | — Pending |
| Python + Gradio for v1 | Fastest path to usable creative tool; Gradio gives sliders/audio/upload natively; can upgrade UI later | — Pending |
| 48kHz/24bit baseline | Professional audio production standard; below this is not credible for serious audio work | — Pending |
| Genre-agnostic architecture | User works across ambient, electronic, experimental, acoustic — model must learn from any audio | — Pending |
| Novel approach emphasis | User wants research-grade innovation, not just wrapping existing models in a GUI | — Pending |

---
*Last updated: 2026-02-12 after initialization*
