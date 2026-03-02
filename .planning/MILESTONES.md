# Milestones
## v1.1 HiFi-GAN Vocoder (Shipped: 2026-03-01)

**Delivered:** Neural vocoder reconstruction replacing Griffin-Lim — BigVGAN-v2 as universal default plus optional per-model HiFi-GAN V2 training for maximum fidelity.

**Phases:** 12-16 (5 phases, 14 plans, ~28 tasks)
**Stats:** 21,762 LOC Python | 70 commits | 57 Python files modified (+8,183/-195) | 16 days (2026-02-12 to 2026-02-28)
**Git range:** 6ad7aab → e0bf03b

**Key accomplishments:**
1. BigVGAN-v2 universal vocoder with vendored source, abstract interface, auto-download weight manager, and mel format adapter
2. Model format v2 (.distillgan) with optional vocoder state bundling and v1 format rejection
3. Full pipeline integration — all 5 generation paths wired through neural vocoder with Kaiser resampler (44.1kHz → 48kHz)
4. UI Vocoder Settings accordion with lazy download, CLI --vocoder flag with Rich progress bar
5. Per-model HiFi-GAN V2 adversarial training pipeline (MPD+MSD discriminators), training dashboard UI, train-vocoder CLI command
6. Griffin-Lim reconstruction code fully removed — neural vocoder is the only path

**Tech debt carried forward:**
- MPS compatibility for BigVGAN Snake activations unverified (no Apple Silicon test hardware)
- HiFi-GAN V2 training convergence on 5-50 file datasets unvalidated with real user data
- BlendEngine creates fresh BigVGANVocoder (~489MB) per blend_generate() call — should cache

---


## v1.0 MVP (Shipped: 2026-02-15)

**Delivered:** A generative audio application that trains on small personal datasets (5-500 files) and generates high-fidelity audio with musically meaningful slider controls, full GUI and CLI, multi-format export, and spatial audio.

**Phases:** 1-11 (11 phases, 34 plans, ~70 tasks)
**Stats:** 17,520 LOC Python | 149 commits | 186 files | 3 days (2026-02-12 to 2026-02-14)
**Git range:** initial commit → c88be10

**Key accomplishments:**
1. Complete AI audio generation from small datasets (5-500 files) with VAE training and overfitting prevention
2. Musical parameter control via PCA-based latent space analysis with safe/warning zones
3. Professional audio export in WAV/MP3/FLAC/OGG with spatial processing (stereo/binaural)
4. Full-featured Gradio web UI with live training, slider controls, model library, presets, A/B comparison
5. Scriptable CLI for batch generation, training, and model management on headless servers
6. Model persistence as .sda files with library management and instant slider restoration on load

**Tech debt carried forward:**
- 16 human listening/verification tests pending (Phases 4, 5, 10)
- HRTF SOFA file not bundled for binaural mode (user must download)

---

