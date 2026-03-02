# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.1 — HiFi-GAN Vocoder

**Shipped:** 2026-03-01
**Phases:** 5 | **Plans:** 14 | **Commits:** 70

### What Was Built
- BigVGAN-v2 universal vocoder with vendored source, abstract interface, auto-download weight manager
- MelAdapter for mel format conversion (VAE log1p → BigVGAN log-clamp), evolved from Griffin-Lim round-trip to direct filterbank transfer
- Model format v2 (.distillgan) with optional vocoder state bundling
- Full pipeline integration — all 5 generation paths through neural vocoder with Kaiser resampler
- UI Vocoder Settings accordion with lazy BigVGAN download, CLI --vocoder flag with Rich progress
- Per-model HiFi-GAN V2 adversarial training (MPD+MSD discriminators), training dashboard, train-vocoder CLI
- Griffin-Lim fully removed — neural vocoder is the only reconstruction path

### What Worked
- Phase-by-phase architecture: each phase built cleanly on the previous, minimal rework needed
- Research-first approach caught BigVGAN compatibility issues (from_pretrained, mel format) before implementation
- Separating BigVGAN integration (Phase 12) from pipeline wiring (Phase 14) — isolated concerns effectively
- Deferred pipeline creation pattern (Phase 15) cleanly solved the lazy download problem
- Context documents per phase captured decisions that later phases could reference

### What Was Inefficient
- MelAdapter went through two implementations: Griffin-Lim waveform round-trip (Phase 12) → direct filterbank transfer (Phase 16). Could have gone direct from the start if research had been deeper
- Milestone audit ran at Phase 14 completion, found only "unsatisfied" gaps for phases not yet started — provided limited value at that stage. Should audit closer to completion
- Phase 16 was the largest phase (5 plans) — could have been split into two phases for better granularity

### Patterns Established
- Vocoder interface abstraction (VocoderBase) enables future vocoder swaps without pipeline changes
- tqdm_class forwarding pattern for UI/CLI progress customization
- Generator-based Gradio handlers for intermediate UI updates during long operations
- VENDOR_PIN.txt pattern for vendored source version tracking

### Key Lessons
1. **Vendor early, vendor correctly** — BigVGAN's from_pretrained was incompatible with vendored source; direct model loading was the right approach from the start
2. **MelAdapter is a critical boundary** — format conversion between VAE and vocoder mel representations is the hardest integration point; invest more research time here
3. **Small-dataset GAN training needs special attention** — discriminator LR at 0.5x generator, data augmentation, mel loss weight 45 — these hyperparameters matter for 5-50 file datasets
4. **Deferred initialization pays off** — creating pipeline/vocoder at use time (not load time) enables lazy downloads and conditional vocoder selection

### Cost Observations
- Model mix: ~80% opus, ~15% sonnet, ~5% haiku (research and verification agents)
- Sessions: ~8 sessions across 16 days
- Notable: Average plan execution stayed at ~3.7 min despite increased complexity vs v1.0 (3.0 min)

---

## Milestone: v1.0 — MVP

**Shipped:** 2026-02-15
**Phases:** 11 | **Plans:** 35 | **Commits:** 149

### What Was Built
- VAE training on small audio datasets (5-500 files) with overfitting prevention
- PCA-based latent space analysis for musically meaningful slider control
- Professional audio export (WAV/MP3/FLAC/OGG) with spatial processing
- 4-tab Gradio UI with live training, generation, model library, settings
- CLI for batch generation, training, and model management
- Model persistence (.sda format) with library management

### What Worked
- Extremely fast execution: 11 phases, 35 plans in ~3 days
- Consistent plan velocity (~3 min average)
- Clean phase separation enabled parallel-friendly architecture

### What Was Inefficient
- 16 human listening tests deferred (Phases 4, 5, 10) — accumulated verification debt
- HRTF SOFA file not bundled — user friction for binaural mode

### Key Lessons
1. **Phase granularity matters** — 1-5 plans per phase kept each phase manageable
2. **Atomic write patterns prevent data loss** — temp file + os.replace pattern proven reliable

### Cost Observations
- Sessions: ~5 sessions across 3 days
- Notable: Highly efficient; most plans completed in 2-5 minutes

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Commits | Phases | Plans | Avg/Plan | Key Change |
|-----------|---------|--------|-------|----------|------------|
| v1.0 | 149 | 11 | 35 | 3.0 min | Established baseline |
| v1.1 | 70 | 5 | 14 | 3.7 min | Added research phase, context docs |

### Cumulative Quality

| Milestone | Verification | Requirements | Dropped |
|-----------|-------------|--------------|---------|
| v1.0 | 11/11 phases passed | 16/16 complete | 0 |
| v1.1 | 5/5 phases passed | 24/25 complete | 1 (PERS-02) |

### Top Lessons (Verified Across Milestones)

1. Phase-by-phase execution with clear success criteria keeps velocity high even as complexity grows
2. Research-first approach catches integration issues before implementation — saves rework
3. Abstract interfaces at boundaries (VocoderBase, persistence format) enable future evolution
