---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: HiFi-GAN Vocoder
status: unknown
last_updated: "2026-02-28T23:06:57Z"
progress:
  total_phases: 16
  completed_phases: 15
  total_plans: 48
  completed_plans: 46
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters
**Current focus:** Phase 16 — Per-Model HiFi-GAN Training & Griffin-Lim Removal

## Current Position

Phase: 16 of 16 (Per-Model HiFi-GAN Training & Griffin-Lim Removal)
Plan: 3 of 5 in current phase -- COMPLETE
Status: Plan 16-03 complete, proceeding to Plan 16-04
Last activity: 2026-02-28 — Completed 16-03-PLAN.md (Training loop & inference)

Progress: ████████████░░░░░░░░ 64% (v1.1) [3/5 Phase 16 plans complete]

## Performance Metrics

**Velocity:**
- Total plans completed: 35 (v1.0)
- Average duration: 3 min
- Total execution time: 1.50 hours

**By Phase (v1.0 summary):**

| Phase | Plans | Avg/Plan |
|-------|-------|----------|
| Phases 1-11 | 35 total | 3 min |

**Recent Trend:**
- Last 5 plans (v1.0): 3min, 4min, 5min, 2min, 2min (avg 3.2min)
- Trend: Consistent

*Updated after each plan completion*
| Phase 12 P01 | 2min | 2 tasks | 75 files |
| Phase 12 P02 | 5min | 2 tasks | 3 files |
| Phase 12 P03 | 3min | 2 tasks | 3 files |
| Phase 13 P01 | 2min | 2 tasks | 2 files |
| Phase 13 P02 | 4min | 2 tasks | 8 files |
| Phase 14 P01 | 4min | 2 tasks | 2 files |
| Phase 14 P02 | 4min | 3 tasks | 6 files |
| Phase 15 P01 | 5min | 2 tasks | 5 files |
| Phase 15 P02 | 2min | 1 tasks | 1 files |
| Phase 16 P01 | 4min | 2 tasks | 5 files |
| Phase 16 P02 | 5min | 2 tasks | 5 files |
| Phase 16 P03 | 4min | 2 tasks | 4 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1 Milestone]: BigVGAN-v2 (122M params) as universal default, HiFi-GAN V2 (0.92M params) for optional per-model training
- [v1.1 Milestone]: Vendor BigVGAN source (MIT, ~50KB) rather than pip install (not on PyPI)
- [v1.1 Milestone]: MelAdapter converts log1p->log(clamp) at vocoder boundary; VAE pipeline unchanged
- [v1.1 Milestone]: Griffin-Lim fully removed (not kept as fallback) — removal in Phase 16 after BigVGAN proven
- [v1.1 Milestone]: librosa (new dep) for Slaney-normalized mel filterbanks matching BigVGAN training data
- [Phase 12]: VENDOR_PIN.txt with commit hash for BigVGAN version pinning (no git submodule)
- [Phase 12]: librosa added as full dependency (Slaney filterbank); numba/llvmlite transitive deps accepted
- [Phase 12]: Direct model loading from cached dir instead of from_pretrained (huggingface_hub mixin compat issue with vendored BigVGAN)
- [Phase 12]: Waveform round-trip (Griffin-Lim) for mel conversion -- simpler than transfer matrix, guaranteed correct BigVGAN mels
- [Phase 12]: Griffin-Lim quality loss accepted as stopgap -- Phase 16 per-model HiFi-GAN eliminates both BigVGAN OOD and mel conversion issues
- [Phase 13]: Omit vocoder_state key from saved dict when None (not null marker) -- cleaner serialization
- [Phase 13]: hasattr guard for optional vocoder field in display layers for backward compat
- [Phase 13]: VocoderInfo populated from vocoder_state training_metadata dict in save_model
- [Phase 14]: Internal sample rate derived from vocoder.sample_rate (44100) instead of hardcoded 48000
- [Phase 14]: Kaiser-windowed sinc resampler with lowpass_filter_width=64 for high-quality output
- [Phase 14]: GenerationConfig.sample_rate default changed from 48000 to 44100 (BigVGAN native)
- [Phase 14]: Training preview functions require vocoder parameter (no Griffin-Lim fallback) -- intentional hard break
- [Phase 14]: Preview sample_rate in PreviewEvent derived from vocoder.sample_rate, not spec_config.sample_rate
- [Phase 15]: Deferred pipeline creation: app_state.pipeline = None at model load, created at generate time with resolved vocoder
- [Phase 15]: Generator-based _generate_audio: yields intermediate button-disable update during download, then final results
- [Phase 15]: tqdm_class forwarded through entire vocoder chain for UI/CLI progress customization
- [Phase 15]: Blend mode vocoder resolution for status/JSON only; BlendEngine manages its own vocoder internally
- [Phase 15]: Rich progress disabled in JSON output mode to avoid polluting machine-readable output
- [Phase 15]: TqdmExperimentalWarning suppressed via warnings.filterwarnings for clean CLI output
- [Phase 16]: Upsample rates [8,8,4,2] (product=512) to match project hop_size; config validates at construction
- [Phase 16]: Loss functions as pure functions with torch.tensor(0.0, device=...) initialization for proper device/grad tracking
- [Phase 16]: Tikhonov regularization (alpha=1e-4) for stable filterbank transfer matrix in MelAdapter
- [Phase 16]: Direct mel-domain filterbank transfer replaces Griffin-Lim waveform round-trip
- [Phase 16]: Analyzer lazily creates BigVGAN vocoder for sweep reconstruction; vocoder parameter optional for backward compat
- [Phase 16]: Discriminator LR at 0.5x generator LR to prevent overfitting on small datasets
- [Phase 16]: Mel loss weight 45 (original HiFi-GAN paper) for strong reconstruction signal
- [Phase 16]: HiFiGANVocoder undoes log1p via expm1 -- no MelAdapter needed, trained on VAE mel format directly
- [Phase 16]: Auto-selection prefers per-model HiFi-GAN over BigVGAN when vocoder_state exists; low-epoch warning at <20

### Pending Todos

None yet.

### Blockers/Concerns

- MPS compatibility for BigVGAN inference with Snake activations unverified (untestable without Apple Silicon; defer to Phase 14 integration testing)
- ~~Quality of mel adapter path for v1.0 HTK-trained models needs listening test validation~~ RESOLVED: Human-verified in 12-03; Griffin-Lim adds distortion but accepted as stopgap for Phase 16
- HiFi-GAN V2 training convergence on 5-50 file datasets is unvalidated (handle in Phase 16)

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 16-03-PLAN.md (Training loop & inference)
Resume file: 16-04-PLAN.md
