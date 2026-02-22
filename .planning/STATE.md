# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters
**Current focus:** Phase 13 — Model Persistence v2

## Current Position

Phase: 13 of 16 (Model Persistence v2)
Plan: 2 of 2 in current phase -- COMPLETE
Status: Phase 13 COMPLETE
Last activity: 2026-02-22 — Completed 13-02-PLAN.md (catalog, CLI, UI sweep for .distillgan)

Progress: ██████░░░░░░░░░░░░░░ 30% (v1.1) [2/2 Phase 13 plans complete]

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

### Pending Todos

None yet.

### Blockers/Concerns

- MPS compatibility for BigVGAN inference with Snake activations unverified (untestable without Apple Silicon; defer to Phase 14 integration testing)
- ~~Quality of mel adapter path for v1.0 HTK-trained models needs listening test validation~~ RESOLVED: Human-verified in 12-03; Griffin-Lim adds distortion but accepted as stopgap for Phase 16
- HiFi-GAN V2 training convergence on 5-50 file datasets is unvalidated (handle in Phase 16)

## Session Continuity

Last session: 2026-02-22
Stopped at: Completed 13-02-PLAN.md (Phase 13 complete)
Resume file: Phase 14 next
