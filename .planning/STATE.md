# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Controllable exploration — users can reliably navigate between sound worlds using musically meaningful parameters
**Current focus:** Phase 15 — UI & CLI Vocoder Controls

## Current Position

Phase: 15 of 16 (UI & CLI Vocoder Controls)
Plan: 1 of 2 in current phase -- COMPLETE
Status: Plan 15-01 complete, ready for Plan 15-02
Last activity: 2026-02-28 — Completed 15-01-PLAN.md (UI vocoder controls)

Progress: ████████░░░░░░░░░░░░ 45% (v1.1) [1/2 Phase 15 plans complete]

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

### Pending Todos

None yet.

### Blockers/Concerns

- MPS compatibility for BigVGAN inference with Snake activations unverified (untestable without Apple Silicon; defer to Phase 14 integration testing)
- ~~Quality of mel adapter path for v1.0 HTK-trained models needs listening test validation~~ RESOLVED: Human-verified in 12-03; Griffin-Lim adds distortion but accepted as stopgap for Phase 16
- HiFi-GAN V2 training convergence on 5-50 file datasets is unvalidated (handle in Phase 16)

## Session Continuity

Last session: 2026-02-28
Stopped at: Completed 15-01-PLAN.md (UI vocoder controls)
Resume file: 15-02-PLAN.md (CLI vocoder controls)
