# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-21)

**Core value:** Controllable exploration -- users can reliably navigate between sound worlds using discrete audio codes and generative priors
**Current focus:** Phase 14 - Autoregressive Prior

## Current Position

Phase: 14 of 18 (Autoregressive Prior)
Plan: 3 of 3 in current phase
Status: Phase Complete
Last activity: 2026-02-22 -- Completed 14-03-PLAN.md (Prior persistence + CLI command)

Progress: [================░░░░] 78% (v1.0 complete, Phase 12 complete, Phase 13 complete, Phase 14 complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 34 (v1.0)
- Average duration: ~12 min (v1.0)
- Total execution time: ~7 hours (v1.0)

**By Phase (v1.0):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1-11 (v1.0) | 34 | ~7h | ~12 min |

**By Phase (v1.1):**

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 12 | 01 | 4 min | 2 | 4 |
| 12 | 02 | 2 min | 2 | 2 |
| 13 | 01 | 7 min | 2 | 8 |
| 13 | 02 | 5 min | 2 | 2 |
| 13 | 03 | 2 min | 1 | 1 |
| 14 | 01 | 3 min | 2 | 2 |
| 14 | 02 | 4 min | 2 | 2 |
| 14 | 03 | 4 min | 3 | 5 |

**Recent Trend:**
- v1.0 shipped in 3 days across 11 phases
- v1.1 Phase 12: 6 min total (2 plans, 4 tasks, 6 files)
- v1.1 Phase 13: Complete (3/3 plans, 14 min total)
- v1.1 Phase 14: Complete (3/3 plans, 11 min total)
- Trend: Stable

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.1: Replace continuous VAE with RVQ-VAE (discrete codes for sharper reconstructions)
- v1.1: Use lucidrains/vector-quantize-pytorch for VQ/RVQ layers
- v1.1: Autoregressive prior (Transformer or LSTM -- resolve before Phase 14)
- v1.1: Clean break from v1.0 models (no backward compatibility)
- v1.1: Dataset-adaptive codebook sizing (64/128/256 for 5-20/20-100/100-500 files)
- 12-01: commitment_weight=0.25 (not library default 1.0) for small-dataset stability
- 12-01: EMA decay scales with dataset size: 0.8/0.9/0.95 for <=20/<=100/>100 files
- 12-01: Spatial embedding encoder (each position independently quantized, 48 per 1-sec mel)
- 12-01: QuantizerWrapper wraps ResidualVQ for codebook health monitoring
- 12-01: ~16x temporal compression (~167ms per position, region-level code editing)
- 12-02: No KL divergence in vqvae_loss -- commitment loss replaces KL per VQ-VAE design
- 12-02: Multi-scale spectral loss at 3 resolutions (full, 2x, 4x) averaged equally
- 12-02: v1.0/v1.1 section comments in __init__.py for Phase 13 migration cleanup
- 13-01: Parallel VQ training functions alongside v1.0 (no modifications to existing train/validate/save)
- 13-01: Codebook health computed every 10 steps during training, full validation set at epoch end
- 13-01: Skip codebook health at step 0 epoch 0 (k-means not initialized)
- 13-01: Low utilization warnings only after epoch 0 to avoid false positives
- 13-01: v2 .distill format with model_type="vqvae" and version=2
- 13-03: Remove --preset flag entirely (v1.0 KL presets irrelevant for VQ-VAE)
- 13-03: Show (auto)/(override) suffix on codebook_size in CLI config summary
- 13-03: Accumulate utilization warnings across epochs for end-of-training summary
- 13-02: Hide resume button for VQ-VAE (checkpoint resume needs runner adaptation)
- 13-02: Duck-type metric detection (hasattr val_commit_loss) in loss chart to avoid circular imports
- 13-02: Per-level codebook health as markdown table in stats panel (utilization%, perplexity, dead codes)
- [Phase 13]: Remove --preset flag entirely (v1.0 KL presets irrelevant for VQ-VAE)
- 14-01: Transformer prior (not LSTM) -- nn.TransformerEncoder + explicit causal mask for GPT-style decoder-only
- 14-01: Level embedding disambiguates RVQ quantizer levels in flattened code sequences
- 14-01: 3-tier adaptive prior scaling: <=20 files (128h/2L), 21-100 (256h/4L), >100 (512h/6L)
- 14-02: VQ-VAE frozen and only used for code extraction -- prior trains on pre-extracted tensors
- 14-02: Memorization thresholds relaxed and adaptive: 2.0/3.0/5.0 by dataset tier (<=20/<=100/>100 files)
- 14-02: Best checkpoint tracked via deepcopy(state_dict) in memory (prior models are small)
- 14-03: Atomic write (temp file + os.replace) for prior bundling into .distill files
- 14-03: Lazy CodePrior import in load_model_v2 to keep fast import time for non-prior loads
- 14-03: CLI train-prior mirrors train.py patterns (Rich, SIGINT, auto/override suffixes, JSON output)

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 14: RESOLVED -- Transformer prior chosen (nn.TransformerEncoder + causal mask, GPT-style decoder-only)
- Phase 12: Empirical codebook sizing defaults need validation on actual 5/50/500 file datasets
- Phase 16: Gradio code grid editor has no established prior art; needs early prototyping

## Session Continuity

Last session: 2026-02-22
Stopped at: Completed 14-03-PLAN.md -- Prior persistence + CLI. Phase 14 complete (3/3 plans).
Resume file: None
