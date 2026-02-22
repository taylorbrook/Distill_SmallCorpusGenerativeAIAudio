---
phase: 14-autoregressive-prior
plan: 01
subsystem: models
tags: [transformer, autoregressive, prior, vq-vae, codebook, pytorch]

# Dependency graph
requires:
  - phase: 12-vqvae-model
    provides: ConvVQVAE model with RVQ producing [B, seq_len, num_quantizers] indices
provides:
  - CodePrior GPT-style transformer model for autoregressive code prediction
  - flatten_codes / unflatten_codes for sequence reshaping
  - extract_code_sequences for frozen VQ-VAE dataset encoding
  - PriorConfig dataclass with all user-facing hyperparameters
  - get_adaptive_prior_config for dataset-scaled model sizing
affects: [14-02 prior training loop, 14-03 prior CLI/persistence, 15-generation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "nn.TransformerEncoder + explicit causal mask for decoder-only GPT architecture"
    - "Three-embedding sum (token + position + level) for RVQ-aware autoregressive modeling"
    - "3-tier adaptive prior config scaling (128/256/512 hidden_size by dataset size)"

key-files:
  created:
    - src/distill/models/prior.py
    - src/distill/training/prior_config.py
  modified: []

key-decisions:
  - "nn.TransformerEncoder with explicit mask tensor (not is_causal flag) for cross-backend compatibility"
  - "Level embedding added to disambiguate quantizer levels in flattened code sequences"
  - "3-tier prior scaling: <=20 files (128h/2L), 21-100 (256h/4L), >100 (512h/6L)"

patterns-established:
  - "Decoder-only transformer via nn.TransformerEncoder + upper-triangular causal mask buffer"
  - "Pure-Python config dataclass with companion get_adaptive_*_config() factory"

requirements-completed: [GEN-01]

# Metrics
duration: 3min
completed: 2026-02-22
---

# Phase 14 Plan 01: CodePrior Model and Adaptive Config Summary

**GPT-style CodePrior transformer with level-aware embeddings over flattened VQ-VAE code sequences, plus 3-tier adaptive PriorConfig scaling**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-22T06:05:59Z
- **Completed:** 2026-02-22T06:09:00Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- CodePrior model accepts [B, T] code indices and returns [B, T, codebook_size] logits via causal self-attention
- Level embedding disambiguates RVQ quantizer levels within the flattened sequence (token + position + level sum)
- PriorConfig + get_adaptive_prior_config() follows same 3-tier pattern as VQVAEConfig for dataset-scaled model sizing
- extract_code_sequences() encodes entire datasets through frozen VQ-VAE for prior training data

## Task Commits

Each task was committed atomically:

1. **Task 1: Create CodePrior transformer model with code flattening utilities** - `40afe77` (feat)
2. **Task 2: Create PriorConfig dataclass with dataset-adaptive scaling** - `1789a94` (feat)

## Files Created/Modified
- `src/distill/models/prior.py` - CodePrior transformer model, flatten_codes, unflatten_codes, extract_code_sequences (257 lines)
- `src/distill/training/prior_config.py` - PriorConfig dataclass, get_adaptive_prior_config factory (155 lines)

## Decisions Made
- Used nn.TransformerEncoder with explicit causal mask tensor rather than is_causal=True flag for guaranteed cross-backend compatibility (CPU, CUDA, MPS)
- Added level embedding (nn.Embedding(num_quantizers, hidden_size)) to disambiguate RVQ quantizer levels in flattened sequences -- cheap (3 vectors) but provides crucial structural information
- Prior model scales with dataset: 128h/2L for <=20 files, 256h/4L for 21-100, 512h/6L for >100 -- prevents memorization on small datasets
- Used string type annotation for ConvVQVAE in extract_code_sequences to avoid circular imports

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CodePrior model ready for training loop implementation (14-02)
- PriorConfig ready for CLI flag integration (14-03)
- extract_code_sequences ready to produce training data from frozen VQ-VAE models
- flatten_codes/unflatten_codes ready for generation pipeline (Phase 15)

## Self-Check: PASSED

- [x] src/distill/models/prior.py exists (257 lines, min 120)
- [x] src/distill/training/prior_config.py exists (155 lines, min 60)
- [x] Commit 40afe77 found in git log
- [x] Commit 1789a94 found in git log
- [x] All 6 verifications passed

---
*Phase: 14-autoregressive-prior*
*Completed: 2026-02-22*
