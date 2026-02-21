---
phase: 12-rvq-vae-core-architecture
plan: 01
subsystem: models
tags: [vq-vae, rvq, vector-quantize-pytorch, codebook, mel-spectrogram]

# Dependency graph
requires:
  - phase: 11-wire-latent-space-analysis
    provides: "v1.0 codebase with ConvVAE, training config, model persistence"
provides:
  - "ConvVQVAE model with VQEncoder, VQDecoder, QuantizerWrapper"
  - "VQVAEConfig dataclass with all VQ-specific parameters"
  - "get_adaptive_vqvae_config() for dataset-size-adaptive codebook sizing"
  - "vector-quantize-pytorch library installed as dependency"
affects: [12-02, 13-vq-vae-training-pipeline, 14-autoregressive-prior, 16-encode-decode-code-visualization]

# Tech tracking
tech-stack:
  added: [vector-quantize-pytorch>=1.27.0, einops>=0.8.0, einx>=0.1.3]
  patterns: [spatial-embedding-encoder, rvq-wrapper-with-monitoring, pad-crop-round-trip, dataset-adaptive-config]

key-files:
  created: [src/distill/models/vqvae.py]
  modified: [pyproject.toml, src/distill/training/config.py, uv.lock]

key-decisions:
  - "commitment_weight=0.25 (not library default 1.0) for small-dataset stability"
  - "EMA decay scales with dataset size: 0.8/0.9/0.95 for <=20/<=100/>100 files"
  - "Spatial embedding encoder (not global vector) -- each position independently quantized"
  - "No lazy init in VQ encoder/decoder -- Conv2d layers are spatially independent"
  - "QuantizerWrapper wraps ResidualVQ (not used directly) for health monitoring"

patterns-established:
  - "Spatial embedding encoder: [B,1,n_mels,time] -> [B,codebook_dim,H,W] via Conv2d + 1x1 proj"
  - "RVQ reshape protocol: permute(0,2,3,1).reshape(B,H*W,D) before, reshape(B,H,W,D).permute(0,3,1,2) after"
  - "Dataset-adaptive VQ config: get_adaptive_vqvae_config(file_count) parallels get_adaptive_config()"
  - "VQ forward returns (recon, indices, commit_loss) -- no mu/logvar/KL"

requirements-completed: [VQVAE-01, VQVAE-02, VQVAE-03, VQVAE-06]

# Metrics
duration: 4min
completed: 2026-02-21
---

# Phase 12 Plan 01: RVQ-VAE Core Architecture Summary

**ConvVQVAE model with 4-layer conv encoder/decoder, ResidualVQ bottleneck via vector-quantize-pytorch, and dataset-adaptive codebook config (64/128/256)**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-21T23:54:13Z
- **Completed:** 2026-02-21T23:58:17Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- ConvVQVAE model with configurable RVQ levels (2-4) produces correct forward pass: input [B,1,128,94] -> recon [B,1,128,94] + indices [B,48,3] + commit_loss scalar
- VQVAEConfig dataclass with dataset-adaptive sizing: 64/128/256 codebook entries for <=20/<=100/>100 files
- QuantizerWrapper provides per-level codebook utilization, perplexity, and dead code tracking
- vector-quantize-pytorch v1.27.21 installed with ResidualVQ featuring k-means init, EMA updates, and dead code reset

## Task Commits

Each task was committed atomically:

1. **Task 1: Install vector-quantize-pytorch and create VQVAEConfig with adaptive sizing** - `0d178c4` (feat)
2. **Task 2: Create ConvVQVAE model with VQEncoder, VQDecoder, and QuantizerWrapper** - `fe47aeb` (feat)

## Files Created/Modified
- `src/distill/models/vqvae.py` - NEW: ConvVQVAE model with VQEncoder, VQDecoder, QuantizerWrapper (547 lines)
- `src/distill/training/config.py` - MODIFIED: Added VQVAEConfig dataclass and get_adaptive_vqvae_config function
- `pyproject.toml` - MODIFIED: Added vector-quantize-pytorch>=1.27.0 dependency
- `uv.lock` - MODIFIED: Lock file updated with new dependency tree

## Decisions Made
- **commitment_weight=0.25**: Lower than library default (1.0) per VQ-VAE paper recommendation, safer for small datasets where encoder-codebook drift must be gentle
- **EMA decay scaling**: 0.8 for <=20 files (fast adaptation), 0.9 for <=100 files, 0.95 for >100 files -- faster decay prevents codebook stagnation on small data
- **Spatial embedding encoder**: Each spatial position independently quantized (48 positions for 1-sec mel), following SoundStream/EnCodec pattern -- NOT a global vector bottleneck
- **QuantizerWrapper over direct ResidualVQ**: Thin wrapper adds codebook health monitoring (utilization, perplexity, dead codes) without modifying library behavior
- **~16x temporal compression**: 94 time frames -> 6 positions per second (~167ms per position), suitable for region-level code editing

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- ConvVQVAE model ready for Plan 12-02 (loss function and __init__.py exports)
- VQVAEConfig ready for Phase 13 (training pipeline integration)
- QuantizerWrapper monitoring methods ready for Phase 13 (codebook health UI)
- codes_to_embeddings method ready for Phase 16 (encode/decode path)
- v1.0 code completely untouched -- ConvVAE, losses.py, persistence.py all preserved

## Self-Check: PASSED

- FOUND: src/distill/models/vqvae.py
- FOUND: src/distill/training/config.py
- FOUND: .planning/phases/12-rvq-vae-core-architecture/12-01-SUMMARY.md
- FOUND: commit 0d178c4
- FOUND: commit fe47aeb

---
*Phase: 12-rvq-vae-core-architecture*
*Completed: 2026-02-21*
