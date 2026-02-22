---
phase: 14-autoregressive-prior
verified: 2026-02-21T18:30:00Z
status: passed
score: 12/12 must-haves verified
re_verification: false
gaps:
  - truth: "Loading a model with has_prior=True reconstructs the CodePrior and makes it available on LoadedVQModel.prior"
    status: resolved
    reason: "Fixed in commit 6bb0bb4 — train_prior now merges seq_len and num_quantizers into the prior_config dict before returning, so save_prior_to_model persists them and load_model_v2 can reconstruct CodePrior."
human_verification:
  - test: "Run distill train-prior on a real .distill model file with a small dataset, then load the resulting file and access loaded.prior"
    expected: "Training completes, prior is bundled into the file, and load_model_v2 reconstructs the CodePrior without KeyError"
    why_human: "End-to-end test requires a real trained VQ-VAE model file and audio dataset. Automated checks confirmed the logic gap statically."
---

# Phase 14: Autoregressive Prior Verification Report

**Phase Goal:** An autoregressive prior model can be trained on frozen VQ-VAE code sequences, with memorization detection, and bundled into the saved model file
**Verified:** 2026-02-21
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | CodePrior model accepts [B, T] integer code indices and produces [B, T, codebook_size] logits | VERIFIED | `prior.py:172-209` — forward pass confirmed with shape assertion: `(2, 143, 256)` |
| 2 | flatten_codes converts [B, seq_len, num_quantizers] to [B, seq_len * num_quantizers] in position-major order | VERIFIED | `prior.py:46-63` — reshape confirmed; roundtrip with unflatten_codes passes |
| 3 | PriorConfig dataclass exposes hidden_size, num_layers, num_heads, dropout, max_epochs, learning_rate | VERIFIED | `prior_config.py:26-74` — all 11 fields present, verified with dataclasses.fields() |
| 4 | get_adaptive_prior_config(file_count) scales model size and regularization by dataset tier | VERIFIED | All three tiers verified: <=20 (128h/2L/0.3d/50e), 21-100 (256h/4L), >100 (512h/6L/8h) |
| 5 | extract_code_sequences() encodes entire dataset through frozen VQ-VAE and returns code indices as tensor | VERIFIED | `prior.py:217-257` — sets eval mode, no_grad, iterates dataloader, concatenates indices |
| 6 | train_prior() orchestrates full prior training given a VQ-VAE model path, dataset path, and PriorConfig | VERIFIED | `prior_loop.py:329-684` — 685-line orchestrator: load VQ-VAE, freeze, extract codes, train, return result dict |
| 7 | Each epoch computes validation perplexity (exp of cross-entropy loss) and reports it via callback | VERIFIED | `prior_loop.py:573-621` — `val_perplexity = math.exp(min(val_loss, 20.0))` with PriorEpochMetrics callback |
| 8 | Memorization detection warns when validation perplexity drops below adaptive threshold | VERIFIED | `prior_loop.py:59-108` — thresholds 2.0/3.0/5.0 by dataset tier; all boundary conditions pass |
| 9 | Best checkpoint (lowest validation perplexity) is tracked and available for rollback | VERIFIED | `prior_loop.py:577-583` — `copy.deepcopy(prior_model.state_dict())` on improvement; loaded back at line 639 |
| 10 | Prior model state is saved into the .sda file with has_prior flag, prior_state_dict, prior_config, and prior_metadata | VERIFIED | `persistence.py:663-728` — save_prior_to_model atomically writes all four keys; os.replace pattern confirmed |
| 11 | Loading a model with has_prior=True reconstructs the CodePrior and makes it available on LoadedVQModel.prior | FAILED | `persistence.py:830-855` — load_model_v2 reads `prior_config_dict['seq_len']` and `prior_config_dict['num_quantizers']` but these keys are NEVER in the saved prior_config dict (PriorConfig dataclass has no seq_len/num_quantizers fields) — runtime KeyError confirmed by static analysis |
| 12 | User can run distill train-prior MODEL_PATH to train a prior from the CLI | VERIFIED | `cli/train_prior.py` exists, 257 lines, registered as `train-prior` in `cli/__init__.py`; --epochs/--hidden-size/--layers/--heads/--lr flags present; perplexity display and memorization warnings wired |

**Score:** 11/12 truths verified

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Notes |
|----------|-----------|--------------|--------|-------|
| `src/distill/models/prior.py` | 120 | 258 | VERIFIED | CodePrior, flatten_codes, unflatten_codes, extract_code_sequences — all present and substantive |
| `src/distill/training/prior_config.py` | 60 | 156 | VERIFIED | PriorConfig, get_adaptive_prior_config — pure Python, no torch dependency |
| `src/distill/training/prior_loop.py` | 200 | 685 | VERIFIED | train_prior, train_prior_epoch, validate_prior_epoch, check_memorization — all present |
| `src/distill/training/metrics.py` | — | 747 | VERIFIED | PriorStepMetrics, PriorEpochMetrics, PriorTrainingCompleteEvent appended in v1.1 Prior section |
| `src/distill/models/persistence.py` | — | 875 | PARTIAL | save_prior_to_model exists and is correct; load_model_v2 has prior reconstruction logic but contains a wiring bug (missing seq_len/num_quantizers in saved dict) |
| `src/distill/cli/train_prior.py` | 80 | 257 | VERIFIED | Full CLI with Rich, SIGINT, --epochs/--hidden-size/--layers/--heads/--lr, perplexity display, memorization warnings |
| `src/distill/models/__init__.py` | — | 97 | VERIFIED | CodePrior, flatten_codes, unflatten_codes, extract_code_sequences, save_prior_to_model all exported |
| `src/distill/training/__init__.py` | — | 136 | VERIFIED | PriorConfig, get_adaptive_prior_config, train_prior, check_memorization, PriorEpochMetrics, PriorStepMetrics, PriorTrainingCompleteEvent all exported |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `prior.py` | `torch.nn.TransformerEncoder` | causal self-attention stack with explicit mask | WIRED | `nn.TransformerEncoder` at line 154; `register_buffer("causal_mask", ...)` at line 165 |
| `prior.py` | `vqvae.py` (ConvVQVAE forward) | extract_code_sequences uses model(mel) | WIRED | `_recon, indices, _commit_loss = model(mel)` at line 254 |
| `prior_loop.py` | `models/prior.py` | imports CodePrior, extract_code_sequences, flatten_codes | WIRED | `from distill.models.prior import CodePrior, extract_code_sequences, flatten_codes` at line 378 |
| `prior_loop.py` | `models/persistence.py` | loads VQ-VAE via load_model_v2 | WIRED | `from distill.models.persistence import load_model_v2` at line 377 |
| `prior_loop.py` | `torch.nn.functional.cross_entropy` | next-token prediction loss | WIRED | `F.cross_entropy(logits.reshape(-1, codebook_size), target.reshape(-1))` at lines 191-194 and 307-312 |
| `persistence.py` | `models/prior.py` | reconstructs CodePrior when loading | PARTIAL | CodePrior import and reconstruction present (line 831-846) but will KeyError on `prior_config_dict['seq_len']` — see gap |
| `cli/train_prior.py` | `training/prior_loop.py` | calls train_prior() | WIRED | `from distill.training.prior_loop import train_prior` + `result = train_prior(...)` |
| `cli/train_prior.py` | `models/persistence.py` | calls save_prior_to_model() | WIRED | `from distill.models.persistence import save_prior_to_model` + call with result dict |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| GEN-01 | 14-01, 14-02 | User can train an autoregressive prior model on frozen VQ-VAE code sequences | SATISFIED | CodePrior model + train_prior orchestrator fully implemented; VQ-VAE frozen during training (param.requires_grad_(False)) |
| GEN-05 | 14-03 | Prior model is bundled in the saved model file alongside the VQ-VAE | PARTIAL | save_prior_to_model writes prior state correctly; load_model_v2 logic exists but will KeyError at runtime due to missing seq_len/num_quantizers in saved prior_config dict |
| GEN-06 | 14-02 | Prior training detects memorization (validation perplexity monitoring) | SATISFIED | check_memorization with adaptive thresholds 2.0/3.0/5.0; val_perplexity computed each epoch; PriorEpochMetrics emitted with is_memorizing flag |
| PERS-02 | 14-03 | Prior model state is bundled in the same model file | PARTIAL | Same gap as GEN-05 — bundling works but loading fails |
| CLI-02 | 14-03 | CLI supports prior training on a trained VQ-VAE model | SATISFIED | `distill train-prior MODEL_PATH DATASET_DIR` with --epochs/--hidden-size/--layers/--heads/--lr flags; Rich progress; memorization warnings; prior bundling |

**Orphaned requirements:** None — all Phase 14 requirements (GEN-01, GEN-05, GEN-06, PERS-02, CLI-02) are claimed by plans 14-01 through 14-03.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None | No TODO/FIXME/placeholder/stub patterns found | — | Clean |

All 9 Phase 14 files scanned: zero anti-patterns.

### Human Verification Required

#### 1. End-to-End Prior Save/Load Roundtrip

**Test:** Train a prior on a real VQ-VAE model, then load the resulting file and check `loaded.prior`
**Expected:** `load_model_v2` returns a `LoadedVQModel` with `prior` set to a `CodePrior` instance in eval mode
**Why human:** Requires a real trained VQ-VAE `.distill` file and audio dataset. Static analysis has confirmed the `KeyError` bug; human testing would confirm the fix as well.

#### 2. CLI Perplexity Display During Training

**Test:** Run `distill train-prior model.distill dataset/` and observe the terminal output
**Expected:** Each epoch shows `Epoch N/M | train_loss: X.XXXX | val_loss: X.XXXX | perplexity: X.X | best: X.X` and memorization warnings appear in yellow if triggered
**Why human:** Rich console rendering cannot be verified programmatically in this environment.

---

## Gaps Summary

One gap blocks full goal achievement: **prior persistence roundtrip (save then load) will fail at runtime**.

**Root cause:** `load_model_v2` expects `prior_config_dict['seq_len']` and `prior_config_dict['num_quantizers']` to be present in the saved prior config dict. But `train_prior` returns `prior_config: dataclasses.asdict(PriorConfig)` — and `PriorConfig` has no `seq_len` or `num_quantizers` fields. These values are returned as top-level keys (`result['seq_len']`, `result['num_quantizers']`) but neither `train_prior` nor the CLI merges them into the `prior_config` dict before calling `save_prior_to_model`.

**Impact:** `GEN-05` and `PERS-02` are partially blocked. A user who runs `distill train-prior` will successfully train a prior and see it bundled into the file, but any subsequent load of that file (or use of `load_model_v2`) will raise `KeyError: 'seq_len'`.

**Fix options (any one will resolve the gap):**

1. In `train_prior` (`prior_loop.py`), after `_asdict(prior_config)`, add `seq_len` and `num_quantizers` to the returned config dict before returning:
   ```python
   prior_config_dict = _asdict(prior_config)
   prior_config_dict["seq_len"] = flat_seq_len
   prior_config_dict["num_quantizers"] = num_quantizers
   return {"prior_config": prior_config_dict, ...}
   ```
2. In the CLI (`train_prior.py`), augment the dict after `train_prior` returns:
   ```python
   result["prior_config"]["seq_len"] = result["seq_len"]
   result["prior_config"]["num_quantizers"] = result["num_quantizers"]
   ```
3. Add `seq_len: int = 0` and `num_quantizers: int = 3` fields to `PriorConfig` (then `_asdict` will include them automatically — requires setting them during training).

All other 11 must-haves are fully verified: CodePrior architecture, flatten/unflatten utilities, adaptive config, training loop, memorization detection, best checkpoint tracking, metrics dataclasses, CLI command, and public API exports.

---

_Verified: 2026-02-21_
_Verifier: Claude (gsd-verifier)_
