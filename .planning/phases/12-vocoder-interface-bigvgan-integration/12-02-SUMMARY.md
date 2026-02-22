---
phase: 12-vocoder-interface-bigvgan-integration
plan: 02
subsystem: vocoder
tags: [bigvgan, huggingface_hub, weight-download, neural-vocoder, inference]

# Dependency graph
requires:
  - phase: 12-01
    provides: Vendored BigVGAN source, VocoderBase abstract class, huggingface_hub dependency
provides:
  - BigVGAN weight download/caching via ensure_bigvgan_weights()
  - BigVGANVocoder class implementing VocoderBase with mel_to_waveform inference
  - get_vocoder("bigvgan") factory returning ready-to-use vocoder instance
affects: [12-03, phase-14, phase-16]

# Tech tracking
tech-stack:
  added: []
  patterns: [direct-model-loading-from-cache, vendored-import-via-syspath, lazy-getattr-import]

key-files:
  created:
    - src/distill/vocoder/weight_manager.py
    - src/distill/vocoder/bigvgan_vocoder.py
  modified:
    - src/distill/vocoder/__init__.py

key-decisions:
  - "Direct model loading from cached directory instead of from_pretrained (avoids huggingface_hub mixin API compatibility issue)"
  - "sys.path manipulation left permanent for vendored imports (BigVGAN internal imports need path available)"
  - "Lazy __getattr__ in __init__.py for BigVGANVocoder to avoid loading torch at package import time"

patterns-established:
  - "Vendored import: sys.path.insert(0, vendor_dir) with path left in place for internal imports"
  - "Weight management: snapshot_download for caching, direct torch.load for model instantiation"
  - "Lazy class export: __getattr__ pattern in package __init__ for heavy imports"

requirements-completed: [VOC-01, VOC-03, VOC-04]

# Metrics
duration: 5min
completed: 2026-02-22
---

# Phase 12 Plan 02: BigVGAN Vocoder Wrapper & Weight Manager Summary

**BigVGAN-v2 vocoder wrapper with automatic HuggingFace Hub weight download, cross-platform inference (CUDA/MPS/CPU), and get_vocoder() factory returning 44.1kHz waveforms from mel spectrograms**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-22T00:25:41Z
- **Completed:** 2026-02-22T00:30:37Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Weight manager downloads BigVGAN weights (~489MB) on first use with HuggingFace Hub progress bar, caches in default HF cache, serves offline on subsequent calls
- BigVGANVocoder class loads vendored 122M-parameter BigVGAN model with use_cuda_kernel=False for cross-platform support
- Dummy mel tensor [1, 128, 10] produces waveform [1, 1, 5120] confirming correct 512x upsampling
- get_vocoder("bigvgan") factory returns ready-to-use VocoderBase instance at 44100 Hz sample rate

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement weight manager** - `8651c7a` (feat)
2. **Task 2: BigVGANVocoder class + get_vocoder() factory** - `bc7b067` (feat)

## Files Created/Modified
- `src/distill/vocoder/weight_manager.py` - BigVGAN weight download, caching, and offline loading via HuggingFace Hub
- `src/distill/vocoder/bigvgan_vocoder.py` - BigVGAN-v2 vocoder wrapper with vendored import mechanism and direct model loading
- `src/distill/vocoder/__init__.py` - Updated with get_vocoder() factory wiring and lazy BigVGANVocoder import

## Decisions Made
- Used direct model loading (torch.load + load_state_dict) instead of BigVGAN's from_pretrained, because the vendored BigVGAN code's _from_pretrained signature is incompatible with the current huggingface_hub version (missing proxies/resume_download kwargs). This avoids modifying vendored source files.
- Left vendor/bigvgan/ in sys.path permanently rather than using a context manager, because BigVGAN's internal imports (from env, from activations, etc.) need the path available during model construction and inference.
- Used __getattr__ lazy import pattern for BigVGANVocoder in the package __init__.py to keep torch loading deferred until actual use.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Bypassed BigVGAN from_pretrained API compatibility issue**
- **Found during:** Task 2 (BigVGANVocoder implementation)
- **Issue:** Vendored BigVGAN's `_from_pretrained` method expects `proxies` and `resume_download` keyword args that the installed huggingface_hub version no longer passes, causing TypeError
- **Fix:** Replaced `from_pretrained` call with direct loading: load config.json with `load_hparams_from_json`, instantiate BigVGAN(h, use_cuda_kernel=False), and load weights with `torch.load` + `load_state_dict` from the cached directory returned by `ensure_bigvgan_weights()`
- **Files modified:** src/distill/vocoder/bigvgan_vocoder.py
- **Verification:** get_vocoder("bigvgan") loads model successfully and produces correct waveform output
- **Committed in:** bc7b067 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential fix to make vendored BigVGAN compatible with current huggingface_hub without modifying vendored source. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviation above.

## User Setup Required
None - BigVGAN weights download automatically on first use. No external service configuration required.

## Next Phase Readiness
- BigVGAN vocoder loads and runs inference on any available device
- get_vocoder() factory returns working BigVGANVocoder instance
- Ready for MelAdapter implementation (Plan 03) to bridge VAE mel format to BigVGAN mel format
- Weight manager provides ensure_bigvgan_weights() for pre-download scenarios

## Self-Check: PASSED

All files verified present, all commits verified in git log.

---
*Phase: 12-vocoder-interface-bigvgan-integration*
*Completed: 2026-02-22*
