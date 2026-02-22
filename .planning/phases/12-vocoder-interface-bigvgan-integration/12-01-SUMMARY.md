---
phase: 12-vocoder-interface-bigvgan-integration
plan: 01
subsystem: vocoder
tags: [bigvgan, vendoring, huggingface_hub, librosa, abstract-interface]

# Dependency graph
requires: []
provides:
  - Vendored BigVGAN source code in vendor/bigvgan/ with version pinning
  - Abstract VocoderBase class defining mel_to_waveform contract
  - huggingface_hub and librosa as project dependencies
  - get_vocoder() factory function stub
affects: [12-02, 12-03, phase-14, phase-16]

# Tech tracking
tech-stack:
  added: [huggingface_hub, librosa, numba, llvmlite]
  patterns: [vendor-pin-txt, lazy-torch-import, abstract-vocoder-interface]

key-files:
  created:
    - vendor/bigvgan/ (complete NVIDIA BigVGAN source)
    - vendor/bigvgan/VENDOR_PIN.txt
    - src/distill/vocoder/__init__.py
    - src/distill/vocoder/base.py
  modified:
    - pyproject.toml
    - uv.lock

key-decisions:
  - "Used VENDOR_PIN.txt with commit hash for version pinning (simple, no submodule complexity)"
  - "librosa added as full dependency (numba/llvmlite pulled in; Slaney filterbank extraction deferred unless problematic)"

patterns-established:
  - "Vendor pinning: commit hash in VENDOR_PIN.txt at vendor root"
  - "Vocoder interface: VocoderBase ABC with mel_to_waveform, sample_rate, to(device)"
  - "Lazy torch imports in vocoder base (TYPE_CHECKING guard + from __future__ import annotations)"

requirements-completed: [VOC-06]

# Metrics
duration: 2min
completed: 2026-02-22
---

# Phase 12 Plan 01: Vendor BigVGAN & Abstract Vocoder Interface Summary

**Vendored NVIDIA BigVGAN source with version pinning, added huggingface_hub and librosa dependencies, and created abstract VocoderBase interface with mel_to_waveform/sample_rate/to contract**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-22T00:20:46Z
- **Completed:** 2026-02-22T00:22:56Z
- **Tasks:** 2
- **Files modified:** 75

## Accomplishments
- Vendored complete NVIDIA/BigVGAN repository into vendor/bigvgan/ (71 files) with MIT license preserved
- Pinned to commit 7d2b454564a6c7d014227f635b7423881f14bdac via VENDOR_PIN.txt
- Added huggingface_hub>=0.20 and librosa>=0.10 to project dependencies (both install cleanly)
- Created abstract VocoderBase class with mel_to_waveform, sample_rate, and to(device) abstract methods
- Stubbed get_vocoder() factory function in vocoder package __init__.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Vendor BigVGAN source code** - `4cac7ef` (feat)
2. **Task 2: Add dependencies and create vocoder interface** - `f92a5b4` (feat)

## Files Created/Modified
- `vendor/bigvgan/` - Complete vendored NVIDIA BigVGAN source (bigvgan.py, meldataset.py, activations.py, configs/, etc.)
- `vendor/bigvgan/VENDOR_PIN.txt` - Commit hash for version pinning
- `vendor/bigvgan/LICENSE` - MIT license (preserved as-is)
- `src/distill/vocoder/__init__.py` - Vocoder package public API with get_vocoder() stub
- `src/distill/vocoder/base.py` - Abstract VocoderBase class
- `pyproject.toml` - Added huggingface_hub and librosa dependencies
- `uv.lock` - Updated lockfile

## Decisions Made
- Used VENDOR_PIN.txt with commit hash (7d2b454) rather than git submodule for version pinning -- simpler, no submodule complexity
- Added librosa as a full dependency rather than extracting Slaney filterbank manually -- installs cleanly with numba/llvmlite, can optimize later if needed
- Followed project lazy-torch-import pattern (TYPE_CHECKING guard) in VocoderBase for consistency with device.py

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Vendored BigVGAN source ready for wrapping in BigVGANVocoder implementation (Plan 02)
- VocoderBase interface ready for concrete implementations
- huggingface_hub available for weight download in weight_manager.py (Plan 02)
- librosa available for Slaney mel filterbank computation in mel_adapter.py (Plan 02/03)

## Self-Check: PASSED

All files verified present, all commits verified in git log.

---
*Phase: 12-vocoder-interface-bigvgan-integration*
*Completed: 2026-02-22*
