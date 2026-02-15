---
phase: 10-multi-format-export-spatial-audio
plan: 02
subsystem: audio, inference
tags: [spatial-audio, hrtf, binaural, sofa, stereo, mid-side, scipy]

# Dependency graph
requires:
  - phase: 04-generation-pipeline-export
    provides: "stereo.py with mid-side widening, dual-seed combiner, and peak normalization"
provides:
  - "SpatialMode enum (stereo, binaural, mono) for output mode selection"
  - "SpatialConfig dataclass with width+depth spatial controls"
  - "apply_spatial dispatcher for mode-based spatial processing"
  - "HRTF loading from SOFA files with caching and resampling"
  - "Binaural convolution with width blending and depth rolloff"
  - "migrate_stereo_config for backward compatibility with Phase 4 presets"
affects: [10-04, ui-generate-tab, cli-generate]

# Tech tracking
tech-stack:
  added: [sofar, scipy.signal.fftconvolve, scipy.signal.butter]
  patterns: [spatial-mode-dispatch, hrtf-caching, depth-as-early-reflection, binaural-width-blend]

key-files:
  created:
    - src/small_dataset_audio/audio/hrtf.py
    - src/small_dataset_audio/inference/spatial.py
  modified:
    - src/small_dataset_audio/audio/__init__.py
    - src/small_dataset_audio/inference/__init__.py
    - pyproject.toml

key-decisions:
  - "sofar library for SOFA file loading (HRTF standard format)"
  - "1st-order Butterworth low-pass for depth rolloff (20kHz->8kHz proportional to depth)"
  - "Early reflection pattern for stereo depth (0-40ms delay, 0.15 mix level)"
  - "Width blends between center and full binaural (not discrete steps)"
  - "migrate_stereo_config maps mid_side->STEREO with depth=0.3 default"

patterns-established:
  - "SpatialMode enum as str,Enum for serialization compatibility"
  - "SpatialConfig.validate() for range checking before processing"
  - "HRTF cache keyed by (sofa_path, sample_rate) for reuse across calls"
  - "Depth effect implementation differs per mode (early reflection for stereo, frequency rolloff for binaural)"

# Metrics
duration: 4min
completed: 2026-02-14
---

# Phase 10 Plan 02: Spatial Audio System Summary

**Spatial audio module with stereo/binaural/mono modes, HRTF-based binaural rendering via SOFA files, and two-dimensional width+depth controls replacing Phase 4 stereo_width**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-15T02:06:56Z
- **Completed:** 2026-02-15T02:11:04Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- HRTF module for loading SOFA files with caching, nearest-position lookup, and optional resampling via torchaudio
- Binaural convolution using scipy.signal.fftconvolve with width blending and depth-based high-frequency rolloff
- SpatialMode enum and SpatialConfig dataclass providing a clean API for stereo/binaural/mono selection
- apply_spatial dispatcher that routes to the correct processing chain per mode
- Backward compatibility via migrate_stereo_config mapping old Phase 4 stereo_mode/stereo_width to new SpatialConfig

## Task Commits

Each task was committed atomically:

1. **Task 1: Create hrtf.py for SOFA-based HRTF loading and binaural convolution** - `9981164` (feat)
2. **Task 2: Create spatial.py replacing stereo system with stereo/binaural/mono modes** - `d753ea7` (feat)

## Files Created/Modified
- `src/small_dataset_audio/audio/hrtf.py` - HRTF loading from SOFA files, binaural convolution with width/depth controls, caching
- `src/small_dataset_audio/inference/spatial.py` - SpatialMode enum, SpatialConfig dataclass, apply_spatial dispatcher, migration helper
- `src/small_dataset_audio/audio/__init__.py` - Re-exports HRTFData, load_hrtf, apply_binaural, clear_hrtf_cache
- `src/small_dataset_audio/inference/__init__.py` - Re-exports SpatialMode, SpatialConfig, apply_spatial, migrate_stereo_config
- `pyproject.toml` - sofar>=1.1 dependency (already present from prior plan)

## Decisions Made
- Used sofar library for SOFA file loading (standard format for HRTF measurement data)
- 1st-order Butterworth low-pass for binaural depth rolloff, scaling cutoff from 20kHz (close) to 8kHz (far)
- Early reflection pattern for stereo depth effect: delayed copy mixed at depth*0.15 with 0-40ms delay
- Width control blends linearly between center (mono sum) and full binaural separation
- migrate_stereo_config maps old "mid_side" to STEREO with depth=0.3, "dual_seed" to STEREO with width=0.7/depth=0.3

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
The MIT KEMAR SOFA file needs to be downloaded and placed at `src/small_dataset_audio/data/hrtf/mit_kemar.sofa` for binaural mode to function. Download from https://sofacoustics.org/data/database/mit/ (file: MIT_KEMAR_normal_pinna.sofa, ~1.4 MB). Without this file, binaural mode raises a clear FileNotFoundError with download instructions.

## Next Phase Readiness
- Spatial audio system complete and ready for integration into GenerationConfig (Plan 04)
- stereo.py preserved -- its utility functions (peak_normalize, create_dual_seed_stereo, apply_mid_side_widening) are still used by spatial.py
- HRTF file download is documented but not automated -- users need to place the SOFA file manually

## Self-Check: PASSED

All files exist. All commits verified.

---
*Phase: 10-multi-format-export-spatial-audio*
*Completed: 2026-02-14*
