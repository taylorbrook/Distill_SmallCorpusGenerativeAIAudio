---
phase: 01-project-setup
plan: 02
subsystem: hardware
tags: [pytorch, mps, cuda, gpu, device-detection, memory, benchmark, psutil, rich]

# Dependency graph
requires:
  - phase: 01-01
    provides: "pip-installable package layout with hardware subpackage stub"
provides:
  - "Device detection with CUDA -> MPS -> CPU auto-detect and smoke test validation"
  - "select_device(), get_device_info(), format_device_report() in hardware/device.py"
  - "get_memory_info(), clear_gpu_memory(), safe_gpu_operation() in hardware/memory.py"
  - "benchmark_max_batch_size(), run_benchmark(), format_benchmark_report() in hardware/benchmark.py"
  - "Consistent memory info dict keys (allocated_gb, total_gb, free_gb) across all backends"
  - "OOM-safe operation wrapper with frame-reference-safe recovery pattern"
affects: [01-03, 03-training, 04-inference, 08-ui]

# Tech tracking
tech-stack:
  added: []
  patterns: [device smoke test before use, OOM flag pattern (recovery outside except block), binary-search batch sizing, lazy torch import for resilient module loading]

key-files:
  created:
    - src/small_dataset_audio/hardware/device.py
    - src/small_dataset_audio/hardware/memory.py
    - src/small_dataset_audio/hardware/benchmark.py
  modified: []

key-decisions:
  - "Explicit CUDA -> MPS -> CPU detection chain instead of torch.accelerator.current_accelerator() for predictability"
  - "Smoke test on all non-CPU devices to catch 'available but broken' GPUs (research pitfall #1)"
  - "OOM recovery code outside except block to release frame references (research pitfall #6)"
  - "CPU benchmark returns default 32 instead of running slow binary search"
  - "MPS memory reported via psutil (unified memory) plus torch.mps.current_allocated_memory()"

patterns-established:
  - "Device selection: select_device('auto') -> validated torch.device with smoke test"
  - "Memory querying: get_memory_info(device) -> dict with allocated_gb/total_gb/free_gb"
  - "OOM-safe wrapper: safe_gpu_operation(fn) -> result or None on OOM"
  - "GPU cleanup: clear_gpu_memory(device) with gc.collect sandwich"
  - "Lazy torch import: import inside function bodies for resilient module loading"

# Metrics
duration: 3min
completed: 2026-02-12
---

# Phase 1 Plan 2: Hardware Detection Summary

**MPS/CUDA/CPU device detection with smoke test validation, consistent memory querying via psutil, OOM-safe operation wrapper, and binary-search GPU batch size benchmark**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-12T22:32:11Z
- **Completed:** 2026-02-12T22:34:45Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Device detection auto-selects MPS on Apple Silicon, CUDA on NVIDIA, CPU as fallback -- with smoke test validation on all GPU backends
- Memory utilities provide consistent `allocated_gb/total_gb/free_gb` dict across CUDA, MPS, and CPU
- OOM-safe operation wrapper uses the frame-reference-safe flag pattern (recovery outside except block)
- Binary-search benchmark found max batch size of 256 on MPS (64 GB Apple Silicon) during verification
- All modules use lazy torch imports so they can be imported for introspection even if PyTorch has issues

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement device detection and memory utilities** - `6083f16` (feat)
2. **Task 2: Implement hardware benchmark** - `55eb5c9` (feat)

## Files Created/Modified
- `src/small_dataset_audio/hardware/device.py` - Device detection, selection, smoke test, info reporting, formatted report
- `src/small_dataset_audio/hardware/memory.py` - Memory querying, GPU cleanup, OOM-safe operation wrapper
- `src/small_dataset_audio/hardware/benchmark.py` - Binary-search batch size benchmark, formatted report, high-level runner

## Decisions Made
- Used explicit CUDA -> MPS -> CPU detection chain instead of `torch.accelerator.current_accelerator()` for predictability (plan specified this)
- Smoke test runs on all non-CPU devices to catch "available but broken" GPUs per research pitfall #1
- OOM recovery code placed outside except block to release Python frame references per research pitfall #6
- CPU benchmark skipped with default of 32 (binary search on CPU would be unreasonably slow)
- MPS memory reported via psutil system RAM (unified memory) plus `torch.mps.current_allocated_memory()` for PyTorch-specific tracking

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Hardware abstraction layer complete for Plan 01-03 (environment validation/startup)
- `select_device()` ready for app.py startup sequence
- `run_benchmark()` returns dict ready for config.toml storage
- `safe_gpu_operation()` ready for training and inference phases
- All three modules importable without error

## Self-Check: PASSED

All 3 created files verified present on disk. Both task commits (6083f16, 55eb5c9) verified in git history.

---
*Phase: 01-project-setup*
*Completed: 2026-02-12*
