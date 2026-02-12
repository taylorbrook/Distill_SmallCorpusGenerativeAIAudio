# Phase 1: Project Setup - Research

**Researched:** 2026-02-12
**Domain:** Python project scaffolding, PyTorch hardware abstraction (MPS/CUDA/CPU), dependency management, environment validation
**Confidence:** HIGH

## Summary

Phase 1 establishes the development foundation: a pip-installable Python package with PyTorch 2.10.0 and TorchAudio 2.10.0, hardware abstraction across MPS/CUDA/CPU, a guided first-run experience, and a single-command setup workflow. No training, generation, or UI lives here -- just the scaffolding everything else builds on.

PyTorch 2.10.0 (released January 21, 2026) provides a mature unified device API via `torch.accelerator` that simplifies hardware detection across MPS, CUDA, and CPU backends. For dependency management, **uv** is the clear winner for this project -- it is 10-100x faster than pip, manages virtual environments automatically, handles PyTorch's multi-index wheel distribution natively, and uses a global cache to avoid re-downloading large ML packages. Configuration should use a single TOML file (`config.toml`) read by the stdlib `tomllib` and written by `tomli-w`, keeping complexity low while being human-readable and consistent with `pyproject.toml`.

**Primary recommendation:** Use uv + pyproject.toml (src layout) + TOML config + Makefile. Hardware detection via `torch.accelerator` with `torch.backends.mps`/`torch.cuda` for detailed memory queries. Binary-search benchmark on first launch to estimate max batch size.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Package-based layout with nested packages (e.g., src/training/, src/audio/, src/models/) for clear domain separation
- Data directories (generated audio, trained models, datasets) default inside project but are user-configurable via config
- Modern Python packaging with pyproject.toml -- installable via pip with proper entry points
- Always display selected device (MPS/CUDA/CPU) at startup -- users always know what's running
- Run a quick hardware benchmark on first launch to estimate training capacity (max batch size, available memory)
- Graceful fallback to CPU when GPU unavailable
- Single setup command to get a new contributor running (e.g., make setup or ./setup.sh)
- Report and exit on missing/incompatible critical dependencies -- clear error message with fix instructions, don't touch the environment
- Guided first-run experience: walk user through initial config (paths, device check, create directories)
- Full environment validation on every launch (check deps, device, paths) -- catches drift early
- Report and exit on critical failures -- clear message with fix instructions, no auto-fixing

### Claude's Discretion
- Configuration approach (single file vs layered -- pick what fits best)
- Virtual environment tool choice (pick best fit for usability and capability)
- PyTorch version pinning strategy (exact vs minimum)
- Whether to include a Makefile or task runner for common operations
- Startup output verbosity (essentials vs detailed)
- OOM recovery strategy (fail with guidance vs auto-fallback to CPU)
- Whether to support --device flag for manual device override

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | Deep learning framework, device abstraction | Released Jan 21 2026. Mature MPS support, unified `torch.accelerator` API, TorchScript deprecated in favor of `torch.export`. 4,160 commits from 536 contributors since 2.9. |
| TorchAudio | 2.10.0 | Audio I/O and transforms | Always matches PyTorch version. Now in maintenance phase -- focused on core audio processing for ML. Decoding/encoding consolidated into TorchCodec. |
| tomllib | stdlib (3.11+) | Read TOML config files | Zero-dependency TOML reading. Part of Python stdlib since 3.11. Reads `config.toml` for application settings. |
| tomli-w | 1.1.0+ | Write TOML config files | Companion to `tomllib` for writing config. Lightweight (pure Python, no deps). Needed because `tomllib` is read-only. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| uv | 0.7+ | Package/environment manager | All dependency installation and virtual environment management. 10-100x faster than pip. Native PyTorch multi-index support. |
| rich | 14.0+ | Terminal output formatting | Startup messages, device info display, first-run guided experience, progress indicators. Much better UX than plain print. |
| psutil | 6.0+ | System info detection | Query system RAM, CPU count for hardware benchmark reporting. Cross-platform. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| uv | poetry | Poetry is more mature for publishing but slower (10-100x), no native PyTorch index handling, heavier. uv wins on speed and PyTorch compatibility. |
| uv | conda | Conda handles binary deps well but is slow, has solver issues, separate ecosystem. Not needed since PyTorch wheels are self-contained now. |
| uv | pip + venv | Works but manual, no lockfile, no auto-sync, no global cache. uv is a strict superset. |
| TOML config | Hydra | Hydra is powerful for experiment configs but heavy for Phase 1 application config. Can add Hydra for training configs in Phase 3. |
| TOML config | YAML + PyYAML | YAML is ambiguous (Norway problem, implicit typing). TOML forces explicit types. Python has stdlib TOML reader. |
| Makefile | just | `just` is cleaner syntax but not pre-installed. Make is universal on macOS/Linux. |
| rich | plain print | Functional but poor UX for guided first-run, device reporting, and error messages. rich is low-cost, high-impact. |

**Installation:**
```bash
# One-command setup (runs uv sync internally)
make setup

# Or manually:
uv sync
```

## Architecture Patterns

### Recommended Project Structure
```
small-dataset-audio/
├── pyproject.toml            # Package metadata, dependencies, entry points
├── uv.lock                   # Lockfile (auto-generated by uv)
├── Makefile                  # Common operations (setup, test, lint, run)
├── config.toml               # User configuration (created on first run)
├── .python-version           # Python version pin (auto-created by uv)
├── src/
│   └── small_dataset_audio/  # Main package (importable)
│       ├── __init__.py       # Package init, version
│       ├── __main__.py       # Entry point: python -m small_dataset_audio
│       ├── app.py            # Application bootstrap (startup, validation, routing)
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py   # Config loading/saving (TOML), defaults, validation
│       │   └── defaults.py   # Default configuration values
│       ├── hardware/
│       │   ├── __init__.py
│       │   ├── device.py     # Device detection (MPS/CUDA/CPU), selection, display
│       │   ├── benchmark.py  # First-launch GPU benchmark (batch size estimation)
│       │   └── memory.py     # Memory querying, OOM handling utilities
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── environment.py  # Dep checks, version verification, path validation
│       │   └── startup.py      # Full startup validation sequence
│       ├── audio/             # Placeholder for Phase 2+
│       │   └── __init__.py
│       ├── models/            # Placeholder for Phase 3+
│       │   └── __init__.py
│       ├── training/          # Placeholder for Phase 3+
│       │   └── __init__.py
│       ├── inference/         # Placeholder for Phase 4+
│       │   └── __init__.py
│       └── ui/                # Placeholder for Phase 8+
│           └── __init__.py
├── data/                      # Default data root (user-configurable)
│   ├── datasets/              # User audio datasets
│   ├── models/                # Trained model checkpoints
│   └── generated/             # Generated audio output
├── tests/
│   ├── __init__.py
│   ├── test_device.py
│   ├── test_config.py
│   ├── test_benchmark.py
│   └── test_validation.py
└── .gitignore
```

### Pattern 1: Unified Device Abstraction with torch.accelerator

**What:** Use PyTorch 2.10's `torch.accelerator` as the primary device detection API, with backend-specific fallbacks for memory queries.

**When to use:** Every time the application needs to select or query a compute device.

**Example:**
```python
# Source: PyTorch 2.10 docs - torch.accelerator
import torch

def detect_device(override: str | None = None) -> torch.device:
    """Detect best available device. Supports --device override."""
    if override:
        device = torch.device(override)
        if not _validate_device(device):
            raise SystemExit(f"Requested device '{override}' is not available.")
        return device

    # Use unified accelerator API (PyTorch 2.10+)
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()

    return torch.device("cpu")

def _validate_device(device: torch.device) -> bool:
    """Check if a specific device is actually available."""
    if device.type == "mps":
        return torch.backends.mps.is_available()
    elif device.type == "cuda":
        return torch.cuda.is_available()
    elif device.type == "cpu":
        return True
    return False

def get_device_info(device: torch.device) -> dict:
    """Query detailed info about the selected device."""
    info = {"type": device.type, "name": str(device)}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        info["capability"] = torch.cuda.get_device_capability(device)
        free, total = torch.cuda.mem_get_info(device)
        info["memory_total_gb"] = total / (1024**3)
        info["memory_free_gb"] = free / (1024**3)
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"
        # MPS uses unified memory -- report system RAM as proxy
        import psutil
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = mem.total / (1024**3)
        info["memory_free_gb"] = mem.available / (1024**3)
        # Also track MPS-specific allocation
        info["mps_allocated_gb"] = torch.mps.current_allocated_memory() / (1024**3)
    elif device.type == "cpu":
        import psutil
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = mem.total / (1024**3)

    return info
```

### Pattern 2: TOML Configuration with Defaults and First-Run Creation

**What:** Single `config.toml` file with sensible defaults. Created interactively on first run. Read with `tomllib`, written with `tomli-w`.

**When to use:** All application configuration (paths, device preferences, benchmark results).

**Example:**
```python
# Source: Python 3.11+ tomllib stdlib + tomli-w
import tomllib
from pathlib import Path

DEFAULT_CONFIG = {
    "general": {
        "project_name": "Small Dataset Audio",
        "first_run_complete": False,
    },
    "paths": {
        "datasets": "data/datasets",
        "models": "data/models",
        "generated": "data/generated",
    },
    "hardware": {
        "device": "auto",          # "auto", "mps", "cuda", "cpu"
        "max_batch_size": None,    # Set by benchmark
        "memory_limit_gb": None,   # Set by benchmark
    },
}

def load_config(config_path: Path) -> dict:
    """Load config from TOML, or return defaults if not found."""
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config: dict, config_path: Path) -> None:
    """Save config to TOML file."""
    import tomli_w
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)
```

### Pattern 3: Binary-Search Batch Size Benchmark

**What:** On first launch, run a quick GPU benchmark to find the maximum batch size that fits in memory. Uses binary search with a dummy tensor operation.

**When to use:** First-run setup to establish intelligent defaults for training.

**Example:**
```python
# Source: Community pattern from PyTorch Forums + TDS article
import torch
import gc

def benchmark_max_batch_size(
    device: torch.device,
    sample_shape: tuple = (1, 48000),  # 1 second of 48kHz mono
    min_batch: int = 1,
    max_batch: int = 256,
) -> int:
    """Binary search for maximum batch size that fits in GPU memory."""
    working_batch = min_batch
    low, high = min_batch, max_batch

    while low <= high:
        mid = (low + high) // 2
        try:
            # Simulate a forward pass with dummy data
            x = torch.randn(mid, *sample_shape, device=device)
            # Simulate some compute (matmul-like operation)
            _ = torch.nn.functional.conv1d(
                x.unsqueeze(1),
                torch.randn(16, 1, 1024, device=device),
                padding=512,
            )
            # If we got here, this batch size works
            working_batch = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
            else:
                raise
        finally:
            # Clean up GPU memory
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()

    return working_batch
```

### Pattern 4: Environment Validation on Startup

**What:** Validate all critical dependencies, device availability, and directory structure on every launch. Report failures with actionable fix instructions.

**When to use:** Every application launch.

**Example:**
```python
# Validation checks run on every startup
def validate_environment() -> list[str]:
    """Run all validation checks. Returns list of error messages (empty = OK)."""
    errors = []

    # 1. Check Python version
    import sys
    if sys.version_info < (3, 11):
        errors.append(
            f"Python 3.11+ required (found {sys.version}). "
            "Install with: uv python install 3.11"
        )

    # 2. Check PyTorch
    try:
        import torch
        if torch.__version__ < "2.10.0":
            errors.append(
                f"PyTorch 2.10.0+ required (found {torch.__version__}). "
                "Run: uv sync"
            )
    except ImportError:
        errors.append("PyTorch not installed. Run: uv sync")

    # 3. Check TorchAudio
    try:
        import torchaudio
    except ImportError:
        errors.append("TorchAudio not installed. Run: uv sync")

    # 4. Check data directories exist
    from pathlib import Path
    config = load_config(Path("config.toml"))
    for key in ["datasets", "models", "generated"]:
        path = Path(config["paths"][key])
        if not path.exists():
            errors.append(f"Directory missing: {path}. Will be created on first run.")

    return errors
```

### Anti-Patterns to Avoid

- **Hard-coding device strings:** Never write `device = "cuda"` or `device = "mps"`. Always detect at runtime via `torch.accelerator` or the device abstraction module.
- **Silently falling back without reporting:** Users must always know what device they are running on. Never silently degrade to CPU.
- **Auto-fixing the environment:** If a dependency is wrong, report the error with fix instructions and exit. Do not run `pip install` or modify the user's environment.
- **Importing torch at module level in config code:** Keep config loading independent of PyTorch so config errors can be reported even if PyTorch is broken.
- **Storing absolute paths in config:** Use relative paths or expanduser paths so configs are portable between machines.

## Discretion Recommendations

### Configuration Approach: Single TOML File
**Recommendation:** Use a single `config.toml` file at the project root.
**Rationale:** Phase 1 needs only paths, device preference, and benchmark results. A single file is simpler, human-editable, and uses Python's stdlib `tomllib` for reading. Layered configs (user + project + defaults) add complexity with no benefit at this stage. If Phase 3+ needs experiment-level config, Hydra can be introduced alongside this base config without conflict.
**Confidence:** HIGH

### Virtual Environment Tool: uv
**Recommendation:** Use **uv** (by Astral, makers of Ruff).
**Rationale:** User wants "ease of use prioritized." uv is 10-100x faster than pip, auto-creates virtual environments, auto-syncs from `pyproject.toml`, has a global package cache (saves gigabytes for PyTorch across projects), and has **dedicated PyTorch integration** with `--torch-backend=auto` and `[tool.uv.sources]` for multi-platform index routing. `uv sync` is the single setup command. Cross-platform (macOS, Linux, Windows). Rust-based, single binary install.
**Confidence:** HIGH

### PyTorch Version Pinning: Minimum with Upper Bound
**Recommendation:** Pin as `torch>=2.10.0,<2.11` and `torchaudio>=2.10.0,<2.11`.
**Rationale:** Exact pinning (`==2.10.0`) breaks when patch versions come out. Pure minimum (`>=2.10.0`) risks breaking changes in 2.11. A minimum with upper bound gets patches automatically while avoiding surprise breaking changes. The lockfile (`uv.lock`) pins the exact resolved version for reproducibility.
**Confidence:** HIGH

### Task Runner: Makefile
**Recommendation:** Include a Makefile with common targets.
**Rationale:** Make is pre-installed on macOS and Linux (the target platforms). Provides a discoverable interface (`make help` lists all targets). Standard targets: `setup`, `run`, `test`, `lint`, `format`, `clean`, `benchmark`. Lower friction than remembering `uv run python -m small_dataset_audio`.
**Confidence:** HIGH

### Startup Output Verbosity: Essentials with --verbose Flag
**Recommendation:** Show essentials by default (device selected, any warnings), support `--verbose` for detailed output.
**Rationale:** Musicians want to get to work, not read logs. Default output should be: device name, memory available, any warnings. Verbose mode adds: Python version, PyTorch version, all dependency versions, detailed memory breakdown, benchmark results.
**Confidence:** HIGH

### OOM Recovery Strategy: Fail with Guidance
**Recommendation:** Catch OOM, clear GPU memory, report what happened and what to do, do NOT auto-fallback to CPU.
**Rationale:** Auto-fallback to CPU during training would silently make training 10-100x slower. Users should make that decision intentionally. The guidance message should say: "GPU out of memory. Try: (1) reduce batch size to N, (2) use --device cpu for CPU training, (3) close other GPU applications." The benchmark results from first-run help prevent OOM in the first place.
**Confidence:** HIGH

### --device Flag: Yes, Support It
**Recommendation:** Support `--device [auto|mps|cuda|cpu]` as a CLI argument.
**Rationale:** Essential for debugging ("works on CPU but fails on MPS"), testing ("verify CUDA path on a different machine"), and user choice ("I want CPU despite having a GPU"). Low implementation cost, high utility. Default is "auto" (use config value, which defaults to auto-detect).
**Confidence:** HIGH

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Virtual environment management | Custom venv scripts | uv (auto-creates, auto-syncs) | Edge cases with Python versions, platform differences, activation scripts |
| Terminal formatting/colors | ANSI escape codes | rich library | Cross-platform color support, table formatting, progress bars, markdown rendering |
| System memory detection | `/proc/meminfo` parsing | psutil | Cross-platform (macOS/Linux/Windows), handles unified memory correctly |
| TOML parsing | Custom parser or regex | tomllib (stdlib) | TOML spec is deceptively complex (multiline strings, datetime, nested tables) |
| TOML writing | String concatenation | tomli-w | Proper escaping, type handling, comment preservation |
| PyTorch device selection | if/elif chains with string matching | torch.accelerator API | Handles new backends (XPU, HPU) automatically, maintained by PyTorch team |
| Dependency version checking | subprocess + pip list | importlib.metadata | stdlib, fast, no subprocess overhead |

**Key insight:** Phase 1 is foundation code that every subsequent phase depends on. Hand-rolling anything here creates technical debt that compounds across all 10 phases.

## Common Pitfalls

### Pitfall 1: MPS "Available but Broken" on macOS Updates
**What goes wrong:** `torch.backends.mps.is_available()` returns True but operations fail at runtime. Reported on macOS 26 (Tahoe) with PyTorch 2.10 nightly builds.
**Why it happens:** macOS updates can change Metal framework behavior. PyTorch's MPS backend is built but the runtime Metal version may be incompatible.
**How to avoid:** After detecting MPS, run a small smoke test (create tensor, do a multiply, check result). If smoke test fails, fall back to CPU with a warning explaining the issue.
**Warning signs:** `MPS built?: True` but `MPS available?: False`. Or MPS available but simple operations throw errors.

### Pitfall 2: PYTORCH_ENABLE_MPS_FALLBACK Not Working for All Ops
**What goes wrong:** Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` is supposed to fall back to CPU for unsupported MPS operations, but it does not work for all operations (e.g., some `nn.Conv1d` edge cases).
**Why it happens:** The fallback mechanism has known gaps in coverage. Some operations throw NotImplementedError instead of using the fallback path.
**How to avoid:** Do not rely solely on `PYTORCH_ENABLE_MPS_FALLBACK`. Instead, set it as a safety net but also catch RuntimeError in critical paths and handle gracefully. Test all operations used in the project on MPS specifically.
**Warning signs:** Operations that work on CUDA/CPU but throw NotImplementedError on MPS even with fallback enabled.

### Pitfall 3: PyTorch Multi-Index Installation Confusion
**What goes wrong:** Installing PyTorch from PyPI gets CPU-only wheels on macOS/Windows, but users expect GPU support. Or mixing PyPI and pytorch.org wheels causes version conflicts.
**Why it happens:** PyTorch distributes GPU wheels on a separate index (`download.pytorch.org/whl/cu128`). PyPI only has CPU wheels for Windows and macOS. PyPI has GPU wheels only for Linux (CUDA 12.8).
**How to avoid:** Use uv with `[tool.uv.sources]` to configure platform-specific indexes in `pyproject.toml`. macOS does NOT need a separate index (MPS support is in the default wheels). Only Linux/Windows CUDA users need the pytorch-cu128 index.
**Warning signs:** `torch.cuda.is_available()` returns False on a CUDA machine, or PyTorch is unexpectedly large/small download.

### Pitfall 4: TorchAudio API Removals in 2.9+
**What goes wrong:** Code examples and tutorials reference TorchAudio functions that were removed in version 2.9 (deprecated in 2.8).
**Why it happens:** TorchAudio transitioned to maintenance phase. Decoding/encoding consolidated into TorchCodec. Many deprecated APIs were removed.
**How to avoid:** Only use TorchAudio APIs documented in the 2.10 stable docs. For audio file I/O, prefer `torchaudio.load()` and `torchaudio.save()` which remain. For advanced codec features, check TorchCodec.
**Warning signs:** ImportError or AttributeError when using TorchAudio functions from older tutorials.

### Pitfall 5: MPS Memory Leaks in Long-Running Applications
**What goes wrong:** Memory usage increases continuously on MPS during repeated operations. Application eventually crashes or system becomes unresponsive.
**Why it happens:** Reported memory leaks in PyTorch MPS backend (GitHub issue #154329, #77753). MPS allocator does not always release memory promptly. Unified memory means GPU memory pressure affects the entire system.
**How to avoid:** Call `torch.mps.empty_cache()` after intensive operations. Implement periodic garbage collection (`gc.collect()`). Monitor `torch.mps.current_allocated_memory()` and warn if it grows unexpectedly. For the benchmark, clean up thoroughly between iterations.
**Warning signs:** Increasing memory in Activity Monitor > GPU History. Application slows down over time. System starts swapping.

### Pitfall 6: OOM Exception Reference Holding
**What goes wrong:** Catching a CUDA/MPS OOM error in a try/except block fails to free the memory because the Python exception object holds a reference to the stack frame where the error occurred.
**Why it happens:** Python exception handling retains references to local variables in the frame that raised the exception. The tensor that caused OOM stays alive inside the exception reference chain.
**How to avoid:** Move recovery code OUTSIDE the except block. Set a flag in the except block, then handle cleanup after the try/except exits. This allows the frame reference to be released.
**Warning signs:** `torch.cuda.empty_cache()` inside except block does not actually free memory. OOM recovery fails to recover.

## Code Examples

### pyproject.toml (Complete)
```toml
# Source: uv PyTorch integration docs + Python Packaging User Guide
[project]
name = "small-dataset-audio"
version = "0.1.0"
description = "Generative audio from small personal datasets"
requires-python = ">=3.11,<3.15"
license = "MIT"
dependencies = [
    "torch>=2.10.0,<2.11",
    "torchaudio>=2.10.0,<2.11",
    "tomli-w>=1.1.0",
    "rich>=14.0",
    "psutil>=6.0",
]

[project.scripts]
sda = "small_dataset_audio.app:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/small_dataset_audio"]

# --- uv PyTorch index configuration ---
# macOS: PyPI wheels include MPS support (no special index needed)
# Linux CUDA: Use pytorch-cu128 index
# CPU fallback: PyPI default works
[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
]
torchaudio = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "ruff>=0.9",
]
```

### Makefile (Complete)
```makefile
# Source: Community best practice for Python ML projects
.PHONY: help setup run test lint format clean benchmark

PYTHON := uv run python
MODULE := small_dataset_audio

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install all dependencies and set up environment
	uv sync
	@echo "Setup complete. Run 'make run' to start."

run: ## Run the application
	$(PYTHON) -m $(MODULE)

run-verbose: ## Run with verbose output
	$(PYTHON) -m $(MODULE) --verbose

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v

lint: ## Run linter
	uv run ruff check src/ tests/

format: ## Auto-format code
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

benchmark: ## Run hardware benchmark
	$(PYTHON) -m $(MODULE) --benchmark
```

### Device Detection with Smoke Test
```python
# Source: PyTorch 2.10 docs (torch.accelerator, torch.backends.mps)
import torch

def select_device(preference: str = "auto") -> torch.device:
    """
    Select compute device with validation.

    Args:
        preference: "auto", "mps", "cuda", "cpu", or "cuda:N"

    Returns:
        Validated torch.device
    """
    if preference == "auto":
        device = _auto_detect()
    else:
        device = torch.device(preference)

    # Smoke test: verify device actually works
    if device.type != "cpu":
        if not _smoke_test(device):
            print(f"WARNING: {device} detected but smoke test failed. "
                  f"Falling back to CPU.")
            device = torch.device("cpu")

    return device

def _auto_detect() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _smoke_test(device: torch.device) -> bool:
    """Run a quick test to verify device works."""
    try:
        a = torch.randn(32, 32, device=device)
        b = torch.randn(32, 32, device=device)
        c = torch.matmul(a, b)
        result = c.sum().item()  # Force computation + transfer to CPU
        # Verify not NaN
        return result == result  # NaN != NaN
    except Exception:
        return False
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
```

### OOM Handler (Exception Reference Safe)
```python
# Source: PyTorch Forums - OOM recovery best practices
import gc
import torch

def safe_gpu_operation(fn, *args, device=None, **kwargs):
    """
    Run a GPU operation with OOM protection.
    Moves recovery code OUTSIDE except block to release frame references.
    """
    oom_occurred = False
    result = None

    try:
        result = fn(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom_occurred = True
        else:
            raise

    # Recovery OUTSIDE except block -- allows frame references to be freed
    if oom_occurred:
        gc.collect()
        if device and device.type == "cuda":
            torch.cuda.empty_cache()
        elif device and device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
        return None  # Caller checks for None and handles

    return result
```

### First-Run Guided Experience
```python
# Source: Design pattern for CLI first-run setup
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()

def first_run_setup(config_path: Path) -> dict:
    """Interactive first-run configuration."""
    console.print("\n[bold]Welcome to Small Dataset Audio![/bold]\n")
    console.print("Let's set up your environment.\n")

    config = DEFAULT_CONFIG.copy()

    # 1. Paths
    console.print("[bold]Data Directories[/bold]")
    for key, default in config["paths"].items():
        path = Prompt.ask(
            f"  {key} directory",
            default=default,
        )
        config["paths"][key] = path
        Path(path).mkdir(parents=True, exist_ok=True)

    # 2. Device detection
    console.print("\n[bold]Hardware Detection[/bold]")
    device = select_device("auto")
    info = get_device_info(device)
    console.print(f"  Device: [green]{info['name']}[/green]")
    if "memory_total_gb" in info:
        console.print(f"  Memory: {info['memory_total_gb']:.1f} GB total")

    # 3. Benchmark
    if device.type != "cpu":
        if Confirm.ask("\n  Run hardware benchmark? (estimates max batch size)"):
            max_batch = benchmark_max_batch_size(device)
            config["hardware"]["max_batch_size"] = max_batch
            console.print(f"  Max batch size: [green]{max_batch}[/green]")

    config["hardware"]["device"] = "auto"
    config["general"]["first_run_complete"] = True

    save_config(config, config_path)
    console.print(f"\n[green]Configuration saved to {config_path}[/green]\n")

    return config
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.device("cuda" if torch.cuda.is_available() else "cpu")` | `torch.accelerator.current_accelerator()` | PyTorch 2.8+ | Automatically handles MPS, CUDA, XPU without if/elif chains |
| TorchScript for model export | `torch.export` | PyTorch 2.10 (TorchScript deprecated) | Use `torch.export` for all new model serialization |
| pip + requirements.txt | uv + pyproject.toml + uv.lock | 2024-2025 | Faster, reproducible, lockfile-based, global cache |
| setup.py / setup.cfg | pyproject.toml (PEP 621) | 2023+ | Single configuration file, standardized metadata |
| TorchAudio full-featured codec | TorchAudio maintenance + TorchCodec | TorchAudio 2.9 | Use TorchAudio for transforms, TorchCodec for advanced codec |
| Manual PyTorch index URLs | uv `[tool.uv.sources]` with platform markers | uv 0.5.3+ (2025) | Automatic platform-correct PyTorch wheels |
| `torch.has_mps` (old) | `torch.backends.mps.is_available()` | PyTorch 2.0+ | Proper API, checks both build and runtime availability |

**Deprecated/outdated:**
- **TorchScript:** Deprecated in PyTorch 2.10. Use `torch.export` instead.
- **setup.py/setup.cfg:** Replaced by pyproject.toml for new projects.
- **pip + requirements.txt:** Replaced by uv + pyproject.toml for reproducible environments.
- **`torch.has_mps`:** Old attribute. Use `torch.backends.mps.is_available()`.
- **TorchAudio deprecated APIs:** Many functions removed in 2.9. Check 2.10 stable docs only.

## Open Questions

1. **macOS 26 (Tahoe) MPS Compatibility**
   - What we know: A GitHub issue (#167679) reports MPS "built but not available" on macOS 26 with PyTorch 2.10 nightly. This may be a nightly-specific issue resolved in the stable release.
   - What's unclear: Whether the stable 2.10.0 release has this issue on macOS 26.
   - Recommendation: The smoke test pattern handles this gracefully. Test on the target macOS version early. If MPS fails, CPU fallback works.

2. **uv Wheel Variants for Automatic GPU Detection**
   - What we know: uv has experimental `--torch-backend=auto` in `uv pip` but not yet in project mode (`uv sync`). An experimental variant-enabled build exists.
   - What's unclear: When automatic GPU detection will land in `uv sync` (project mode).
   - Recommendation: Use `[tool.uv.sources]` with platform markers for now. macOS MPS works from PyPI default wheels. Linux CUDA needs the cu128 index. This covers 100% of the use cases.

3. **MPS Memory Query Completeness**
   - What we know: `torch.mps.current_allocated_memory()` exists but does not include cached allocations. There is no MPS equivalent of `torch.cuda.mem_get_info()` (free/total).
   - What's unclear: Whether a more complete MPS memory API will be added.
   - Recommendation: Use `psutil.virtual_memory()` for total/available system memory on MPS (unified memory), and `torch.mps.current_allocated_memory()` for tracking PyTorch-specific allocations.

## Sources

### Primary (HIGH confidence)
- [PyTorch 2.10 Release Blog](https://pytorch.org/blog/pytorch-2-10-release-blog/) - Release features, deprecations, 2026 release cadence
- [PyTorch 2.10 torch.accelerator docs](https://docs.pytorch.org/docs/stable/accelerator.html) - Unified device API
- [PyTorch 2.10 torch.accelerator device management](https://docs.pytorch.org/docs/stable/accelerator/device.html) - Device detection functions
- [PyTorch 2.10 MPS Backend docs](https://docs.pytorch.org/docs/stable/notes/mps.html) - MPS usage, availability checks, code examples
- [PyTorch 2.10 torch.mps docs](https://docs.pytorch.org/docs/stable/mps.html) - MPS memory functions
- [PyTorch 2.10 torch.cuda.mem_get_info](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.mem_get_info.html) - CUDA memory query
- [uv PyTorch Integration Guide](https://docs.astral.sh/uv/guides/integration/pytorch/) - Complete pyproject.toml examples for all platforms
- [uv Project Documentation](https://docs.astral.sh/uv/guides/projects/) - Project setup, sync, lockfiles
- [Python Packaging User Guide - pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) - Modern Python packaging
- [Python tomllib docs](https://docs.python.org/3/library/tomllib.html) - TOML reading (stdlib)
- [tomli-w on PyPI](https://pypi.org/project/tomli-w/) - TOML writing library

### Secondary (MEDIUM confidence)
- [PyTorch 2.10 GA announcement](https://dev-discuss.pytorch.org/t/pytorch-2-10-0-general-availability/3291) - Release confirmation
- [PyTorch MPS environment variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html) - PYTORCH_ENABLE_MPS_FALLBACK and others
- [PyTorch OOM Recovery Forum Thread](https://discuss.pytorch.org/t/recover-from-cuda-out-of-memory/29051) - Exception reference holding pattern
- [PyTorch Batch Size Auto-Finding (TDS)](https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1/) - Binary search batch size pattern
- [uv vs Poetry comparison (Multiple)](https://envelope.dev/blog/poetry-vs-uv-vs-pip-choosing-the-right-package-installer) - Package manager comparison
- [Fastest Way to Install PyTorch Using uv (2026 Guide)](https://pratikpathak.com/fastest-way-to-install-pytorch-using-uv/) - uv + PyTorch setup
- [Real Python - pyproject.toml Guide](https://realpython.com/python-pyproject-toml/) - pyproject.toml best practices
- [KDnuggets - Makefiles in Python Projects](https://www.kdnuggets.com/the-case-for-makefiles-in-python-projects-and-how-to-get-started) - Makefile patterns

### Tertiary (LOW confidence)
- [MPS built but not available on macOS 26 - Issue #167679](https://github.com/pytorch/pytorch/issues/167679) - May be nightly-only issue, needs verification against stable release
- [MPS Memory Leak - Issue #154329](https://github.com/pytorch/pytorch/issues/154329) - Memory leak reports, unclear if fixed in 2.10.0
- [PYTORCH_ENABLE_MPS_FALLBACK not working for Conv1d - Issue #134416](https://github.com/pytorch/pytorch/issues/134416) - Fallback gaps, unclear current status

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch 2.10.0 verified from official release blog and docs. uv verified from official docs with dedicated PyTorch guide. tomllib is stdlib.
- Architecture: HIGH - Project structure follows Python Packaging User Guide (src layout). Device patterns from PyTorch 2.10 official docs.
- Pitfalls: MEDIUM-HIGH - MPS issues documented in GitHub issues (some may be resolved). OOM pattern from PyTorch forums. PyTorch index issues from uv docs.
- Discretion recommendations: HIGH - Each recommendation backed by official docs and community consensus.

**Research date:** 2026-02-12
**Valid until:** 2026-03-14 (30 days - stable ecosystem, PyTorch on 2-month release cadence)
