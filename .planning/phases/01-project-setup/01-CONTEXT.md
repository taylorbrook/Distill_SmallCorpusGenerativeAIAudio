# Phase 1: Project Setup - Context

**Gathered:** 2026-02-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish development environment with PyTorch, hardware abstraction (MPS/CUDA/CPU), and foundational project structure. No training, generation, or UI — just the foundation everything else builds on.

</domain>

<decisions>
## Implementation Decisions

### Project structure
- Package-based layout with nested packages (e.g., src/training/, src/audio/, src/models/) for clear domain separation
- Data directories (generated audio, trained models, datasets) default inside project but are user-configurable via config
- Modern Python packaging with pyproject.toml — installable via pip with proper entry points

### Hardware detection
- Always display selected device (MPS/CUDA/CPU) at startup — users always know what's running
- Run a quick hardware benchmark on first launch to estimate training capacity (max batch size, available memory)
- Graceful fallback to CPU when GPU unavailable

### Dependency management
- Single setup command to get a new contributor running (e.g., make setup or ./setup.sh)
- Report and exit on missing/incompatible critical dependencies — clear error message with fix instructions, don't touch the environment

### Startup behavior
- Guided first-run experience: walk user through initial config (paths, device check, create directories)
- Full environment validation on every launch (check deps, device, paths) — catches drift early
- Report and exit on critical failures — clear message with fix instructions, no auto-fixing

### Claude's Discretion
- Configuration approach (single file vs layered — pick what fits best)
- Virtual environment tool choice (pick best fit for usability and capability)
- PyTorch version pinning strategy (exact vs minimum)
- Whether to include a Makefile or task runner for common operations
- Startup output verbosity (essentials vs detailed)
- OOM recovery strategy (fail with guidance vs auto-fallback to CPU)
- Whether to support --device flag for manual device override

</decisions>

<specifics>
## Specific Ideas

- User specifically wants ease of use prioritized for dependency tooling — pick whatever makes the smoothest experience for end users
- Benchmark on first launch should help set intelligent defaults so users don't need to manually tune batch sizes

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-project-setup*
*Context gathered: 2026-02-12*
