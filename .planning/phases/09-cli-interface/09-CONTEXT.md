# Phase 9: CLI Interface - Context

**Gathered:** 2026-02-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Command-line interface for batch generation, scripting, and headless operation. Exposes generation, training, and model management operations. Does NOT add new capabilities — wraps existing Phase 1-8 functionality for terminal use.

</domain>

<decisions>
## Implementation Decisions

### Command structure
- Claude's discretion on subcommand vs flat style (recommend subcommand-based given 3 operation domains)
- Claude's discretion on entry point command name (recommend short, ergonomic)
- Operations exposed: generate, train, model management (list/info/delete)
- Presets are loadable from CLI (read-only — preset creation/management stays in GUI)
- Include `sda ui` (or equivalent) subcommand to launch Gradio GUI from the CLI entry point

### Output & progress
- Claude's discretion on default verbosity level
- Training progress: tqdm-style progress bar with epoch/loss/ETA
- Claude's discretion on --json flag for machine-readable output
- Claude's discretion on stderr/stdout routing for errors vs data

### Batch generation
- Claude's discretion on batch specification approach (count flag, seed lists, config file, or combination)
- Output defaults to project's configured output directory (from config), with --output-dir override
- No dry-run mode — generation is fast enough that it's unnecessary
- Claude's discretion on parameter sweep support

### Headless workflow
- Claude's discretion on model resolution (library name, file path, or both)
- Claude's discretion on path override mechanism (flags, env vars, or both)
- Claude's discretion on training config exposure (full flags, named presets + overrides, or hybrid)

### Claude's Discretion
- CLI framework choice (argparse, click, typer, etc.)
- Command naming conventions
- Default verbosity level
- JSON output support
- Stderr/stdout routing
- Batch specification syntax
- Parameter sweep support
- Model resolution strategy
- Path override mechanism (flags vs env vars vs both)
- Training config flag design
- Exit codes and error message formatting
- Help text style and depth

</decisions>

<specifics>
## Specific Ideas

- User wants a unified entry point: CLI for scripting AND `sda ui` to launch the GUI from the same command
- Presets created in GUI should be accessible from CLI for reproducible batch generation
- Training progress must use progress bars (not log lines) for terminal UX
- Output files go to project output directory by default (not cwd)
- No dry-run mode needed

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-cli-interface*
*Context gathered: 2026-02-13*
