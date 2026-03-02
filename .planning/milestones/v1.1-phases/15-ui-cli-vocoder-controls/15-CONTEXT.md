# Phase 15: UI & CLI Vocoder Controls - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can select their vocoder and see download progress in both the Gradio web UI and the CLI. This phase adds vocoder selection controls (Auto / BigVGAN Universal / Per-model HiFi-GAN), download progress visibility, and resolution feedback. Training a per-model vocoder and removing Griffin-Lim are Phase 16.

</domain>

<decisions>
## Implementation Decisions

### Selector placement & behavior
- Vocoder controls live in a **collapsible "Vocoder Settings" accordion** below the Generation Config row
- Accordion is collapsed by default (Claude's discretion on smart-open for first download or when per-model vocoder exists)
- Dropdown options: Auto, BigVGAN Universal, Per-model HiFi-GAN
- **Per-model HiFi-GAN is disabled (grayed out) with tooltip** when the current model doesn't have a trained per-model vocoder — prevents selecting an unavailable option
- Status text inside accordion shows **readiness + resolution**: e.g., "Using: BigVGAN Universal" or "Using: Per-model HiFi-GAN"

### Download experience
- BigVGAN download progress appears **inline in the vocoder accordion** with progress bar and MB counter (e.g., "Downloading BigVGAN universal model... 245/489 MB")
- Download is **lazy** — triggered on first generate attempt, not on app startup
- **Generate button disabled** during download with tooltip "Downloading vocoder..."
- On download failure: **error message in accordion + Retry Download button** (not a Gradio toast)

### CLI vocoder output
- **Always print vocoder line** on every generate call to stderr: `Vocoder: BigVGAN Universal (auto)`
- `--vocoder` flag accepts: auto, bigvgan, hifigan
- **Auto is the default** — no flag needed, `distill generate model` just works
- When `--vocoder hifigan` is specified but model has no per-model vocoder: **error and exit** with non-zero status and message: "Error: model X has no trained per-model vocoder. Use --vocoder auto or bigvgan."
- BigVGAN download uses **Rich progress bar** (consistent with CLI's existing Rich console), not HuggingFace Hub's tqdm default

### Auto-resolution feedback
- **UI: both places** — status text in accordion shows resolution before generation ("Using: BigVGAN Universal"), quality badge after generation confirms vocoder used alongside seed/sample-rate/bit-depth
- **CLI: label + reason** — `Vocoder: BigVGAN Universal (auto — no per-model vocoder)` or `Vocoder: Per-model HiFi-GAN (auto — per-model available)` — explains WHY auto chose what it chose
- **JSON output includes vocoder field** — `{"vocoder": {"name": "bigvgan_universal", "selection": "auto"}}` added to `--json` output

### Claude's Discretion
- Accordion smart-open logic (whether to auto-open on first download or when per-model vocoder available)
- Exact tooltip wording for disabled Per-model HiFi-GAN option
- Progress bar styling details within Gradio constraints
- Rich progress bar formatting (speed, ETA display)

</decisions>

<specifics>
## Specific Ideas

- Vocoder accordion mockup: dropdown + status line, with progress bar appearing inline during downloads
- CLI output pattern: `Vocoder: BigVGAN Universal (auto — no per-model vocoder)` — always shown, explains auto-selection reasoning
- Error-in-accordion pattern for download failures (not Gradio toast): warning icon + message + retry button
- JSON vocoder field: `{"name": "bigvgan_universal", "selection": "auto"}` nested object
- Quality badge after generation should include vocoder name alongside existing metadata (seed, sample rate, bit depth)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-ui-cli-vocoder-controls*
*Context gathered: 2026-02-27*
