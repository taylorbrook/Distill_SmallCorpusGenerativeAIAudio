---
phase: 15-ui-cli-vocoder-controls
verified: 2026-02-27T00:00:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
human_verification:
  - test: "Open the Gradio UI, load any model, open the Vocoder Settings accordion, and verify the dropdown shows 'Auto' and 'BigVGAN Universal' (no 'Per-model HiFi-GAN' unless model has vocoder_state)"
    expected: "Accordion is collapsed by default; dropdown reads 'Auto'; status text reads '**Using:** BigVGAN Universal'"
    why_human: "Gradio UI rendering cannot be verified without a running browser session"
  - test: "Click Generate when BigVGAN is not yet cached — inspect the Generate button label during the download"
    expected: "Button label changes to 'Downloading vocoder...' and becomes non-interactive; after download completes it reverts to 'Generate'"
    why_human: "Generator yield timing and Gradio streaming state cannot be verified statically"
  - test: "Run `distill generate some_model --vocoder bigvgan` in a terminal"
    expected: "A line matching 'Vocoder: BigVGAN Universal (bigvgan -- explicit)' is printed to stderr before generation results appear"
    why_human: "Stderr output and Rich markup rendering require a live CLI process"
  - test: "Run `distill generate some_model --vocoder bigvgan` when BigVGAN weights are NOT cached"
    expected: "A Rich-styled progress bar appears in the terminal (not a bare tqdm bar) showing download progress"
    why_human: "Rich vs. tqdm rendering distinction requires visual inspection of a live terminal"
---

# Phase 15: UI & CLI Vocoder Controls Verification Report

**Phase Goal:** Users can select their vocoder and see download progress in both the Gradio web UI and the CLI
**Verified:** 2026-02-27
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Generate tab shows a collapsible Vocoder Settings accordion with dropdown (Auto / BigVGAN Universal) and status text | VERIFIED | `generate_tab.py` lines 1087-1101: `gr.Accordion("Vocoder Settings", open=False)` contains `vocoder_dropdown` with choices `["Auto", "BigVGAN Universal"]` and `vocoder_status` Markdown component |
| 2 | Per-model HiFi-GAN only appears as a dropdown choice when the loaded model has vocoder_state | VERIFIED | `_update_vocoder_choices()` at lines 613-643: checks `app_state.loaded_model.vocoder_state is not None` before adding `"Per-model HiFi-GAN"` to choices |
| 3 | BigVGAN download progress is visible in the UI during first download (not a frozen interface) | VERIFIED | `_generate_audio()` is a generator (`yield`) with `progress=gr.Progress(track_tqdm=True)` at line 220; tqdm_class is NOT forwarded to `resolve_vocoder` in the UI path (see Note 1) — progress comes via Gradio's tqdm wrapper |
| 4 | Generate button is disabled with 'Downloading vocoder...' label during BigVGAN download and re-enabled after | VERIFIED | Lines 272-287: checks `is_bigvgan_cached()`, then yields `gr.update(interactive=False, value="Downloading vocoder...")` before resolution; every return/yield path re-enables button via `gr.update(interactive=True, value="Generate")` |
| 5 | Vocoder is resolved at generate time (not model load time), enabling lazy download | VERIFIED | `library_tab.py` line 166: `app_state.pipeline = None` — no `get_vocoder()` call at model load; `_generate_audio()` calls `resolve_vocoder()` before creating/updating pipeline |
| 6 | Quality badge after generation shows which vocoder was used | VERIFIED | `_quality_badge_markdown()` at lines 69-104 accepts `vocoder_info` and appends `"| **Vocoder:** BigVGAN Universal"` (or Per-model HiFi-GAN); called at line 405 with `vocoder_info=vocoder_info` |
| 7 | Running `distill generate model --vocoder bigvgan` selects BigVGAN vocoder | VERIFIED | `generate.py` lines 230-233: `vocoder: str = typer.Option("auto", "--vocoder", ...)` parameter declared; lines 592-598: `resolve_vocoder(selection=vocoder, ...)` called with the user's flag value |
| 8 | Running `distill generate model` (no flag) uses auto selection and prints vocoder info to stderr | VERIFIED | Default is `"auto"` (line 231); lines 603-612: `console.print(vocoder_line)` always executed after resolution; vocoder_line includes label + reason |
| 9 | Running `distill generate model --vocoder hifigan` on a model without per-model vocoder exits with non-zero status and clear error | VERIFIED | `resolve_vocoder()` in `__init__.py` lines 112-119 raises `ValueError` with message `"Model '...' has no trained per-model vocoder. Use --vocoder auto or bigvgan."`; CLI catches this at lines 599-600 and raises `typer.BadParameter` |
| 10 | BigVGAN first-time download shows Rich progress bar in terminal (not default tqdm) | VERIFIED | Lines 583-590: `tqdm_cls = tqdm_rich` when `not json_output`; `tqdm_cls` forwarded as `tqdm_class=tqdm_cls` to `resolve_vocoder()` at line 597; `warnings.filterwarnings` suppresses `TqdmExperimentalWarning` |
| 11 | JSON output includes vocoder field with name and selection | VERIFIED | Lines 650-658 (single model) and 449-457 (blend): `results.append({"file": ..., "vocoder": {"name": vocoder_info["name"], "selection": vocoder_info["selection"]}})` |

**Score:** 11/11 truths verified

**Note 1 (UI download progress):** The UI path in `_generate_audio()` calls `resolve_vocoder()` without a `tqdm_class` argument (line 292-296). The `gr.Progress(track_tqdm=True)` at line 220 hooks Gradio's own tqdm integration for download progress. This is architecturally correct — Gradio's progress wrapper intercepts tqdm globally when `track_tqdm=True`. The plan's intent (non-frozen UI + visible progress) is achieved via this mechanism.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/distill/vocoder/__init__.py` | `resolve_vocoder()` with auto/bigvgan/hifigan logic | VERIFIED | Lines 74-148: full implementation with `has_per_model` check, all three selection branches, `ValueError` on unknown selection, `NotImplementedError` placeholder for per-model (Phase 16). `resolve_vocoder` in `__all__`. |
| `src/distill/vocoder/weight_manager.py` | `tqdm_class` parameter for customizable download progress | VERIFIED | Line 17: `ensure_bigvgan_weights(tqdm_class: type | None = None)`. Lines 47-49: `if tqdm_class is not None: download_kwargs["tqdm_class"] = tqdm_class`. Used in `snapshot_download()` call. |
| `src/distill/ui/tabs/generate_tab.py` | Vocoder Settings accordion with dropdown, status, progress, retry button | VERIFIED | Lines 1088-1101: accordion exists with all four components. Lines 1256-1271: `generate_btn.click()` wires `vocoder_dropdown` as input and `generate_btn, vocoder_status, vocoder_progress, retry_download_btn` as outputs. |
| `src/distill/ui/tabs/library_tab.py` | Deferred vocoder creation (no vocoder at model load time) | VERIFIED | Line 166: `app_state.pipeline = None`. No `get_vocoder` or `BigVGANVocoder` import anywhere in file. Log message "vocoder deferred" confirms intent. |
| `src/distill/cli/generate.py` | `--vocoder` flag, vocoder status line, Rich download progress, JSON vocoder field | VERIFIED | Lines 230-233: flag declared. Lines 580-612: `tqdm_rich` wiring, `resolve_vocoder()` call, `console.print(vocoder_line)`. Lines 650-658: JSON vocoder field. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `generate_tab.py` | `vocoder/__init__.py` | `resolve_vocoder()` call in `_generate_audio` | WIRED | Line 241: `from distill.vocoder import resolve_vocoder`; line 292: `vocoder, vocoder_info = resolve_vocoder(...)` |
| `vocoder/__init__.py` | `vocoder/weight_manager.py` | `tqdm_class` forwarded to `ensure_bigvgan_weights` | WIRED | `get_vocoder()` passes `tqdm_class=tqdm_class` to `BigVGANVocoder`; `BigVGANVocoder.__init__` forwards to `ensure_bigvgan_weights(tqdm_class=tqdm_class)` at line 81 |
| `library_tab.py` | `inference/generation.py` | `GenerationPipeline` created without vocoder at model load | WIRED | `app_state.pipeline = None` at load time; `_generate_audio()` creates `GenerationPipeline(vocoder=vocoder)` after resolution at lines 314-320 |
| `cli/generate.py` | `vocoder/__init__.py` | `resolve_vocoder()` with `tqdm_class=tqdm_rich` | WIRED | Lines 577-598: import and call present with `tqdm_class=tqdm_cls` |
| `cli/generate.py` | `tqdm.rich` | `tqdm_rich` class passed as `tqdm_class` for Rich progress | WIRED | Lines 586-590: `from tqdm.rich import tqdm_rich; tqdm_cls = tqdm_rich` with `ImportError` fallback |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| UI-01 | 15-01-PLAN.md | Generate tab has vocoder selection (Auto / BigVGAN Universal / Per-model HiFi-GAN) | SATISFIED | `vocoder_dropdown` in Vocoder Settings accordion; `_update_vocoder_choices()` adds "Per-model HiFi-GAN" when `vocoder_state` present |
| UI-02 | 15-01-PLAN.md | BigVGAN download progress shown in UI on first use | SATISFIED | `gr.Progress(track_tqdm=True)` on `_generate_audio`; intermediate yield disables button before download; vocoder_progress Markdown + retry button for failures |
| CLI-01 | 15-02-PLAN.md | `--vocoder` flag on generate command selects vocoder (auto/bigvgan/hifigan) | SATISFIED | `typer.Option("auto", "--vocoder")` at lines 230-233; `resolve_vocoder(selection=vocoder, ...)` at lines 592-598; `typer.BadParameter` for hifigan on incompatible model |
| CLI-03 | 15-02-PLAN.md | BigVGAN download progress shown via Rich progress bar | SATISFIED | `tqdm_rich` imported and passed as `tqdm_class` when `not json_output`; `warnings.filterwarnings` suppresses `TqdmExperimentalWarning` |

No orphaned requirements. All four IDs declared across both plans are mapped to Phase 15 in REQUIREMENTS.md traceability table (lines 99, 100, 103, 105) and marked Complete.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `vocoder/__init__.py` | 65, 119 | `raise NotImplementedError("Per-model HiFi-GAN vocoder is Phase 16.")` | INFO | Expected intentional stub — per-model HiFi-GAN is explicitly Phase 16 work. Does not block phase goal. |

No blockers or warnings found. The `return []` instances in `generate_tab.py` at lines 742 and 802 are legitimate empty-collection returns from history/AB choice builders when the history store is empty, not stub implementations.

---

### Human Verification Required

#### 1. Vocoder Settings Accordion Rendering

**Test:** Launch the Gradio UI (`distill ui`), load any model from the Library tab, navigate to Generate tab, and expand the Vocoder Settings accordion.
**Expected:** Accordion is collapsed by default; dropdown shows "Auto" and "BigVGAN Universal"; status text shows "**Using:** BigVGAN Universal".
**Why human:** Gradio component rendering requires a live browser session.

#### 2. Generate Button Disable During Download

**Test:** With BigVGAN weights NOT yet cached, click Generate.
**Expected:** The Generate button immediately changes label to "Downloading vocoder..." and becomes non-interactive. After download completes (or uses cache), it reverts to "Generate" and the audio plays.
**Why human:** Generator yield timing and Gradio streaming behavior require a live session.

#### 3. CLI Vocoder Status Line

**Test:** Run `distill generate <model_name> --vocoder bigvgan`.
**Expected:** Line `Vocoder: BigVGAN Universal (bigvgan -- explicit)` appears on stderr before any generated file paths appear on stdout.
**Why human:** Stderr output and Rich markup rendering require a live CLI process.

#### 4. Rich Download Progress in Terminal

**Test:** Run `distill generate <model_name>` on a machine where BigVGAN is NOT yet cached.
**Expected:** A Rich-styled progress bar (styled by tqdm.rich, not default tqdm) renders in the terminal during the ~489MB download.
**Why human:** Visual distinction between Rich and default tqdm progress bars requires terminal observation.

---

### Gaps Summary

No gaps. All 11 observable truths are verified against the codebase. All five required artifacts exist and are substantive. All five key links are wired. All four phase requirements (UI-01, UI-02, CLI-01, CLI-03) are satisfied with implementation evidence. The phase goal — "Users can select their vocoder and see download progress in both the Gradio web UI and the CLI" — is achieved in the code.

Four items require human verification to confirm live rendering behavior, but these are quality confirmations rather than blockers; the underlying implementations are correctly wired.

---

_Verified: 2026-02-27_
_Verifier: Claude (gsd-verifier)_
