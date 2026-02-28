---
phase: 16-encode-decode-code-visualization
verified: 2026-02-27T00:00:00Z
status: passed
score: 15/15 must-haves verified
re_verification: null
gaps: []
human_verification:
  - test: "Click a colored code cell in the live Codes tab and listen to preview audio"
    expected: "The preview audio player autoplays with the sound of that codebook entry within 2-3 seconds"
    why_human: "Autoplay, audio quality, and latency require a running browser session with a real model loaded"
  - test: "Click a column header, then click a row Play button"
    expected: "Column header plays time-slice preview; row Play concatenates that level's audio across time"
    why_human: "Multi-event JS dispatch chain requires live Gradio + browser environment to verify"
  - test: "Encode an audio file and inspect the grid coloring"
    expected: "Each cell shows a code index number, background color from tab20 palette, white or black text for contrast, and the playhead line is visible above the grid"
    why_human: "Visual inspection of color contrast and grid layout requires a browser"
  - test: "Scroll the grid horizontally with a long audio file"
    expected: "Level label column (Structure/Timbre/Detail) stays fixed on the left while data columns scroll right"
    why_human: "CSS sticky positioning requires browser rendering to confirm"
  - test: "Load a VQ-VAE model, encode audio, then click Decode"
    expected: "Decoded reconstruction audio plays and is audibly similar to the original audio"
    why_human: "Audio quality of reconstruction is a perceptual judgment"
---

# Phase 16: Encode/Decode Code Visualization Verification Report

**Phase Goal:** Users can encode audio files into discrete code representations, view them as a labeled timeline grid, preview individual codebook entries as audio, and decode codes back to audio
**Verified:** 2026-02-27
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `encode_audio_file()` returns code indices, spatial_shape, mel_shape, and metadata from any audio file | VERIFIED | Function exists in `codes.py` (260 lines), all return dict keys confirmed, lazy imports, `torch.no_grad()`, mono mixdown, `_spatial_shape` captured immediately after forward |
| 2 | `decode_code_grid()` reconstructs audio waveform from code indices | VERIFIED | Calls `codes_to_embeddings` -> `decode` -> `mel_to_waveform`, returns 1-D float32 numpy array; fully wired to real model methods |
| 3 | `preview_single_code()` produces audio for a single codebook entry | VERIFIED | Creates zeroed indices tensor, sets target level to code_index at all positions, delegates to `decode_code_grid` |
| 4 | `preview_time_slice()` produces audio for all levels at one time position | VERIFIED | Extracts `pos_codes` at position, broadcasts to all positions, delegates to `decode_code_grid` |
| 5 | `play_row_audio()` concatenates decoded audio for one level across all time positions | VERIFIED | Iterates positions, calls `preview_single_code` per position, `np.concatenate` at end |
| 6 | `render_code_grid()` produces interactive HTML with per-cell coloring, JS onclick, and playhead | VERIFIED | `python -c` test confirms: `code-cell`, `Structure`, `playhead`, `code-grid-cell-clicked`, `dispatchEvent`, `sticky`, `nativeSet` all present in rendered HTML |
| 7 | Level labels follow cascading scheme: Structure/Detail (2L), Structure/Timbre/Detail (3L), Structure/Timbre/Texture/Detail (4L) | VERIFIED | `DEFAULT_LEVEL_LABELS` dict confirmed; `get_level_labels(2/3/4/5)` all return correct values including fallback |
| 8 | User can select a trained VQ-VAE model from a dropdown (v1.0 models filtered out) | VERIFIED | `_get_vqvae_model_choices()` checks `version >= 2 and model_type == "vqvae"` per file |
| 9 | User can upload an audio file and click Encode to see the code grid | VERIFIED | `_encode_audio` handler calls `encode_audio_file` + `render_code_grid`, wired to `encode_btn.click` |
| 10 | User can click any cell in the grid to hear that codebook entry as audio | VERIFIED | `_handle_cell_click` parses `"cell,{level},{position}"`, calls `preview_single_code`, returns `(48000, audio)` to `preview_audio` with `autoplay=True` |
| 11 | User can click a column header to hear the full time-slice decoded | VERIFIED | `_handle_cell_click` parses `"col,{position}"`, calls `preview_time_slice` |
| 12 | User can click Play on a row label to hear that level's contribution across time | VERIFIED | `_handle_cell_click` parses `"row,{level}"`, calls `play_row_audio` |
| 13 | User can click Decode to hear the full reconstruction and compare side-by-side with original | VERIFIED | `decode_btn.click` -> `_decode_current()` -> `decode_code_grid` -> `(48000, wav_array)` to `decoded_audio`; auto-decode on encode also provides immediate A/B |
| 14 | Grid shows level labels (Structure/Timbre/Detail) as sticky row headers | VERIFIED | `render_code_grid` generates `position: sticky; left: 0; z-index: 5` on `.code-grid-label`; Play button per row dispatches `"row,{level}"` |
| 15 | Playhead sweeps across grid during audio playback | VERIFIED | `playhead-line` div with `@keyframes playhead-sweep`, `animation-play-state: paused` default, `.playing` class toggle, `startPlayhead/stopPlayhead` JS functions in rendered HTML |

**Score:** 15/15 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/distill/inference/codes.py` | Encode/decode/preview pipeline functions | VERIFIED | 260 lines; exports `encode_audio_file`, `decode_code_grid`, `preview_single_code`, `preview_time_slice`, `play_row_audio`; all verified importable |
| `src/distill/ui/components/code_grid.py` | HTML grid renderer with JS bridge, playhead, level labels | VERIFIED | 375 lines; exports `render_code_grid`, `DEFAULT_LEVEL_LABELS`, `get_level_labels`; all verified importable and functional |
| `src/distill/ui/tabs/codes_tab.py` | Complete Codes tab with model selector, encode/decode, grid, audio players | VERIFIED | 537 lines (exceeds min_lines: 200); exports `build_codes_tab`; all event handlers wired |
| `src/distill/ui/app.py` | Codes tab registered alongside Data/Train/Generate/Library | VERIFIED | Tab order: Data, Train, Generate, **Codes**, Library; `build_codes_tab` imported and called; `codes_refs` used in cross-tab wiring |
| `src/distill/inference/__init__.py` | All 5 codes.py functions re-exported | VERIFIED | Lines 56-62 and `__all__` lines 105-110 confirm all 5 functions exported |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/distill/inference/codes.py` | `distill.models.vqvae.ConvVQVAE` | `loaded.model.forward()`, `codes_to_embeddings()`, `decode()` | WIRED | All three methods called in code; `loaded.model._spatial_shape` captured; `model.eval()` and `torch.no_grad()` applied |
| `src/distill/inference/codes.py` | `distill.audio.spectrogram.AudioSpectrogram` | `spectrogram.waveform_to_mel()`, `spectrogram.mel_to_waveform()` | WIRED | Both methods called on `loaded.spectrogram`; input/output shapes follow contract |
| `src/distill/ui/components/code_grid.py` | `codes_tab.py` (consumer) via JS bridge | `code-grid-cell-clicked` hidden Textbox, `dispatchEvent` | WIRED | `code-grid-cell-clicked` elem_id in `code_grid.py` JS; `cell_clicked = gr.Textbox(elem_id="code-grid-cell-clicked")` in `codes_tab.py`; `cell_clicked.change` -> `_handle_cell_click` |
| `src/distill/ui/tabs/codes_tab.py` | `distill.inference.codes` | `encode_audio_file, decode_code_grid, preview_single_code, preview_time_slice, play_row_audio` | WIRED | Pattern `from distill.inference.codes import` found in `_encode_audio`, `_decode_current`, `_handle_cell_click` handlers |
| `src/distill/ui/tabs/codes_tab.py` | `distill.ui.components.code_grid` | `render_code_grid`, `get_level_labels` | WIRED | Pattern `from distill.ui.components.code_grid import` found in `_encode_audio`, `_handle_cell_click`, `_update_level_labels` |
| `src/distill/ui/app.py` | `src/distill/ui/tabs/codes_tab.py` | `import build_codes_tab`, `gr.Tab("Codes")` | WIRED | `from distill.ui.tabs.codes_tab import build_codes_tab` at top of `app.py`; `codes_refs = build_codes_tab()` inside `gr.Tab("Codes", id="codes")` |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CODE-01 | 16-01, 16-02 | User can encode any audio file into its discrete code representation | SATISFIED | `encode_audio_file()` in `codes.py`; Encode button in `codes_tab.py` calls it; wired to `encode_btn.click` |
| CODE-02 | 16-01, 16-02 | User can decode a code grid back to audio with playback preview | SATISFIED | `decode_code_grid()` in `codes.py`; Decode button calls `_decode_current()`; auto-decode on encode for immediate A/B; `decoded_audio` player returned |
| CODE-03 | 16-01, 16-02 | User can view codes as a timeline grid (rows = quantizer levels, columns = time positions) | SATISFIED | `render_code_grid()` produces CSS grid; rows = quantizer levels (coarsest=Structure at top), columns = H*W positions; time markers every 10 positions |
| CODE-07 | 16-01, 16-02 | User can preview individual codebook entries as audio (click a code, hear it) | SATISFIED | Cell click -> `_handle_cell_click` -> `preview_single_code`; column click -> `preview_time_slice`; row Play -> `play_row_audio`; all return `(48000, audio)` to autoplay Preview player |
| CODE-09 | 16-01, 16-02 | Per-layer manipulation is labeled (Structure/Timbre/Detail) | SATISFIED | `DEFAULT_LEVEL_LABELS` with cascading scheme; row labels shown as sticky headers; Level Labels accordion for editing; `get_level_labels()` used throughout |

**Notes on REQUIREMENTS.md traceability:**
- UI-01 ("New Codes tab in Gradio UI for code visualization and editing") is mapped to Phase 17 in REQUIREMENTS.md and NOT claimed in Phase 16 plan frontmatter. However, Phase 16 Plan 02 did build the Codes tab. This reflects a requirement that will be fully satisfied when Phase 17 adds editing. No gap for Phase 16 — the tab was built as designed, editing is Phase 17 scope.
- All 5 requirement IDs from both plan frontmatter entries (CODE-01, CODE-02, CODE-03, CODE-07, CODE-09) are accounted for and satisfied.
- No orphaned requirements found for Phase 16.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `codes_tab.py` | 64 | `return []` | Info | Legitimate early return in `_get_vqvae_model_choices()` when `models_dir` does not exist — not a stub |

No blocking or warning anti-patterns found. No TODO/FIXME/HACK/PLACEHOLDER comments in any phase file.

---

### Human Verification Required

#### 1. Cell Click Audio Preview

**Test:** Load a VQ-VAE v2 model from the Codes tab dropdown, upload a short WAV file, click Encode, then click any colored cell in the grid.
**Expected:** The Preview audio player autoplays within 2-3 seconds with audio reflecting that codebook entry.
**Why human:** Autoplay, audio quality, and latency require a live browser session with a real model loaded.

#### 2. Column Header and Row Play Buttons

**Test:** After encoding, click a column header (time-position label) and then click a "Play" button on a row label.
**Expected:** Column header triggers time-slice preview audio; row Play button concatenates audio for that quantizer level across all time positions.
**Why human:** JS dispatch chain to Gradio textbox change event requires live browser to confirm end-to-end.

#### 3. Grid Visual Layout and Cell Coloring

**Test:** Encode any audio file and visually inspect the grid.
**Expected:** Cells have colored backgrounds (20 distinct colors from tab20), code index numbers are readable (white or black text by luminance), playhead line is visible above the grid, sticky labels visible on left.
**Why human:** CSS rendering and color contrast are visual qualities that require browser inspection.

#### 4. Horizontal Scroll with Sticky Labels

**Test:** Encode a long audio file (>10 seconds) to produce many grid columns, then scroll the grid horizontally.
**Expected:** "Structure", "Timbre", "Detail" row labels stay pinned to the left; data cells scroll under them.
**Why human:** CSS `position: sticky` behavior requires browser rendering to confirm.

#### 5. Decode Reconstruction Quality

**Test:** Encode audio, click Decode (or use auto-decoded audio), and compare with Original Audio.
**Expected:** Decoded reconstruction is audibly similar to the original (not silence, not noise).
**Why human:** Audio reconstruction quality is a perceptual judgment.

---

### Gaps Summary

No gaps. All 15 observable truths verified, all 5 artifacts are substantive and wired, all 6 key links confirmed wired, all 5 requirement IDs satisfied. The implementation is complete and correctly structured.

**Commit hashes verified:** `dd58e10` (codes.py), `1fd01f0` (code_grid.py), `02fa4d9` (codes_tab.py), `ce72c1c` (app.py) — all confirmed in git log.

---

_Verified: 2026-02-27_
_Verifier: Claude (gsd-verifier)_
