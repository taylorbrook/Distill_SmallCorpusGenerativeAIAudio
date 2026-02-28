# Phase 16: Encode/Decode + Code Visualization - Research

**Researched:** 2026-02-27
**Domain:** VQ-VAE encode/decode pipeline, interactive code grid visualization, Gradio UI
**Confidence:** HIGH

## Summary

Phase 16 adds a new "Codes" tab to the Gradio UI where users can load an audio file, encode it through a trained VQ-VAE model into discrete codes, visualize those codes as an interactive timeline grid, preview individual codebook entries as audio, and decode the full code grid back to audio for A/B comparison with the original. All backend infrastructure already exists: `ConvVQVAE.encode()`, `quantize()`, `decode()`, `codes_to_embeddings()`, `QuantizerWrapper.get_output_from_indices()`, `AudioSpectrogram.waveform_to_mel()` / `mel_to_waveform()`, and `load_audio()` / `load_model_v2()`. The work is primarily UI construction and wiring.

The critical architectural insight is that the VQ-VAE encodes audio into a 2D spatial grid of shape `[H, W]` where H=8 (downsampled frequency) and W varies by duration (6 per second of audio). Each spatial position is independently quantized through `num_quantizers` (2-4) RVQ levels. The code visualization grid should present this as a timeline where columns represent spatial positions (H*W total, ordered in raster scan), and rows represent quantizer levels labeled with semantic roles (Structure/Timbre/Detail). For user-facing display, the grid treats the flattened `H*W` positions as a linear timeline since the 16x frequency downsampling makes individual frequency positions opaque to the user -- what matters is the temporal progression.

The UI should be built using `gr.HTML` with inline JavaScript for the interactive grid, following the established `model_card.py` pattern of hidden `gr.Textbox` with `elem_id` for JavaScript-to-Python event communication. Audio preview uses `gr.Audio` components with `autoplay=True`. No new Python dependencies are needed.

**Primary recommendation:** Build the Codes tab with an HTML-rendered code grid using the established JS-to-Python communication pattern, backed by a new `distill.inference.codes` module containing the encode/decode/preview pipeline functions.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Cell coloring and content display: Claude's discretion (optimize for readability at typical grid sizes)
- Grid scrolls horizontally for audio longer than one screen width; all quantizer levels remain visible vertically
- Playback position indicator: a vertical playhead line sweeps across the grid during audio playback, showing which codes are currently sounding
- Click a cell = instant auto-play of that single code decoded through the VQ-VAE decoder (the "pure sound" of that codebook entry)
- Click a column header = instant auto-play of the full time-slice (all levels at that position decoded together)
- Clicked cell gets a visible selection highlight in the grid
- "Play row" button per level row: plays all codes in that row sequentially along the timeline, so the user can hear a single level's contribution
- Fixed default labels, but user can rename them
- Cascading detail scheme based on number of RVQ levels:
  - 2 levels: Structure / Detail
  - 3 levels: Structure / Timbre / Detail
  - 4 levels: Structure / Timbre / Texture / Detail
- Labels appear as row headers on the left side of the grid (always visible)
- Row ordering (coarsest top vs bottom): Claude's discretion
- New top-level "Codes" tab alongside Train, Generate, etc. (Phase 17 editing will extend this tab)
- Layout: controls at top (upload audio, model selector dropdown, encode/decode buttons), code grid takes main area below
- Explicit model selector dropdown -- no auto-detection, user picks a trained VQ-VAE model
- Side-by-side audio players: original audio on one side, decoded reconstruction on the other, for A/B quality comparison

### Claude's Discretion
- Cell coloring approach (color by code index, by level, or hybrid)
- Cell content display (color block only vs color + code number)
- Row ordering (coarsest at top or bottom)
- Exact grid cell sizing and spacing
- Playhead implementation approach within Gradio constraints
- How single-code audio preview is rendered (duration, windowing)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CODE-01 | User can encode any audio file into its discrete code representation | `load_audio()` + `waveform_to_mel()` + `ConvVQVAE.forward()` returns indices `[B, H*W, num_quantizers]`; pipeline already proven in `extract_code_sequences()` and `generate_audio_from_prior()` |
| CODE-02 | User can decode a code grid back to audio with playback preview | `codes_to_embeddings(indices, spatial_shape)` + `decode(quantized, target_shape)` + `mel_to_waveform()` already exists and is used in generation pipeline |
| CODE-03 | User can view codes as a timeline grid (rows = quantizer levels, columns = time positions) | HTML grid rendered via `gr.HTML` with CSS grid layout; indices reshaped from `[B, H*W, Q]` to display as Q rows x (H*W) columns |
| CODE-07 | User can preview individual codebook entries as audio (click a code, hear it) | `rvq.codebooks[level, code_index]` gives embedding; pass through decoder + `mel_to_waveform()` for audio preview |
| CODE-09 | Per-layer manipulation is labeled (Structure/Timbre/Detail) | Fixed cascading labels with user-editable override via `gr.Textbox`; stored in UI state |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gradio | 6.6.0 (installed) | UI framework for Codes tab, HTML grid, Audio players | Already used for all UI tabs |
| torch | 2.10.x (installed) | VQ-VAE inference, tensor operations | Already used everywhere |
| torchaudio | 2.10.x (installed) | Griffin-Lim waveform reconstruction via AudioSpectrogram | Already used for mel-to-waveform |
| vector-quantize-pytorch | >=1.27 (installed) | `ResidualVQ.get_output_from_indices()`, `get_codes_from_indices()`, `codebooks` tensor access | Already used by QuantizerWrapper |
| soundfile | >=0.13 (installed) | Audio file loading via `load_audio()` | Already used for audio I/O |
| numpy | >=1.26 (installed) | Audio array handling for Gradio Audio component | Already used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | >=3.9 (installed) | Color map generation for code cell coloring (e.g., `plt.cm.tab20`) | For generating deterministic, perceptually distinct cell colors |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| HTML grid via `gr.HTML` | `gr.Dataframe` | Dataframe lacks per-cell coloring, click-to-play, playhead animation -- too limited for this interactive grid |
| Hidden Textbox JS bridge | Gradio custom component | Custom components require separate npm build, packaging overhead; hidden Textbox is proven pattern in this codebase |
| `matplotlib.cm` for colors | Hand-coded color palette | matplotlib colormap gives consistent perceptually-distinct colors for arbitrary codebook sizes; already installed |

**Installation:**
```bash
# No new packages needed -- all dependencies already installed
```

## Architecture Patterns

### Recommended Project Structure
```
src/distill/
├── inference/
│   └── codes.py            # NEW: encode/decode/preview pipeline functions
├── ui/
│   ├── tabs/
│   │   └── codes_tab.py    # NEW: Codes tab builder
│   └── components/
│       └── code_grid.py    # NEW: HTML grid renderer (like model_card.py)
└── ...
```

### Pattern 1: Encode Pipeline (audio -> indices)
**What:** Load audio, convert to mel, encode through VQ-VAE, return indices + spatial shape
**When to use:** When user uploads audio and clicks "Encode"
**Example:**
```python
# Source: Codebase analysis of models/prior.py extract_code_sequences()
# and inference/generation.py generate_audio_from_prior()

def encode_audio(
    audio_path: Path,
    loaded: LoadedVQModel,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Encode an audio file to VQ-VAE code indices.

    Returns:
        (indices, spatial_shape, mel_shape) where
        indices is [1, H*W, num_quantizers],
        spatial_shape is (H, W),
        mel_shape is (n_mels, time_frames).
    """
    from distill.audio.io import load_audio

    audio_file = load_audio(audio_path, target_sample_rate=48000)
    waveform = audio_file.waveform  # [channels, samples]

    # Mono mixdown if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert to mel: [1, 1, n_mels, time]
    mel = loaded.spectrogram.waveform_to_mel(
        waveform.unsqueeze(0).to(loaded.device)
    )
    mel_shape = (mel.shape[2], mel.shape[3])

    with torch.no_grad():
        loaded.model.eval()
        recon, indices, _ = loaded.model(mel)
        spatial_shape = loaded.model._spatial_shape

    return indices, spatial_shape, mel_shape
```

### Pattern 2: Decode Pipeline (indices -> audio)
**What:** Convert code indices back to audio waveform through VQ-VAE decoder
**When to use:** When user clicks "Decode" or for code preview playback
**Example:**
```python
# Source: Codebase analysis of inference/generation.py generate_audio_from_prior()

def decode_codes(
    indices: torch.Tensor,
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
    loaded: LoadedVQModel,
) -> np.ndarray:
    """Decode code indices back to audio waveform.

    Returns:
        1-D float32 numpy array of audio samples at 48kHz.
    """
    with torch.no_grad():
        loaded.model.eval()
        quantized = loaded.model.codes_to_embeddings(
            indices.to(loaded.device), spatial_shape,
        )
        mel = loaded.model.decode(quantized, target_shape=mel_shape)
        wav = loaded.spectrogram.mel_to_waveform(mel)

    return wav.squeeze().numpy().astype(np.float32)
```

### Pattern 3: Single Code Preview (click cell -> hear it)
**What:** Decode a single codebook entry to audio for instant preview
**When to use:** When user clicks a cell in the code grid
**Example:**
```python
# Source: Codebase analysis of vector-quantize-pytorch ResidualVQ API
# rvq.codebooks shape: [num_quantizers, codebook_size, dim]
# rvq.get_codes_from_indices shape: [num_quantizers, B, seq_len, dim]

def preview_single_code(
    level: int,
    code_index: int,
    loaded: LoadedVQModel,
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
) -> np.ndarray:
    """Generate audio preview for a single codebook entry.

    Creates a code grid with the target code at all positions for the
    given level, zeros for other levels. This produces the 'pure sound'
    of that codebook entry.

    Returns:
        1-D float32 numpy array of audio samples.
    """
    H, W = spatial_shape
    seq_len = H * W
    num_q = loaded.model.num_quantizers

    # Create indices: target code at all positions for this level,
    # zeros elsewhere (level 0 code 0 has minimal contribution)
    indices = torch.zeros(1, seq_len, num_q, dtype=torch.long, device=loaded.device)
    indices[0, :, level] = code_index

    return decode_codes(indices, spatial_shape, mel_shape, loaded)
```

### Pattern 4: HTML Grid with JS Bridge (established codebase pattern)
**What:** Render interactive HTML via `gr.HTML`, communicate clicks to Python via hidden `gr.Textbox`
**When to use:** For the code grid visualization with click-to-preview
**Example:**
```python
# Source: Codebase analysis of ui/components/model_card.py

# Hidden textbox for JS -> Python bridge
cell_clicked = gr.Textbox(
    value="", visible=False,
    elem_id="code-grid-cell-clicked",
)

# HTML grid with onclick handlers
grid_html = gr.HTML(value=render_code_grid(None))

# Wire: cell click -> update textbox -> trigger Python handler
cell_clicked.change(
    fn=_handle_cell_click,
    inputs=[cell_clicked],
    outputs=[preview_audio],
)
```

```javascript
// In the HTML onclick handler (same pattern as model_card.py):
onclick="
  var cellInfo = this.getAttribute('data-cell');
  var tb = document.querySelector('#code-grid-cell-clicked textarea');
  if (tb) {
    var nativeSet = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, 'value').set;
    nativeSet.call(tb, cellInfo);
    tb.dispatchEvent(new Event('input', {bubbles: true}));
  }
"
```

### Pattern 5: Playhead Animation via CSS
**What:** A vertical line sweeping across the grid during audio playback
**When to use:** When original or decoded audio is playing
**Example:**
```css
/* CSS animation approach for playhead */
@keyframes playhead-sweep {
  from { left: 0%; }
  to { left: 100%; }
}

.playhead-line {
  position: absolute;
  top: 0;
  width: 2px;
  height: 100%;
  background: #ef4444;
  z-index: 10;
  pointer-events: none;
  animation: playhead-sweep var(--duration) linear;
  animation-play-state: paused;
}

.playhead-line.playing {
  animation-play-state: running;
}
```

### Anti-Patterns to Avoid
- **Building a custom Gradio component for the grid:** Requires npm build pipeline, separate package, breaks the single-repo pattern. Use `gr.HTML` + CSS grid instead.
- **Using `gr.Dataframe` for code visualization:** No per-cell coloring, no click-to-play, no playhead -- too limited. HTML is the right approach.
- **Real-time re-encoding during playback:** Griffin-Lim is too slow for interactive feedback. Pre-decode everything and use CSS animation for the playhead.
- **Storing code grids in app_state as nested Python lists:** Use `torch.Tensor` for indices (efficient storage, easy slicing) and convert to display format only for rendering.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio file loading + resampling | Custom file reader | `distill.audio.io.load_audio()` | Handles format detection, resampling, mono/stereo |
| Mel spectrogram conversion | Custom mel transform | `AudioSpectrogram.waveform_to_mel()` / `mel_to_waveform()` | Handles log1p normalization, CPU-forced InverseMelScale |
| Code index to embedding lookup | Manual codebook indexing | `QuantizerWrapper.get_output_from_indices()` and `rvq.get_codes_from_indices()` | Handles multi-level RVQ summation correctly |
| Spatial embedding reshaping | Manual reshape logic | `ConvVQVAE.codes_to_embeddings(indices, spatial_shape)` | Already handles the [B,S,D] -> [B,D,H,W] reshape |
| Model loading | Custom loader | `load_model_v2()` returning `LoadedVQModel` | Handles version detection, config reconstruction, device placement |
| VQ-VAE model listing for dropdown | Custom file scanning | `ModelLibrary.list_all()` with v2 filter | Already has catalog management, file_path resolution |
| JS-to-Python event bridge | Custom WebSocket | Hidden `gr.Textbox` with `elem_id` + `dispatchEvent` | Proven pattern in `model_card.py`, works reliably in Gradio 6 |
| Perceptually distinct colors | Hand-coded palette | `matplotlib.cm.tab20` or similar colormap | Deterministic, handles up to 256 codes gracefully |

**Key insight:** The entire encode/decode pipeline already exists across `models/vqvae.py`, `audio/spectrogram.py`, `audio/io.py`, and `models/persistence.py`. Phase 16 is primarily UI construction -- new backend code is limited to thin orchestration functions in a new `inference/codes.py` module.

## Common Pitfalls

### Pitfall 1: Spatial Shape Not Stored After Encode
**What goes wrong:** After encoding, `model._spatial_shape` is set but if the model processes another input, it gets overwritten.
**Why it happens:** `_spatial_shape` is set during `forward()` as a side effect.
**How to avoid:** Capture and store `spatial_shape` immediately after encode, pass it explicitly to decode functions. Never rely on `model._spatial_shape` persisting across calls.
**Warning signs:** Decode produces garbled audio after a second encode call.

### Pitfall 2: Griffin-Lim Latency for Preview Audio
**What goes wrong:** Clicking a cell to preview a code triggers a full `mel_to_waveform()` call with 128 Griffin-Lim iterations, causing noticeable delay (200-500ms).
**Why it happens:** Griffin-Lim is iterative and CPU-bound.
**How to avoid:** Pre-compute preview audio for all codes at encode time is too expensive (codebook_size * num_quantizers previews). Instead, accept the ~200-500ms latency for single-code preview and use `gr.Audio(autoplay=True)` so it plays immediately once computed. For "play row", pre-compute each chunk and concatenate.
**Warning signs:** UI feels sluggish on code clicks.

### Pitfall 3: Gradio Autoplay Browser Restrictions
**What goes wrong:** `autoplay=True` on `gr.Audio` may be blocked by browsers that require user interaction before audio can play.
**Why it happens:** Browser autoplay policies (Chrome, Firefox, Safari all have them).
**How to avoid:** The click on the grid cell IS a user interaction, so autoplay should be permitted in most browsers since it's in response to a user gesture. If issues arise, use a small `gr.Audio` component and set its value + trigger play via the event handler.
**Warning signs:** Audio doesn't play on first click, works after interacting with other audio components.

### Pitfall 4: Large HTML Grid Performance
**What goes wrong:** For 10 seconds of audio: H*W=472 positions x 3 levels = 1416 cells. Rendering this as HTML with inline styles can be slow to update in Gradio.
**Why it happens:** Each `gr.HTML` value update triggers a full DOM replacement.
**How to avoid:** Use CSS grid with minimal inline styles (class-based coloring). For very long audio, consider chunked rendering with horizontal scroll. Cell sizing should be compact (20-30px) so many fit on screen.
**Warning signs:** Visible lag when switching between encoded files.

### Pitfall 5: Wrong Mel Shape for Decode
**What goes wrong:** Decoded audio has wrong duration or sounds distorted.
**Why it happens:** `mel_to_waveform()` and `decode()` need the original mel shape `(n_mels, time_frames)` to crop correctly. If the wrong shape is passed, the decoder output is either truncated or includes padding artifacts.
**How to avoid:** Store `mel_shape` alongside `indices` and `spatial_shape` at encode time. Always pass it to `decode(quantized, target_shape=mel_shape)`.
**Warning signs:** Decoded audio is shorter/longer than original or has clicks at the end.

### Pitfall 6: Column Header Click vs Cell Click Disambiguation
**What goes wrong:** User clicks a column header expecting to hear the full time-slice, but the handler treats it as a cell click.
**Why it happens:** Both use the same HTML element onclick pattern.
**How to avoid:** Use different `data-*` attributes for cell clicks vs column header clicks. Parse the click info string in the Python handler to dispatch to the correct preview function (`preview_single_code` vs `preview_time_slice`).
**Warning signs:** Wrong audio plays when clicking headers vs cells.

### Pitfall 7: Model Not Filtering VQ-VAE Only
**What goes wrong:** User selects a v1.0 (continuous VAE) model from the dropdown, which has no codebook or RVQ.
**Why it happens:** Model dropdown shows all models from the library.
**How to avoid:** Filter `ModelLibrary.list_all()` to only show v2 models with `model_type="vqvae"`. The library catalog stores `file_path` -- peek at the `.distill` file version/type, or add a `model_type` field to `ModelEntry` if not present.
**Warning signs:** Error when trying to encode with a v1.0 model.

## Code Examples

### Encode Audio File to Code Grid
```python
# Source: Codebase analysis -- combines load_audio, waveform_to_mel, model.forward()

import torch
import numpy as np
from pathlib import Path
from distill.audio.io import load_audio
from distill.models.persistence import LoadedVQModel

def encode_audio_file(
    audio_path: Path,
    loaded: LoadedVQModel,
) -> dict:
    """Encode audio file to code grid for visualization.

    Returns dict with:
        indices: [1, H*W, num_quantizers] tensor
        spatial_shape: (H, W)
        mel_shape: (n_mels, time_frames)
        num_quantizers: int
        codebook_size: int
        duration_s: float
    """
    audio_file = load_audio(audio_path, target_sample_rate=48000)
    waveform = audio_file.waveform  # [channels, samples]

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = loaded.spectrogram.waveform_to_mel(
        waveform.unsqueeze(0).to(loaded.device)
    )
    mel_shape = (mel.shape[2], mel.shape[3])

    with torch.no_grad():
        loaded.model.eval()
        _recon, indices, _loss = loaded.model(mel)
        spatial_shape = loaded.model._spatial_shape

    return {
        "indices": indices.cpu(),
        "spatial_shape": spatial_shape,
        "mel_shape": mel_shape,
        "num_quantizers": loaded.model.num_quantizers,
        "codebook_size": loaded.model.codebook_size,
        "duration_s": waveform.shape[-1] / 48000,
    }
```

### Decode Code Grid to Audio
```python
# Source: Codebase analysis -- combines codes_to_embeddings, decode, mel_to_waveform

def decode_code_grid(
    indices: torch.Tensor,
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
    loaded: LoadedVQModel,
) -> np.ndarray:
    """Decode code indices back to audio waveform.

    Returns 1-D float32 numpy array at 48kHz.
    """
    with torch.no_grad():
        loaded.model.eval()
        quantized = loaded.model.codes_to_embeddings(
            indices.to(loaded.device), spatial_shape,
        )
        mel = loaded.model.decode(quantized, target_shape=mel_shape)
        wav = loaded.spectrogram.mel_to_waveform(mel)

    return wav.squeeze().cpu().numpy().astype(np.float32)
```

### Preview Single Codebook Entry
```python
# Source: vector-quantize-pytorch API analysis
# rvq.codebooks: [num_quantizers, codebook_size, dim]

def preview_single_code(
    level: int,
    code_index: int,
    loaded: LoadedVQModel,
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
) -> np.ndarray:
    """Generate audio for a single codebook entry at a given level.

    Places the target code at all spatial positions for the given level,
    with zero codes at other levels (minimal contribution from code 0).

    Returns 1-D float32 numpy array at 48kHz.
    """
    H, W = spatial_shape
    seq_len = H * W
    num_q = loaded.model.num_quantizers

    indices = torch.zeros(1, seq_len, num_q, dtype=torch.long)
    indices[0, :, level] = code_index

    return decode_code_grid(indices, spatial_shape, mel_shape, loaded)
```

### Preview Time Slice (Column Click)
```python
# Source: Codebase analysis -- all levels at one time position

def preview_time_slice(
    position: int,
    full_indices: torch.Tensor,
    loaded: LoadedVQModel,
    spatial_shape: tuple[int, int],
    mel_shape: tuple[int, int],
) -> np.ndarray:
    """Generate audio for a single time position (all levels).

    Extracts the codes at the given position from the full grid,
    places them at all positions (to fill the spatial grid), and decodes.

    Returns 1-D float32 numpy array at 48kHz.
    """
    H, W = spatial_shape
    seq_len = H * W
    num_q = loaded.model.num_quantizers

    # Get codes at this position for all levels
    pos_codes = full_indices[0, position, :]  # [num_quantizers]

    # Fill all positions with these codes (broadcast)
    indices = pos_codes.unsqueeze(0).unsqueeze(0).expand(1, seq_len, num_q).clone()

    return decode_code_grid(indices, spatial_shape, mel_shape, loaded)
```

### Render Code Grid HTML
```python
# Source: Pattern from ui/components/model_card.py

import html as html_mod
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def render_code_grid(
    indices: torch.Tensor | None,
    num_quantizers: int = 3,
    codebook_size: int = 256,
    spatial_shape: tuple[int, int] = (8, 6),
    level_labels: list[str] | None = None,
    selected_cell: tuple[int, int] | None = None,
) -> str:
    """Render code indices as an interactive HTML grid.

    Grid layout: rows = quantizer levels (coarsest at top),
    columns = spatial positions (time progression left to right).

    Each cell shows the code index with a background color derived
    from a colormap. Clicking a cell dispatches the position info
    to a hidden textbox for Python handling.
    """
    if indices is None:
        return '<div style="text-align:center; padding:40px; color:#888;">No codes encoded yet.</div>'

    H, W = spatial_shape
    seq_len = H * W

    # Default level labels
    if level_labels is None:
        label_schemes = {
            2: ["Structure", "Detail"],
            3: ["Structure", "Timbre", "Detail"],
            4: ["Structure", "Timbre", "Texture", "Detail"],
        }
        level_labels = label_schemes.get(num_quantizers,
            [f"Level {i}" for i in range(num_quantizers)])

    # Generate color map: code_index -> hex color
    cmap = plt.cm.tab20

    # Build grid HTML with CSS grid
    cells_html = []
    for q in range(num_quantizers):
        for pos in range(seq_len):
            code_idx = int(indices[0, pos, q].item())
            color = mcolors.to_hex(cmap(code_idx % 20))
            is_selected = (selected_cell == (q, pos))
            border = "3px solid #000" if is_selected else "1px solid #e5e7eb"

            cells_html.append(
                f'<div class="code-cell" '
                f'data-cell="{q},{pos}" '
                f'style="background:{color}; border:{border}; '
                f'cursor:pointer; text-align:center; font-size:10px; '
                f'line-height:24px; min-width:24px; min-height:24px;" '
                f'onclick="...(JS bridge code)...">'
                f'{code_idx}</div>'
            )

    # Assemble with CSS grid
    grid_css = (
        f'display:grid; '
        f'grid-template-columns: 80px repeat({seq_len}, 24px); '
        f'grid-template-rows: repeat({num_quantizers}, 24px); '
        f'gap:1px; overflow-x:auto; position:relative;'
    )

    return f'<div style="{grid_css}">...</div>'
```

### Level Label Cascading Scheme
```python
# Source: CONTEXT.md decisions

DEFAULT_LEVEL_LABELS = {
    2: ["Structure", "Detail"],
    3: ["Structure", "Timbre", "Detail"],
    4: ["Structure", "Timbre", "Texture", "Detail"],
}

def get_level_labels(num_quantizers: int) -> list[str]:
    """Get default semantic labels for RVQ levels."""
    return DEFAULT_LEVEL_LABELS.get(
        num_quantizers,
        [f"Level {i}" for i in range(num_quantizers)],
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom Gradio components for interactive HTML | `gr.HTML` with `js_on_load` and hidden Textbox bridge | Gradio 5+ (2024) | Simpler than building custom npm components; project already uses this pattern |
| Manual codebook embedding lookup | `rvq.codebooks` tensor + `get_output_from_indices()` | vector-quantize-pytorch v1.x | Direct tensor access for single-entry preview; summed output for full decode |
| Server-side grid rendering only | CSS Grid + JS onclick for client-side interactivity | Modern web pattern | Fast click response, smooth playhead animation without server roundtrips |

**Deprecated/outdated:**
- `gr.HTML` `server_functions` parameter: Documented in Gradio docs but NOT available in Gradio 6.6.0 (checked via `inspect.signature`). Use hidden Textbox bridge instead.
- Gradio custom components: Overkill for this use case; the HTML+CSS+JS approach is simpler and proven in this project.

## Open Questions

1. **Playhead synchronization accuracy**
   - What we know: CSS animation can sweep a line across the grid for the audio duration. `gr.Audio` component supports `play` and `stop` events.
   - What's unclear: How precisely the CSS animation start can be synchronized with `gr.Audio.play()` in Gradio 6. The `play` event fires when playback begins, but there may be a small delay.
   - Recommendation: Accept approximate sync. Use `js_on_load` on the `gr.Audio` component to listen for the native HTML5 `play` event and start the CSS animation. The visual error will be <100ms which is acceptable.

2. **Optimal single-code preview approach**
   - What we know: Setting one level's code at all positions and zeros elsewhere produces audio, but zeros at other levels still contribute (code 0's embedding is not the zero vector).
   - What's unclear: Whether the "pure sound" of a single code is perceptually meaningful with background contributions from code 0 at other levels.
   - Recommendation: Start with the "one-level, zeros-elsewhere" approach. If code 0 contributions are problematic, try using `get_codes_from_indices()` to get only the target level's embedding and pass it directly to the decoder (bypassing other levels' contributions).

3. **Model type filtering in dropdown**
   - What we know: `ModelLibrary.list_all()` returns all models. `ModelEntry` has `has_analysis` and `n_active_components` fields but no explicit `model_type` field.
   - What's unclear: Whether the catalog already distinguishes v1 vs v2 models, or if we need to peek at `.distill` file contents.
   - Recommendation: Add a quick version/type check when loading the library -- either peek at the file (like `_detect_model_version()` in `cli/generate.py`) or filter by checking `n_active_components == 0` (VQ-VAE models have no PCA analysis). The cleanest solution is to use the existing `_detect_model_version()` pattern.

## Sources

### Primary (HIGH confidence)
- **Codebase analysis** -- `models/vqvae.py`: ConvVQVAE with encode(), quantize(), decode(), codes_to_embeddings(), QuantizerWrapper.get_output_from_indices()
- **Codebase analysis** -- `models/persistence.py`: LoadedVQModel, load_model_v2() with full reconstruction
- **Codebase analysis** -- `audio/spectrogram.py`: AudioSpectrogram with waveform_to_mel() and mel_to_waveform() (Griffin-Lim)
- **Codebase analysis** -- `audio/io.py`: load_audio() with format detection, resampling
- **Codebase analysis** -- `models/prior.py`: flatten_codes/unflatten_codes utilities, extract_code_sequences() pipeline
- **Codebase analysis** -- `inference/generation.py`: generate_audio_from_prior() showing full decode path
- **Codebase analysis** -- `ui/components/model_card.py`: HTML grid + hidden Textbox JS bridge pattern
- **Codebase analysis** -- `ui/tabs/generate_tab.py`: Tab builder pattern, gr.Audio usage, event wiring
- **Codebase analysis** -- `ui/tabs/library_tab.py`: Model dropdown pattern, load handler
- **Codebase analysis** -- `ui/state.py`: AppState singleton with loaded_vq_model field
- **Runtime verification** -- `ResidualVQ.codebooks` shape: `[num_quantizers, codebook_size, dim]` (verified via Python test)
- **Runtime verification** -- `get_codes_from_indices()` returns `[num_quantizers, B, seq_len, dim]` (verified)
- **Runtime verification** -- Spatial shape for 1s audio: H=8, W=6, indices shape [1, 48, 3] (verified)
- **Runtime verification** -- Gradio 6.6.0 installed, `gr.HTML` supports click/change/input/select events (verified)
- **Runtime verification** -- `server_functions` NOT available in gr.HTML 6.6.0 (verified via inspect.signature)

### Secondary (MEDIUM confidence)
- [Gradio HTML component docs](https://www.gradio.app/docs/gradio/html) -- Parameters, events, js_on_load
- [Gradio Audio component docs](https://www.gradio.app/docs/gradio/audio) -- autoplay, play/stop events, numpy tuple format
- [Gradio Custom CSS and JS guide](https://www.gradio.app/guides/custom-CSS-and-JS) -- JS event listener integration
- [Gradio Blocks and Event Listeners](https://www.gradio.app/guides/blocks-and-event-listeners) -- Event wiring patterns

### Tertiary (LOW confidence)
- None -- all critical claims verified via codebase analysis or runtime testing.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries already installed and used in project; no new dependencies
- Architecture: HIGH -- All backend patterns verified via codebase reading and runtime tests; UI pattern (HTML grid + JS bridge) proven in model_card.py
- Pitfalls: HIGH -- Spatial shape persistence, Griffin-Lim latency, and browser autoplay are well-understood from codebase experience

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable -- all dependencies are pinned, no moving targets)
