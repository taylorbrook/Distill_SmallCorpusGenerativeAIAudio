# Phase 17: Code Editing - Research

**Researched:** 2026-02-27
**Domain:** Interactive VQ-VAE code manipulation (edit, swap, blend) in Gradio UI
**Confidence:** HIGH

## Summary

Phase 17 extends the existing Codes tab (Phase 16) with interactive editing capabilities: double-click cell editing, rectangular region selection for swapping between two audio files, embedding-space blending with per-region control, and full undo/redo. The existing infrastructure is well-suited for this: the HTML code grid already has a JS-to-Python bridge (nativeSet + dispatchEvent pattern), the `codes.py` backend has encode/decode/preview functions, and the `vector-quantize-pytorch` library (v1.27.21) exposes per-level codebook embeddings via `get_codes_from_indices()` and the `codebooks` property -- both essential for embedding-space blending.

The core technical challenges are: (1) adding double-click and drag-select JS interactions to the server-rendered HTML grid without breaking existing click previews, (2) implementing embedding interpolation using the library's `codebooks` tensor for nearest-neighbor snapping, (3) managing an undo/redo stack in module-level state (project pattern -- no `gr.State` used anywhere), and (4) expanding the single-grid Codes tab into a dual-grid layout for swap/blend operations while keeping the existing single-file workflow intact.

**Primary recommendation:** Extend the existing `code_grid.py` renderer and `codes_tab.py` handler with new JS event types (dblclick for edit, mousedown/mousemove for selection), a new `code_editing.py` backend module for swap/blend/undo operations, and a sub-tab or mode-switcher within the Codes tab for Edit vs. Swap/Blend views.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Double-click a cell to enter inline edit mode — type a new codebook index directly in-place
- Input is constrained to valid codebook indices only (0 to codebook_size-1), impossible to enter out-of-range values
- No auto-play after editing — user manually triggers playback when ready to hear the change
- Two audio files displayed as side-by-side grids, each with their own upload/encode controls
- Click-drag rectangle on a grid to select a region for swapping (time positions x quantizer levels)
- After swap, both grids update in place simultaneously showing the new code values (with brief highlight animation)
- Per-region + global blend control: select a region first, then a slider controls blend ratio for that region; also a global slider for whole-grid blending
- Multiple active regions can have different blend ratios, all adjustable before committing
- Real-time grid updates as the slider moves (cells update continuously during drag)
- Show both the snapped codebook index AND interpolation distance/confidence
- Undo/Redo buttons in the UI toolbar plus Ctrl+Z / Ctrl+Y keyboard shortcuts
- Silent operation — no visible history panel or step counter, just the buttons and shortcuts
- Undo history resets when a new audio file is uploaded (new file = fresh start)

### Claude's Discretion
- Whether multi-select for bulk cell editing is worth the complexity
- Region size matching policy for swaps (exact match vs. truncate/pad)
- Decode trigger pattern (manual button vs. auto-decode after each edit)
- Exact highlight animation for swapped regions
- Keyboard shortcut conflict resolution with Gradio/browser defaults
- Side-by-side grid layout details (sizing, spacing, responsive behavior)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CODE-04 | User can edit individual code cells (change codebook index) | Double-click JS event on grid cells, inline `<input type="number">` with min=0 max=codebook_size-1, JS dispatch of `edit,{level},{pos},{new_value}` to hidden textbox bridge, Python handler updates `_current_encode["indices"]` tensor and re-renders grid |
| CODE-05 | User can swap code regions between two encoded audio files | Dual-grid layout with two encode states, rectangular drag-select via mousedown/mousemove/mouseup JS, region coordinates dispatched to Python, tensor slice swap on both index tensors, simultaneous grid re-render with highlight CSS class |
| CODE-06 | User can blend codes in embedding space (smoother than index swapping) | Use `rvq.get_codes_from_indices()` to get per-level embeddings `[Q, B, N, D]`, interpolate between two files' embeddings at blend ratio, find nearest codebook entry per level via `torch.cdist()` against `rvq.codebooks` tensor, snap to index, compute distance metric for confidence display |
| CODE-08 | Code edits support undo/redo | Module-level undo stack (list of `torch.Tensor` snapshots of indices), push-before-edit pattern, Undo/Redo buttons wired to stack pop/push, Ctrl+Z/Y via JS keydown listener dispatching to hidden textbox bridge, stack clears on new file upload |
| UI-01 | New Codes tab in Gradio UI for code visualization and editing | Codes tab already exists from Phase 16; extend with edit toolbar (Undo/Redo/Decode buttons), mode switcher or sub-tabs (Single File / Swap & Blend), dual-grid layout for swap/blend mode, blend ratio sliders per region |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gradio | 6.6.0 | UI framework (already installed) | Project UI framework, all tabs built with it |
| torch | (project version) | Tensor operations for indices, embeddings | All model inference uses PyTorch |
| vector-quantize-pytorch | 1.27.21 | RVQ codebook access, `get_codes_from_indices`, `codebooks` property | Already used for quantization; has the exact APIs needed for embedding blending |
| matplotlib | (project version) | tab20 colormap for grid cell coloring | Already used in code_grid.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (project version) | Waveform arrays for decoded audio | Already used in decode_code_grid return |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Module-level undo stack | gr.State for undo history | gr.State not used anywhere in project; module-level state is the established pattern (single-user desktop app) |
| Server-rendered HTML grid | Gradio custom component | Far more complex, no precedent in project, existing HTML+JS bridge works well |
| torch.cdist for nearest neighbor | Manual L2 loop | cdist is vectorized and handles batched distance computation efficiently |

**Installation:** No new dependencies needed. All libraries already installed.

## Architecture Patterns

### Recommended Project Structure
```
src/distill/
├── inference/
│   ├── codes.py              # Existing: encode/decode/preview (extend with blend/swap)
│   └── code_editing.py       # NEW: swap_regions, blend_embeddings, undo stack ops
├── ui/
│   ├── components/
│   │   └── code_grid.py      # Existing: extend with dblclick, drag-select, edit input JS
│   └── tabs/
│       └── codes_tab.py      # Existing: extend with dual-grid, blend controls, undo/redo
```

### Pattern 1: JS Bridge Extension (double-click edit)
**What:** Add `ondblclick` handler to grid cells that replaces the cell content with an `<input>` element, constrained to valid indices. On Enter/blur, dispatch `edit,{level},{pos},{newValue}` to the existing hidden textbox bridge.
**When to use:** For all inline cell editing interactions.
**Example:**
```javascript
// Extend existing onclick pattern -- dblclick spawns inline input
cell.ondblclick = function(e) {
  e.stopPropagation();
  var oldVal = this.textContent;
  var inp = document.createElement('input');
  inp.type = 'number';
  inp.min = 0;
  inp.max = CODEBOOK_SIZE - 1;
  inp.value = oldVal;
  inp.style.cssText = 'width:100%;height:100%;border:none;text-align:center;font-size:9px;';
  this.textContent = '';
  this.appendChild(inp);
  inp.focus();
  inp.select();
  inp.onblur = function() { commitEdit(level, pos, inp.value); };
  inp.onkeydown = function(e) {
    if (e.key === 'Enter') { inp.blur(); }
    if (e.key === 'Escape') { cancelEdit(cell, oldVal); }
  };
};
```

### Pattern 2: Rectangular Drag Selection (for swap/blend regions)
**What:** On mousedown, record start cell (level, pos). On mousemove, highlight rectangle of cells. On mouseup, dispatch `select,{startLevel},{startPos},{endLevel},{endPos}` to bridge. CSS class `.code-cell-in-selection` highlights selected cells.
**When to use:** For region selection in swap and blend operations.
**Example:**
```javascript
// Track drag state across grid cells
var dragState = { active: false, startLevel: -1, startPos: -1 };
cell.onmousedown = function(e) {
  if (e.button !== 0) return;
  dragState = { active: true, startLevel: level, startPos: pos };
  e.preventDefault(); // prevent text selection
};
cell.onmouseenter = function() {
  if (!dragState.active) return;
  highlightRectangle(dragState.startLevel, dragState.startPos, level, pos);
};
document.onmouseup = function() {
  if (!dragState.active) return;
  // dispatch selection to Python
  dispatchBridge('select,' + dragState.startLevel + ',' + dragState.startPos + ',' + endLevel + ',' + endPos);
  dragState.active = false;
};
```

### Pattern 3: Embedding Space Blending
**What:** For two encoded files, get per-level embedding vectors via `rvq.get_codes_from_indices()`, linearly interpolate at the given ratio, then find the nearest codebook entry per level using `torch.cdist()`. Return both the snapped index and the L2 distance to the nearest entry.
**When to use:** For the blend slider interaction.
**Example:**
```python
def blend_embeddings(
    indices_a: torch.Tensor,  # [1, seq_len, num_q]
    indices_b: torch.Tensor,  # [1, seq_len, num_q]
    ratio: float,             # 0.0 = all A, 1.0 = all B
    region: tuple | None,     # (start_level, start_pos, end_level, end_pos) or None for global
    loaded: LoadedVQModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (blended_indices, distances) where distances shows confidence."""
    rvq = loaded.model.quantizer.rvq

    # Get per-level embeddings: [num_q, 1, seq_len, dim]
    codes_a = rvq.get_codes_from_indices(indices_a)
    codes_b = rvq.get_codes_from_indices(indices_b)

    # Interpolate in embedding space
    blended = (1 - ratio) * codes_a + ratio * codes_b  # [num_q, 1, seq_len, dim]

    # Snap to nearest codebook entry per level
    codebooks = rvq.codebooks  # [num_q, codebook_size, dim]
    blended_indices = torch.empty_like(indices_a)
    distances = torch.empty(indices_a.shape, dtype=torch.float32)

    for q in range(rvq.num_quantizers):
        # [1, seq_len, dim] vs [codebook_size, dim]
        dists = torch.cdist(blended[q], codebooks[q].unsqueeze(0))  # [1, seq_len, codebook_size]
        min_dists, min_indices = dists.min(dim=-1)
        blended_indices[0, :, q] = min_indices[0]
        distances[0, :, q] = min_dists[0]

    return blended_indices, distances
```

### Pattern 4: Undo/Redo Stack (Module-Level State)
**What:** A simple list-based undo stack storing deep copies of the indices tensor. Push before every edit. Pop on undo, push popped state to redo stack. Clear both stacks on new file upload.
**When to use:** For all code editing operations.
**Example:**
```python
# Module-level state (follows codes_tab.py pattern)
_undo_stack: list[torch.Tensor] = []
_redo_stack: list[torch.Tensor] = []

def push_undo(indices: torch.Tensor) -> None:
    """Save current state before an edit."""
    _undo_stack.append(indices.clone())
    _redo_stack.clear()  # new edit invalidates redo history

def undo(current_indices: torch.Tensor) -> torch.Tensor | None:
    """Pop previous state. Returns None if nothing to undo."""
    if not _undo_stack:
        return None
    _redo_stack.append(current_indices.clone())
    return _undo_stack.pop()

def redo(current_indices: torch.Tensor) -> torch.Tensor | None:
    if not _redo_stack:
        return None
    _undo_stack.append(current_indices.clone())
    return _redo_stack.pop()

def clear_history() -> None:
    _undo_stack.clear()
    _redo_stack.clear()
```

### Anti-Patterns to Avoid
- **Modifying `_current_encode["indices"]` in-place without cloning first:** Tensors share memory; must `.clone()` before pushing to undo stack, otherwise all stack entries point to same memory.
- **Re-encoding audio on every edit:** Edits modify the index tensor directly -- never re-run the encoder. Only decode (indices -> audio) when user requests playback.
- **Using gr.State for undo history:** Project pattern is module-level state. gr.State would be inconsistent and adds Gradio serialization overhead for tensor data.
- **Blocking UI on blend slider movement:** Blend computation involves codebook lookups and distance calculations. For real-time slider updates, the grid HTML update must be fast. Pre-compute embeddings for both files once, then only the interpolation + nearest-neighbor step runs on each slider tick.
- **Full grid re-render on every slider tick:** For real-time blend updates, consider updating only the affected cells (region) rather than re-rendering the entire HTML grid. However, given the grid is typically small (2-4 rows x ~48 columns), full re-render may be fast enough.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Nearest codebook entry lookup | Manual L2 distance loops | `torch.cdist()` + `.min()` | Vectorized, handles batched computation, numerically stable |
| Per-level embedding extraction | Manual codebook index lookup | `rvq.get_codes_from_indices(indices)` | Library method handles all reshaping, mask handling, projection |
| Codebook embedding access | Reaching into internal `_codebook.embed` | `rvq.codebooks` property | Clean API, handles rearrange from `1 c d` to `c d`, stacks across quantizers |
| Index-to-summed-embedding | Sum-and-project manually | `rvq.get_output_from_indices(indices)` | Handles projection layer, dropout masking, proper summation |

**Key insight:** The `vector-quantize-pytorch` library already provides the exact primitives needed for embedding blending. `get_codes_from_indices()` returns per-level embeddings (not summed), and `codebooks` gives direct access to the codebook vectors for nearest-neighbor search. No need to reach into private internals.

## Common Pitfalls

### Pitfall 1: Tensor In-Place Mutation Breaking Undo
**What goes wrong:** Editing `_current_encode["indices"][0, pos, level] = new_value` mutates the tensor. If the undo stack holds a reference (not a clone), the "previous" state is also mutated.
**Why it happens:** PyTorch tensors are reference types. `list.append(tensor)` does not copy.
**How to avoid:** Always `_undo_stack.append(indices.clone())` before any mutation. Use `indices = indices.clone()` to create a fresh tensor for the edit, then replace the reference in `_current_encode`.
**Warning signs:** Undo produces the same state as current, or all undo states are identical.

### Pitfall 2: Double-Click vs. Single-Click Conflict
**What goes wrong:** The existing grid uses `onclick` for cell preview (play codebook entry audio). Adding `ondblclick` for editing creates a timing conflict -- single click fires before double click is detected.
**Why it happens:** Browser fires click event on first click, then click + dblclick on second click.
**How to avoid:** Use a 200ms click delay pattern: on first click, set a timeout. If a second click arrives before timeout, cancel the timeout and trigger dblclick handler. If timeout fires, trigger single-click (preview) handler. Alternatively, use the existing single-click for preview and a separate "Edit mode" toggle button.
**Warning signs:** Every double-click also triggers a preview audio playback.

### Pitfall 3: Gradio Hidden Textbox Bridge Race Conditions
**What goes wrong:** Rapid JS dispatches (e.g., real-time blend slider sending many values) queue up in Gradio's event system. If the Python handler takes time (e.g., decoding), earlier events may still be processing when later ones arrive.
**Why it happens:** Gradio 6 processes events sequentially per component by default. The hidden textbox `.change()` handler fires for every value change.
**How to avoid:** For blend slider, debounce on the JS side (only dispatch after slider stops moving for 50-100ms). For the Python handler, check a timestamp or sequence number to skip stale events. Alternatively, use Gradio's `every=` parameter or a separate `gr.Slider` component with built-in throttling.
**Warning signs:** Grid updates lag behind slider movement, or updates arrive out of order.

### Pitfall 4: Blend Distance Interpretation
**What goes wrong:** Raw L2 distances from `torch.cdist` are in the codebook embedding space, which users cannot interpret. Distances vary with codebook dimension and training.
**Why it happens:** Absolute distance magnitudes depend on codebook_dim (128 by default) and how spread-out the learned codebook is.
**How to avoid:** Normalize distances relative to the codebook's own statistics. For each level, compute the mean pairwise distance between codebook entries as a reference. Display confidence as `1 - (distance / mean_pairwise_distance)` clamped to [0, 1], or use a percentile rank. Show as a color overlay (green = close match, red = far from any entry).
**Warning signs:** All distances look similar, or users cannot tell when a blend landed near vs. far from a real codebook entry.

### Pitfall 5: Side-by-Side Grid Overflow
**What goes wrong:** Two grids side-by-side, each with horizontal scroll, overflow the viewport on smaller screens. The Gradio container doesn't handle this gracefully.
**Why it happens:** Each grid has width = label_col + seq_len * 26px. Two grids double the minimum width.
**How to avoid:** Use `gr.Row()` with `equal_height=True` and each grid in a `gr.Column(scale=1)`. Each grid container should have `overflow-x: auto` independently. Set `max-width: 50%` on each column. For very long audio, consider a vertical (stacked) layout option.
**Warning signs:** Horizontal scrollbar on the page level, or grids overlap.

### Pitfall 6: Keyboard Shortcut Conflicts with Browser/Gradio
**What goes wrong:** Ctrl+Z in the browser triggers the browser's built-in undo (e.g., undoing text input in Gradio textboxes). Ctrl+Y may trigger redo or history in some browsers.
**Why it happens:** Browser keyboard shortcuts have priority over JavaScript event handlers unless explicitly prevented.
**How to avoid:** Attach the keydown listener to the code grid container specifically, not `document`. Only intercept Ctrl+Z/Y when the grid container (or its parent) has focus. Use `e.preventDefault()` and `e.stopPropagation()` to suppress browser defaults. Add `tabindex="0"` to the grid container so it can receive focus.
**Warning signs:** Ctrl+Z undoes text in the model name dropdown instead of undoing a code edit.

## Code Examples

### Verified: Existing JS Bridge Pattern (from code_grid.py)
```python
# Source: src/distill/ui/components/code_grid.py lines 93-106
_JS_DISPATCH = """\
var tb = document.querySelector('#code-grid-cell-clicked textarea');
if (tb) {
  var nativeSet = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype, 'value').set;
  nativeSet.call(tb, '{value}');
  tb.dispatchEvent(new Event('input', {bubbles: true}));
}"""
```

### Verified: Per-Level Embedding Extraction (from vector-quantize-pytorch)
```python
# Source: .venv/Lib/site-packages/vector_quantize_pytorch/residual_vq.py lines 305-358
# get_codes_from_indices returns shape [num_quantizers, batch, seq_len, dim]
# For uniform codebook (our case):
#   all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)
```

### Verified: Codebooks Property (from vector-quantize-pytorch)
```python
# Source: .venv/Lib/site-packages/vector_quantize_pytorch/residual_vq.py lines 293-303
@property
def codebooks(self):
    codebooks = [layer._codebook.embed for layer in self.layers]
    codebooks = tuple(rearrange(codebook, '1 ... -> ...') for codebook in codebooks)
    if not self.uniform_codebook_size:
        return codebooks
    codebooks = torch.stack(codebooks)
    return codebooks
# Returns: [num_quantizers, codebook_size, dim] for uniform codebook sizes (our case)
```

### Verified: Existing Module-Level State Pattern (from codes_tab.py)
```python
# Source: src/distill/ui/tabs/codes_tab.py lines 41-45
_current_encode: dict | None = None
"""Stores the latest encode result (indices, spatial_shape, mel_shape, etc.)."""

_current_labels: list[str] | None = None
"""Stores the current level labels for grid re-rendering."""
```

### Verified: Existing Grid Rendering Pattern (from code_grid.py)
```python
# Source: src/distill/ui/components/code_grid.py lines 114-151
def render_code_grid(
    indices, num_quantizers, codebook_size, spatial_shape,
    level_labels, selected_cell, duration_s,
) -> str:
    # Returns complete HTML string with CSS grid, JS bridge, playhead animation
```

### Proposed: Cell Edit Handler
```python
def _handle_cell_edit(edit_info: str):
    """Parse edit from JS: 'edit,{level},{pos},{new_value}'"""
    global _current_encode
    if not edit_info or _current_encode is None:
        return [gr.update(), gr.update()]

    parts = edit_info.strip().split(",")
    if parts[0] != "edit" or len(parts) < 4:
        return [gr.update(), gr.update()]

    level, pos, new_val = int(parts[1]), int(parts[2]), int(parts[3])

    # Validate
    if new_val < 0 or new_val >= _current_encode["codebook_size"]:
        return [gr.update(), gr.update()]

    # Push undo before edit
    from distill.inference.code_editing import push_undo
    push_undo(_current_encode["indices"])

    # Apply edit (clone to avoid mutation)
    indices = _current_encode["indices"].clone()
    indices[0, pos, level] = new_val
    _current_encode["indices"] = indices

    # Re-render grid
    grid_html = render_code_grid(...)
    return [grid_html, ""]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom VQ implementation | lucidrains/vector-quantize-pytorch | Project decision (v1.1) | `codebooks` property and `get_codes_from_indices` provide clean embedding access for blending |
| Gradio 4 custom components | Gradio 6 HTML + JS bridge | Gradio 6 release | `gr.HTML` with inline JS onclick is the established pattern; no need for custom Svelte components |
| Continuous latent blending (v1.0) | Discrete code manipulation (v1.1) | Architecture switch | Blending now requires nearest-neighbor snapping to discrete codebook entries instead of direct interpolation |

**Deprecated/outdated:**
- v1.0 `BlendEngine` / continuous latent interpolation: Not applicable to VQ-VAE discrete codes. Embedding blending for VQ-VAE requires per-level codebook lookup, not direct latent space interpolation.

## Open Questions

1. **Click delay vs. Edit mode toggle for double-click**
   - What we know: Browser double-click fires two click events first. A 200ms delay would slow down single-click preview response.
   - What's unclear: Whether the 200ms delay is noticeable enough to hurt UX for cell preview.
   - Recommendation: Implement the click delay pattern (200ms). If testing reveals it feels sluggish, fall back to a toggle button ("Edit Mode" on/off). The delay approach is more natural since users expect double-click = edit.

2. **Real-time blend slider performance**
   - What we know: Blend computation is: interpolate embeddings + nearest-neighbor per level. For typical grids (48 positions, 3 levels, codebook_size 64-256), this is fast (<10ms).
   - What's unclear: Whether Gradio's event round-trip (JS -> Python -> re-render HTML -> JS) is fast enough for "real-time" feel during slider drag.
   - Recommendation: Use a `gr.Slider` component for blend ratio (Gradio handles debouncing internally). Pre-compute embeddings for both files once on encode. If round-trip is too slow, degrade gracefully: update grid HTML only, defer decode-to-audio until slider release.

3. **Multi-select for bulk cell editing**
   - What we know: Rectangular drag-select is already needed for swap/blend regions. Extending it to bulk-set all selected cells to a given value is minimal additional code.
   - What's unclear: Whether users would actually use bulk edit (set 10 cells to the same codebook index) vs. just editing one at a time.
   - Recommendation: Include multi-select. The drag-select infrastructure is already being built for swap/blend. Add a "Set Selected" input that applies a single codebook index to all cells in the selection. Low marginal complexity, good power-user feature.

4. **Region size matching for swaps**
   - What we know: Two audio files may produce different spatial shapes (different durations -> different W in spatial_shape). Regions selected on each grid may differ in size.
   - What's unclear: Whether mismatched region sizes produce acceptable audio.
   - Recommendation: Require exact-size match for swap. If grids have different dimensions, only allow swapping the intersection (min of each dimension). This avoids ambiguity about padding/truncation and keeps the operation predictable. Show a warning if selected regions don't match.

5. **Undo granularity for blend operations**
   - What we know: User can adjust blend slider back and forth many times before "committing." Should each slider tick push an undo state?
   - What's unclear: Whether unlimited undo states from slider movement would be useful or just noise.
   - Recommendation: Push undo state only once when a blend operation starts (capturing pre-blend state). Slider adjustments are considered part of the same operation. A "Commit Blend" button finalizes the current blend. Undo after commit returns to the pre-blend state, not to intermediate slider positions.

## Discretion Recommendations

Based on the Claude's Discretion items from CONTEXT.md:

1. **Multi-select for bulk editing:** YES, include it. Drag-select is needed anyway for swap/blend; adding a "Set Selected" field is trivial overhead.

2. **Region size matching for swaps:** Exact match required. Swap only the intersection rectangle. Display warning if grids have different spatial dimensions.

3. **Decode trigger pattern:** Manual decode button. Consistent with user decision "No auto-play after editing." A prominent "Decode" button (already exists from Phase 16) decodes the current grid state to audio. This is simple and predictable.

4. **Highlight animation for swapped regions:** CSS transition: 0.5s yellow background flash on swapped cells, then fade to the new code's color. Use a `.code-cell-swapped` class that auto-removes after animation completes via `animationend` event.

5. **Keyboard shortcut conflict resolution:** Bind Ctrl+Z/Y only when the code grid container has focus (`tabindex="0"` + `keydown` listener on container). Use `e.preventDefault()` to suppress browser defaults. Do NOT bind globally on `document` -- this would interfere with Gradio textbox undo.

6. **Side-by-side grid layout:** Two `gr.Column(scale=1)` inside a `gr.Row()`. Each column contains its own upload/encode controls and grid. Each grid has independent horizontal scroll (`overflow-x: auto`). On the blend view, a shared controls row spans both columns below (blend slider, commit button).

## Sources

### Primary (HIGH confidence)
- `src/distill/ui/components/code_grid.py` - Existing grid renderer, JS bridge pattern, CSS structure
- `src/distill/ui/tabs/codes_tab.py` - Existing Codes tab structure, event handlers, module-level state pattern
- `src/distill/inference/codes.py` - Existing encode/decode/preview backend functions
- `src/distill/models/vqvae.py` - ConvVQVAE architecture, `codes_to_embeddings()`, `quantize()`, spatial shape handling
- `.venv/Lib/site-packages/vector_quantize_pytorch/residual_vq.py` - ResidualVQ `codebooks` property, `get_codes_from_indices()`, `get_output_from_indices()` implementations
- `src/distill/ui/app.py` - Tab layout, cross-tab wiring patterns
- `src/distill/ui/state.py` - AppState singleton pattern, `loaded_vq_model` field

### Secondary (MEDIUM confidence)
- Gradio 6.6.0 event handling - `gr.Slider.change()`, `gr.HTML`, hidden textbox bridge pattern (verified by existing project usage)
- Browser double-click event timing - standard DOM behavior, well-documented

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in use, APIs verified by reading source code
- Architecture: HIGH - extends well-established patterns from Phase 16, all extension points identified
- Pitfalls: HIGH - identified from direct code analysis of existing JS bridge, tensor mutation patterns, and Gradio event handling

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (stable -- no library upgrades expected)
