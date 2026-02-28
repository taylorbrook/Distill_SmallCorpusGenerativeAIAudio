"""Interactive HTML code grid renderer for VQ-VAE code visualization.

Renders discrete VQ-VAE code indices as a CSS grid with per-cell coloring,
JavaScript click bridge (cell/column/row events dispatched to a hidden
Gradio Textbox), playhead animation, and sticky row labels.

The grid layout:
- Rows = quantizer levels (coarsest at top = level 0 = "Structure")
- Columns = spatial positions (H*W flattened, left to right = time)
- Fixed left column for level labels with Play buttons
- Horizontal scroll for long audio, row labels stay visible

Design notes:
- Uses matplotlib.cm.tab20 for cell coloring (already installed).
- JS onclick uses nativeSet + dispatchEvent pattern from model_card.py.
- html.escape() for any user-provided text.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Level label scheme
# ---------------------------------------------------------------------------

DEFAULT_LEVEL_LABELS: dict[int, list[str]] = {
    2: ["Structure", "Detail"],
    3: ["Structure", "Timbre", "Detail"],
    4: ["Structure", "Timbre", "Texture", "Detail"],
}
"""Cascading label scheme per user decision: Structure is always first,
Detail always last, Timbre and Texture fill the middle."""


def get_level_labels(
    num_quantizers: int,
    custom_labels: list[str] | None = None,
) -> list[str]:
    """Return level labels for a given number of quantizers.

    Parameters
    ----------
    num_quantizers:
        Number of RVQ levels (typically 2-4).
    custom_labels:
        Optional user-provided labels. If provided and length matches,
        used as-is.

    Returns
    -------
    list[str]
        Level labels from coarsest (index 0) to finest.
    """
    if custom_labels is not None and len(custom_labels) == num_quantizers:
        return custom_labels
    if num_quantizers in DEFAULT_LEVEL_LABELS:
        return DEFAULT_LEVEL_LABELS[num_quantizers]
    # Fallback for unusual quantizer counts
    return [f"Level {i}" for i in range(num_quantizers)]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _code_to_hex(code_index: int) -> str:
    """Map a code index to a hex background color via tab20 colormap."""
    rgba = cm.tab20(code_index % 20)
    return mcolors.to_hex(rgba)


def _text_color_for_bg(hex_color: str) -> str:
    """Return white or black text depending on background luminance."""
    rgb = mcolors.to_rgb(hex_color)
    # Relative luminance (ITU-R BT.709)
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "#ffffff" if lum < 0.5 else "#000000"


# ---------------------------------------------------------------------------
# JS bridge helper
# ---------------------------------------------------------------------------

_JS_DISPATCH = """\
var tb = document.querySelector('#code-grid-cell-clicked textarea');
if (tb) {
  var nativeSet = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype, 'value').set;
  nativeSet.call(tb, '{value}');
  tb.dispatchEvent(new Event('input', {bubbles: true}));
}"""


def _js_onclick(value: str) -> str:
    """Return an onclick handler string that dispatches value to the hidden textbox."""
    escaped = html.escape(value, quote=True)
    return _JS_DISPATCH.replace("{value}", escaped)


# ---------------------------------------------------------------------------
# Grid renderer
# ---------------------------------------------------------------------------


def render_code_grid(
    indices: "torch.Tensor | None",
    num_quantizers: int,
    codebook_size: int,
    spatial_shape: tuple[int, int],
    level_labels: list[str],
    selected_cell: tuple[int, int] | None,
    duration_s: float,
) -> str:
    """Render an interactive HTML code grid.

    Parameters
    ----------
    indices:
        Shape ``[1, seq_len, num_quantizers]`` code indices, or ``None``
        for the empty state.
    num_quantizers:
        Number of RVQ levels.
    codebook_size:
        Number of entries per codebook.
    spatial_shape:
        ``(H, W)`` spatial dimensions from encoder.
    level_labels:
        Labels for each quantizer level.
    selected_cell:
        ``(level, position)`` of the currently selected cell, or ``None``.
    duration_s:
        Duration of the encoded audio in seconds (for time markers
        and playhead animation).

    Returns
    -------
    str
        Complete HTML string with CSS grid layout, JS bridge, and
        playhead animation.
    """
    # Empty state
    if indices is None:
        return (
            '<div style="display: flex; align-items: center; justify-content: center; '
            'min-height: 200px; color: #888; font-size: 1.1em; text-align: center; '
            'padding: 40px;">'
            "<p>Upload an audio file and click Encode to see codes here.</p>"
            "</div>"
        )

    H, W = spatial_shape
    seq_len = H * W

    # Validate tensor shape
    if indices.shape[1] != seq_len or indices.shape[2] != num_quantizers:
        return (
            '<div style="color: #c00; padding: 20px;">'
            f"Error: indices shape {list(indices.shape)} does not match "
            f"expected [1, {seq_len}, {num_quantizers}]"
            "</div>"
        )

    # Column width
    col_w = 26  # px per data column
    label_w = 80  # px for sticky label column
    grid_data_width = seq_len * col_w

    # Build CSS
    css = f"""\
<style>
.code-grid-container {{
  position: relative;
  overflow-x: auto;
  border: 1px solid #444;
  border-radius: 6px;
  background: #1a1a2e;
}}
.code-grid {{
  display: grid;
  grid-template-columns: {label_w}px repeat({seq_len}, {col_w}px);
  gap: 1px;
  background: #333;
  width: fit-content;
  min-width: 100%;
}}
.code-grid-label {{
  position: sticky;
  left: 0;
  z-index: 5;
  background: #1a1a2e;
  color: #ccc;
  font-size: 11px;
  font-weight: 600;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2px 4px;
  min-height: 28px;
  border-right: 2px solid #555;
}}
.code-grid-label button {{
  background: #2d2d4e;
  color: #aaa;
  border: 1px solid #555;
  border-radius: 3px;
  font-size: 9px;
  padding: 1px 6px;
  cursor: pointer;
  margin-top: 2px;
}}
.code-grid-label button:hover {{
  background: #3d3d6e;
  color: #fff;
}}
.code-cell {{
  width: {col_w}px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  font-family: monospace;
  cursor: pointer;
  border: 1px solid transparent;
  transition: border-color 0.15s;
  box-sizing: border-box;
}}
.code-cell:hover {{
  border-color: #fff;
  opacity: 0.85;
}}
.code-cell-selected {{
  border: 3px solid #fff !important;
  box-shadow: 0 0 6px rgba(255,255,255,0.5);
}}
.code-grid-header {{
  position: sticky;
  left: 0;
  z-index: 5;
  background: #1a1a2e;
  height: 20px;
  border-right: 2px solid #555;
}}
.code-grid-col-header {{
  width: {col_w}px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 8px;
  color: #888;
  background: #222;
  cursor: pointer;
  font-family: monospace;
}}
.code-grid-col-header:hover {{
  color: #fff;
  background: #333;
}}
.playhead-line {{
  position: absolute;
  top: 20px;
  left: {label_w}px;
  width: 2px;
  height: calc(100% - 20px);
  background: rgba(255, 100, 100, 0.8);
  z-index: 10;
  pointer-events: none;
  animation: playhead-sweep var(--duration, 1s) linear forwards;
  animation-play-state: paused;
}}
.playhead-line.playing {{
  animation-play-state: running;
}}
@keyframes playhead-sweep {{
  from {{ left: {label_w}px; }}
  to {{ left: calc({label_w}px + {grid_data_width}px); }}
}}
</style>"""

    # Build JS for playhead control
    js = """\
<script>
function startPlayhead(durationSec) {
  var el = document.querySelector('.playhead-line');
  if (!el) return;
  el.style.setProperty('--duration', durationSec + 's');
  el.style.animation = 'none';
  el.offsetHeight; /* trigger reflow */
  el.style.animation = '';
  el.classList.add('playing');
}
function stopPlayhead() {
  var el = document.querySelector('.playhead-line');
  if (!el) return;
  el.classList.remove('playing');
  el.style.animation = 'none';
}
</script>"""

    # Build header row (time markers)
    header_cells = ['<div class="code-grid-header"></div>']  # empty corner
    time_per_pos = duration_s / seq_len if seq_len > 0 else 0
    for pos in range(seq_len):
        time_s = pos * time_per_pos
        # Show time marker every ~10 positions
        label = ""
        if pos % 10 == 0:
            label = f"{time_s:.1f}s"
        onclick = _js_onclick(f"col,{pos}")
        header_cells.append(
            f'<div class="code-grid-col-header" '
            f'onclick="{html.escape(onclick, quote=True)}" '
            f'title="Position {pos} (~{time_s:.2f}s)">'
            f"{html.escape(label)}</div>"
        )

    # Build data rows
    data_rows = []
    for level in range(num_quantizers):
        level_label = html.escape(level_labels[level]) if level < len(level_labels) else f"Level {level}"
        play_onclick = _js_onclick(f"row,{level}")

        # Row label cell
        row_label = (
            f'<div class="code-grid-label">'
            f'<span>{level_label}</span>'
            f'<button onclick="{html.escape(play_onclick, quote=True)}">Play</button>'
            f"</div>"
        )
        data_rows.append(row_label)

        # Data cells
        for pos in range(seq_len):
            code_idx = int(indices[0, pos, level].item())
            bg = _code_to_hex(code_idx)
            fg = _text_color_for_bg(bg)
            cell_onclick = _js_onclick(f"cell,{level},{pos}")

            # Check if selected
            selected_cls = ""
            if selected_cell is not None and selected_cell[0] == level and selected_cell[1] == pos:
                selected_cls = " code-cell-selected"

            data_rows.append(
                f'<div class="code-cell{selected_cls}" '
                f'style="background:{bg}; color:{fg};" '
                f'onclick="{html.escape(cell_onclick, quote=True)}" '
                f'title="Level {level} ({level_label}), Pos {pos}, Code {code_idx}">'
                f"{code_idx}</div>"
            )

    # Assemble full HTML
    grid_html = (
        f"{css}\n{js}\n"
        f'<div class="code-grid-container">\n'
        f'  <div class="code-grid">\n'
        f'    {"".join(header_cells)}\n'
        f'    {"".join(data_rows)}\n'
        f"  </div>\n"
        f'  <div class="playhead-line" style="--duration: {duration_s}s;"></div>\n'
        f"</div>"
    )

    return grid_html
