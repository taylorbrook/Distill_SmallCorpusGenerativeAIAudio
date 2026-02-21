"""HTML card grid renderer for the model library.

Renders :class:`~distill.library.catalog.ModelEntry` objects
as responsive CSS-grid HTML cards for the Library tab's card-grid view.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distill.library.catalog import ModelEntry


def _format_date(iso_date: str) -> str:
    """Format an ISO 8601 date string as a short human-readable date.

    Returns the original string if parsing fails.
    """
    if not iso_date:
        return "Unknown"
    try:
        # Take just the date portion (YYYY-MM-DD)
        return iso_date[:10]
    except (IndexError, TypeError):
        return str(iso_date)


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable units."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def render_single_card(model: "ModelEntry") -> str:
    """Render a single model card as an HTML string.

    Parameters
    ----------
    model : ModelEntry
        Model metadata entry from the library catalog.

    Returns
    -------
    str
        HTML string for one card element.
    """
    name = html.escape(model.name)
    dataset = html.escape(model.dataset_name or "Unknown dataset")
    date = _format_date(model.training_date)
    epochs = model.training_epochs
    components = model.n_active_components
    file_count = model.dataset_file_count
    size = _format_file_size(model.file_size_bytes)
    tags_str = ""
    if model.tags:
        tags_html = " ".join(
            f'<span style="background: #e0e7ff; color: #3730a3; '
            f'padding: 2px 8px; border-radius: 12px; font-size: 0.75em;">'
            f"{html.escape(t)}</span>"
            for t in model.tags[:5]
        )
        tags_str = f'<div style="margin-top: 8px;">{tags_html}</div>'

    return f"""\
<div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px;
            transition: box-shadow 0.2s, border-color 0.2s; cursor: pointer;"
     data-model-name="{name}"
     onclick="
       var modelName = this.getAttribute('data-model-name');
       var tb = document.querySelector('#model-card-selected-name textarea');
       if (tb) {{
         var nativeSet = Object.getOwnPropertyDescriptor(
           window.HTMLTextAreaElement.prototype, 'value').set;
         nativeSet.call(tb, modelName);
         tb.dispatchEvent(new Event('input', {{bubbles: true}}));
       }}
     "
     onmouseover="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)'; this.style.borderColor='#6366f1'"
     onmouseout="this.style.boxShadow='none'; this.style.borderColor='#ddd'">
    <h3 style="margin: 0 0 8px 0; font-size: 1.1em;">{name}</h3>
    <p style="margin: 4px 0; color: #666; font-size: 0.9em;">
        {dataset} &middot; {file_count} files
    </p>
    <p style="margin: 4px 0; color: #666; font-size: 0.9em;">
        Trained: {date} &middot; {epochs} epochs
    </p>
    <p style="margin: 4px 0; color: #666; font-size: 0.9em;">
        {components} components &middot; {size}
    </p>{tags_str}
</div>"""


def render_model_cards(models: list["ModelEntry"]) -> str:
    """Render a list of model entries as a responsive card grid.

    Parameters
    ----------
    models : list[ModelEntry]
        Model entries to render.  An empty list produces an
        informational message instead of an empty grid.

    Returns
    -------
    str
        Complete HTML string with CSS grid layout.
    """
    if not models:
        return (
            '<div style="text-align: center; padding: 40px; color: #888;">'
            "<p>No models in library yet.</p>"
            "<p>Train a model on the <strong>Train</strong> tab, "
            "then save it to build your library.</p>"
            "</div>"
        )

    cards = "\n".join(render_single_card(m) for m in models)
    return (
        '<div style="display: grid; '
        "grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); "
        'gap: 16px;">\n'
        f"{cards}\n"
        "</div>"
    )
