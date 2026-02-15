"""Data tab: file import, stats display, and waveform thumbnails.

Provides drag-and-drop file upload, folder browse, a stats panel
showing count/duration/sample rate, a waveform thumbnail gallery,
and clickable playback of individual files.

Uses ``matplotlib.use("Agg")`` before any pyplot imports (project
convention for headless compatibility).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gradio as gr  # noqa: E402

from distill.data.dataset import Dataset
from distill.data.summary import compute_summary
from distill.ui.state import app_state

logger = logging.getLogger(__name__)

# Supported audio file extensions
_AUDIO_EXTENSIONS = [".wav", ".aiff", ".mp3", ".flac", ".ogg"]


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or Xs for short durations."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _import_files(file_paths: list[Path]) -> tuple[str, list[str], str | None]:
    """Common import logic for both drag-and-drop and folder upload.

    Copies files to the datasets directory, creates a Dataset, computes
    summary with thumbnails, and updates app_state.

    Returns:
        Tuple of (stats_markdown, thumbnail_paths, None).
    """
    if not file_paths:
        return "No files provided.", [], None

    # Ensure import directory exists
    import_dir = app_state.datasets_dir / "imported"
    import_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to import directory
    copied: list[Path] = []
    for src in file_paths:
        src = Path(src)
        if not src.exists():
            continue
        # Only copy supported audio files
        if src.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue
        dest = import_dir / src.name
        # Avoid overwriting by appending a counter
        if dest.exists():
            stem = src.stem
            suffix = src.suffix
            counter = 1
            while dest.exists():
                dest = import_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            shutil.copy2(str(src), str(dest))
            copied.append(dest)
        except OSError as exc:
            logger.warning("Failed to copy %s: %s", src.name, exc)

    if not copied:
        return "No valid audio files found.", [], None

    # Create Dataset from the imported directory
    dataset = Dataset.from_directory(import_dir, name="imported")
    app_state.current_dataset = dataset

    # Compute summary with thumbnails
    summary = compute_summary(dataset, generate_thumbnails=True)
    app_state.current_summary = summary

    # Build stats markdown
    duration_str = _format_duration(summary.total_duration_seconds)
    stats = (
        f"**{summary.valid_file_count}** files | "
        f"**{duration_str}** total | "
        f"**{summary.dominant_sample_rate} Hz**"
    )

    # Build thumbnail list for gallery
    thumbnails = [str(p) for p in summary.thumbnail_paths.values() if p.exists()]

    return stats, thumbnails, None


def _handle_file_upload(files: list[str] | None) -> tuple:
    """Handle drag-and-drop file upload.

    Args:
        files: List of file path strings from gr.File upload.

    Returns:
        Tuple of (stats_md, gallery_images, audio_player,
                  stats_visible, gallery_visible).
    """
    if not files:
        return "", [], None, gr.update(visible=False), gr.update(visible=False)

    file_paths = [Path(f) for f in files]
    stats, thumbnails, audio = _import_files(file_paths)

    has_results = bool(thumbnails)
    return (
        stats,
        thumbnails,
        audio,
        gr.update(visible=has_results),
        gr.update(visible=has_results),
    )


def _handle_folder_upload(files: list[str] | None) -> tuple:
    """Handle folder browse upload.

    Args:
        files: List of file path strings from gr.UploadButton directory upload.

    Returns:
        Tuple of (stats_md, gallery_images, audio_player,
                  stats_visible, gallery_visible).
    """
    if not files:
        return "", [], None, gr.update(visible=False), gr.update(visible=False)

    file_paths = [Path(f) for f in files]
    stats, thumbnails, audio = _import_files(file_paths)

    has_results = bool(thumbnails)
    return (
        stats,
        thumbnails,
        audio,
        gr.update(visible=has_results),
        gr.update(visible=has_results),
    )


def _handle_thumbnail_click(evt: gr.SelectData) -> str | None:
    """Play the audio file corresponding to the clicked thumbnail.

    Args:
        evt: Gradio SelectData event with the selected index.

    Returns:
        File path string for gr.Audio playback, or None.
    """
    if app_state.current_dataset is None:
        return None

    idx = evt.index
    valid_files = app_state.current_dataset.valid_files

    if 0 <= idx < len(valid_files):
        audio_path = valid_files[idx]
        if audio_path.exists():
            return str(audio_path)

    return None


def build_data_tab() -> None:
    """Build the Data tab UI within the current Blocks context.

    Layout:
    - Header
    - Upload row (drag-and-drop file zone + folder browse button)
    - Stats panel (hidden until import)
    - Waveform thumbnail gallery (hidden until import)
    - Audio player (hidden until thumbnail click)
    """
    gr.Markdown("## Data")

    with gr.Row():
        file_upload = gr.File(
            label="Drop audio files here",
            file_count="multiple",
            file_types=_AUDIO_EXTENSIONS,
            scale=3,
        )
        folder_btn = gr.UploadButton(
            "Browse Folder",
            file_count="directory",
            scale=1,
        )

    stats_display = gr.Markdown(
        value="",
        visible=False,
    )

    thumbnail_gallery = gr.Gallery(
        label="Waveform Thumbnails",
        visible=False,
        columns=4,
        height=400,
    )

    audio_player = gr.Audio(
        label="Playback",
        visible=False,
        interactive=False,
    )

    # Wire events
    outputs = [stats_display, thumbnail_gallery, audio_player, stats_display, thumbnail_gallery]

    file_upload.upload(
        fn=_handle_file_upload,
        inputs=[file_upload],
        outputs=outputs,
    )

    folder_btn.upload(
        fn=_handle_folder_upload,
        inputs=[folder_btn],
        outputs=outputs,
    )

    thumbnail_gallery.select(
        fn=_handle_thumbnail_click,
        inputs=None,
        outputs=[audio_player],
    )
