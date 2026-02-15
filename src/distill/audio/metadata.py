"""Format-aware audio metadata embedding via mutagen.

Embeds provenance metadata (model name, seed, preset) into exported audio
files using format-appropriate tag systems: ID3 for MP3, Vorbis Comments
for FLAC and OGG.  WAV files rely on sidecar JSON only (no embedded tags).

Design notes:
- Lazy imports for ``mutagen`` sub-modules (project pattern).
- Default artist tag "Distill Generator" for app-branded provenance.
- All fields are optional -- missing keys are silently skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distill.inference.export import ExportFormat

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_METADATA: dict[str, str] = {
    "artist": "Distill Generator",
    "album": "",
}


# ---------------------------------------------------------------------------
# Tag embedding
# ---------------------------------------------------------------------------


def embed_metadata(
    path: Path,
    format: "ExportFormat",
    metadata: dict,
) -> None:
    """Embed metadata tags into an audio file.

    Parameters
    ----------
    path : Path
        Path to the encoded audio file (must already exist on disk).
    format : ExportFormat
        The audio format of the file.
    metadata : dict
        Metadata dict with optional keys: ``artist``, ``album``, ``title``,
        ``seed``, ``model_name``, ``preset_name``.
    """
    from distill.inference.export import ExportFormat  # noqa: WPS433

    if format == ExportFormat.MP3:
        _embed_id3(path, metadata)
    elif format == ExportFormat.FLAC:
        _embed_flac(path, metadata)
    elif format == ExportFormat.OGG:
        _embed_ogg(path, metadata)
    # WAV: no-op (uses sidecar JSON only, per Phase 4 pattern)


def _embed_id3(path: Path, metadata: dict) -> None:
    """Embed ID3v2 tags into an MP3 file."""
    from mutagen.id3 import ID3, ID3NoHeaderError  # noqa: WPS433
    from mutagen.id3 import TALB, TIT2, TPE1, TXXX  # noqa: WPS433

    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()

    # Standard frames
    artist = metadata.get("artist", DEFAULT_METADATA["artist"])
    tags.add(TPE1(encoding=3, text=[artist]))

    album = metadata.get("album", DEFAULT_METADATA.get("album", ""))
    if album:
        tags.add(TALB(encoding=3, text=[album]))

    title = metadata.get("title", "")
    if title:
        tags.add(TIT2(encoding=3, text=[title]))

    # Custom TXXX frames for Distill provenance
    if metadata.get("seed") is not None:
        tags.add(TXXX(encoding=3, desc="DISTILL_SEED", text=[str(metadata["seed"])]))

    if metadata.get("model_name"):
        tags.add(TXXX(encoding=3, desc="DISTILL_MODEL", text=[metadata["model_name"]]))

    if metadata.get("preset_name"):
        tags.add(TXXX(encoding=3, desc="DISTILL_PRESET", text=[metadata["preset_name"]]))

    tags.save(str(path))


def _embed_flac(path: Path, metadata: dict) -> None:
    """Embed Vorbis Comment tags into a FLAC file."""
    from mutagen.flac import FLAC  # noqa: WPS433

    audio = FLAC(str(path))
    _set_vorbis_tags(audio, metadata)
    audio.save()


def _embed_ogg(path: Path, metadata: dict) -> None:
    """Embed Vorbis Comment tags into an OGG Vorbis file."""
    from mutagen.oggvorbis import OggVorbis  # noqa: WPS433

    audio = OggVorbis(str(path))
    _set_vorbis_tags(audio, metadata)
    audio.save()


def _set_vorbis_tags(audio: object, metadata: dict) -> None:
    """Apply Vorbis Comment key/value pairs to a mutagen file object.

    Works for both FLAC and OGG Vorbis (both use Vorbis Comments).
    Vorbis Comment values must be lists of strings.
    """
    artist = metadata.get("artist", DEFAULT_METADATA["artist"])
    audio["artist"] = [artist]  # type: ignore[index]

    album = metadata.get("album", DEFAULT_METADATA.get("album", ""))
    if album:
        audio["album"] = [album]  # type: ignore[index]

    title = metadata.get("title", "")
    if title:
        audio["title"] = [title]  # type: ignore[index]

    if metadata.get("seed") is not None:
        audio["distill_seed"] = [str(metadata["seed"])]  # type: ignore[index]

    if metadata.get("model_name"):
        audio["distill_model"] = [metadata["model_name"]]  # type: ignore[index]

    if metadata.get("preset_name"):
        audio["distill_preset"] = [metadata["preset_name"]]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def build_export_metadata(
    model_name: str,
    seed: int | None,
    preset_name: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """Build a standard metadata dict for audio export.

    Parameters
    ----------
    model_name : str
        Name of the model that generated the audio.
    seed : int | None
        Random seed used for generation.
    preset_name : str | None
        Name of the preset used, if any.
    overrides : dict | None
        Optional overrides for any metadata field (user can override any
        tag per locked decision).

    Returns
    -------
    dict
        Metadata dict ready to pass to :func:`embed_metadata`.
    """
    meta: dict = {
        "artist": DEFAULT_METADATA["artist"],
        "album": model_name,
        "title": "",
        "seed": seed,
        "model_name": model_name,
        "preset_name": preset_name or "",
    }

    if overrides:
        meta.update(overrides)

    return meta
