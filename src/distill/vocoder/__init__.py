"""Vocoder package for mel-to-waveform conversion.

Public API
----------
VocoderBase : Abstract base class for vocoder implementations.
BigVGANVocoder : BigVGAN-v2 universal vocoder (122M params, 44.1kHz).
MelAdapter : Converts VAE mel format to BigVGAN mel format.
HiFiGANVocoder : Per-model HiFi-GAN V2 vocoder (0.97M params, 48kHz).
get_vocoder : Factory function to obtain a ready-to-use vocoder.
resolve_vocoder : Resolve a user selection to a concrete vocoder + info dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from distill.vocoder.base import VocoderBase

if TYPE_CHECKING:
    from distill.models.persistence import LoadedModel

__all__ = [
    "VocoderBase",
    "BigVGANVocoder",
    "MelAdapter",
    "HiFiGANVocoder",
    "get_vocoder",
    "resolve_vocoder",
]


def __getattr__(name: str):
    """Lazy import for vocoder classes to avoid loading torch at package import."""
    if name == "BigVGANVocoder":
        from distill.vocoder.bigvgan_vocoder import BigVGANVocoder

        return BigVGANVocoder
    if name == "MelAdapter":
        from distill.vocoder.mel_adapter import MelAdapter

        return MelAdapter
    if name == "HiFiGANVocoder":
        from distill.vocoder.hifigan.vocoder import HiFiGANVocoder

        return HiFiGANVocoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_vocoder(
    vocoder_type: str = "bigvgan",
    device: str = "auto",
    tqdm_class: type | None = None,
    vocoder_state: dict | None = None,
) -> VocoderBase:
    """Get a vocoder instance by type.

    Parameters
    ----------
    vocoder_type : str
        One of "bigvgan" (universal) or "hifigan" (per-model).
    device : str
        Device preference: "auto", "cuda", "mps", or "cpu".
    tqdm_class : type | None
        Optional tqdm-compatible class for download progress display.
        Forwarded to BigVGANVocoder for weight downloads.
    vocoder_state : dict | None
        Per-model vocoder state dict. Required when ``vocoder_type``
        is ``"hifigan"``.

    Returns
    -------
    VocoderBase
        Ready-to-use vocoder instance on the selected device.
    """
    if vocoder_type == "bigvgan":
        # Lazy import to avoid loading torch at module import time
        from distill.vocoder.bigvgan_vocoder import BigVGANVocoder

        return BigVGANVocoder(device=device, tqdm_class=tqdm_class)
    elif vocoder_type == "hifigan":
        from distill.vocoder.hifigan.vocoder import HiFiGANVocoder

        if vocoder_state is None:
            raise ValueError(
                "vocoder_state is required for HiFi-GAN vocoder. "
                "Load a model with a trained per-model vocoder."
            )
        return HiFiGANVocoder(vocoder_state=vocoder_state, device=device)
    else:
        raise ValueError(
            f"Unknown vocoder type: {vocoder_type!r}. Use 'bigvgan' or 'hifigan'."
        )


def resolve_vocoder(
    selection: str,
    loaded_model: LoadedModel,
    device: str = "auto",
    tqdm_class: type | None = None,
) -> tuple[VocoderBase, dict]:
    """Resolve vocoder selection to a concrete vocoder instance.

    Auto-selection priority: per-model HiFi-GAN > BigVGAN universal.
    When a model has a trained per-model vocoder (``vocoder_state``),
    auto-selection prefers it over the universal BigVGAN vocoder.

    Parameters
    ----------
    selection : str
        User selection: ``"auto"``, ``"bigvgan"``, or ``"hifigan"``.
    loaded_model : LoadedModel
        The currently loaded model (checked for ``vocoder_state``).
    device : str
        Device preference forwarded to vocoder constructor.
    tqdm_class : type | None
        Optional tqdm-compatible class for download progress display.

    Returns
    -------
    tuple[VocoderBase, dict]
        ``(vocoder, info_dict)`` where *info_dict* contains:

        - ``name``: ``"bigvgan_universal"`` or ``"per_model_hifigan"``
        - ``selection``: the original *selection* string
        - ``reason``: human-readable explanation of auto-selection
        - ``warning`` (optional): present when vocoder trained < 20 epochs

    Raises
    ------
    ValueError
        If *selection* is ``"hifigan"`` but the model has no per-model
        vocoder, or if *selection* is unrecognised.
    """
    has_per_model = loaded_model.vocoder_state is not None

    if selection == "hifigan":
        if not has_per_model:
            raise ValueError(
                f"Model '{loaded_model.metadata.name}' has no trained "
                "per-model vocoder. Use --vocoder auto or bigvgan."
            )
        vocoder = get_vocoder(
            "hifigan",
            device=device,
            vocoder_state=loaded_model.vocoder_state,
        )
        training_meta = loaded_model.vocoder_state.get("training_metadata", {})
        epochs = training_meta.get("epochs", 0)
        info: dict = {
            "name": "per_model_hifigan",
            "selection": "hifigan",
            "reason": "explicit",
        }
        if epochs < 20:
            info["warning"] = (
                f"Trained for {epochs} epochs -- quality may be limited"
            )
        return vocoder, info

    if selection == "auto":
        if has_per_model:
            # Prefer per-model HiFi-GAN over BigVGAN universal
            vocoder = get_vocoder(
                "hifigan",
                device=device,
                vocoder_state=loaded_model.vocoder_state,
            )
            training_meta = loaded_model.vocoder_state.get(
                "training_metadata", {}
            )
            epochs = training_meta.get("epochs", 0)
            info = {
                "name": "per_model_hifigan",
                "selection": "auto",
                "reason": f"per-model vocoder ({epochs} epochs)",
            }
            if epochs < 20:
                info["warning"] = (
                    f"Trained for {epochs} epochs -- quality may be limited"
                )
            return vocoder, info

        # No per-model vocoder: fall back to BigVGAN universal
        return (
            get_vocoder("bigvgan", device=device, tqdm_class=tqdm_class),
            {
                "name": "bigvgan_universal",
                "selection": "auto",
                "reason": "no per-model vocoder",
            },
        )

    if selection == "bigvgan":
        return (
            get_vocoder("bigvgan", device=device, tqdm_class=tqdm_class),
            {
                "name": "bigvgan_universal",
                "selection": "bigvgan",
                "reason": "explicit",
            },
        )

    raise ValueError(
        f"Unknown vocoder selection: {selection!r}. "
        "Use 'auto', 'bigvgan', or 'hifigan'."
    )
