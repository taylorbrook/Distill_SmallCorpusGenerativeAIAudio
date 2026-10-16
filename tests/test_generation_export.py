"""Generation output shapes and export formats."""

from __future__ import annotations

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from distill.controls.mapping import SliderState, sliders_to_latent
from distill.inference.export import export_ogg
from distill.inference.generation import GenerationConfig, GenerationPipeline
from distill.inference.spatial import SpatialConfig, SpatialMode
from distill.models.persistence import load_model
from distill.ui.tabs.generate_tab import _to_gradio_audio


def test_stereo_slider_generation_reaches_gradio_as_samples_by_channels(trained):
    loaded = load_model(trained["saved_model_path"])
    n = loaded.analysis.n_active_components
    z = sliders_to_latent(SliderState([3] + [0] * (n - 1), n), loaded.analysis)
    pipeline = GenerationPipeline(loaded.model, loaded.spectrogram, torch.device("cpu"))

    result = pipeline.generate(
        GenerationConfig(
            duration_s=1.5,
            seed=1,
            latent_vector=z,
            spatial=SpatialConfig(mode=SpatialMode.STEREO, width=0.8),
        )
    )
    assert result.audio.shape[0] == 2

    # Run it through gr.Audio's real postprocess (raised struct.error
    # when handed the pipeline's [channels, samples] array directly)
    out = gr.Audio(type="numpy").postprocess(
        _to_gradio_audio(result.sample_rate, result.audio)
    )
    info = sf.info(out.path)
    assert info.channels == 2
    assert info.frames == result.audio.shape[1]

    mono = np.zeros(100, dtype=np.float32)
    assert _to_gradio_audio(48_000, mono)[1].shape == (100,)


def test_ogg_export_longer_than_libsndfile_single_write_limit(tmp_path):
    # A single Vorbis write of >= 2**21 frames (~43.7 s at 48 kHz)
    # segfaults libsndfile 1.2.2; the exporter must write in blocks.
    seconds = 45
    t = np.arange(seconds * 48_000) / 48_000
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    path = export_ogg(np.stack([tone, tone * 0.5]), tmp_path / "long.ogg")

    info = sf.info(str(path))
    assert info.channels == 2
    assert info.frames == seconds * 48_000
