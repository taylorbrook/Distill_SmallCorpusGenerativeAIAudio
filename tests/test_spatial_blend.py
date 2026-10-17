"""Blending and binaural rendering."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from distill.controls.mapping import SliderState, sliders_to_latent
from distill.inference.blending import BlendEngine
from distill.inference.generation import GenerationConfig, GenerationPipeline
from distill.inference.spatial import SpatialConfig, SpatialMode, apply_spatial
from distill.models.persistence import ModelMetadata, load_model

CPU = torch.device("cpu")


def test_blend_mixes_each_models_output_and_keeps_the_settings(trained):
    loaded = load_model(trained["saved_model_path"])
    n = loaded.analysis.n_active_components
    engine = BlendEngine()
    engine.add_model(
        loaded.model,
        loaded.spectrogram,
        loaded.analysis,
        loaded.metadata,
        CPU,
        weight=70,
    )
    # Same weights but no analysis: this slot decodes the zero latent
    engine.add_model(
        loaded.model,
        loaded.spectrogram,
        None,
        ModelMetadata(name="plain"),
        CPU,
        weight=30,
    )
    positions = [4] + [0] * (n - 1)
    config = GenerationConfig(
        duration_s=1.0,
        seed=5,
        spatial=SpatialConfig(mode=SpatialMode.STEREO, width=0.8),
        evolution_amount=0.0,
        export_format="flac",
    )

    result = engine.blend_generate(positions, config)

    assert result.audio.shape[0] == 2  # stereo was dropped before
    assert result.channels == 2
    assert result.config.export_format == "flac"
    assert result.config.evolution_amount == 0.0

    # Each model generates on its own; the outputs are mixed 70/30
    pipeline = GenerationPipeline(loaded.model, loaded.spectrogram, CPU)
    z = sliders_to_latent(SliderState(positions, n), loaded.analysis)
    a = pipeline.generate(replace(config, latent_vector=z)).audio
    b = pipeline.generate(
        replace(config, latent_vector=np.zeros(loaded.model.latent_dim, np.float32))
    ).audio
    np.testing.assert_allclose(result.audio, 0.7 * a + 0.3 * b, atol=1e-5)


# ---------------------------------------------------------------------------
# Binaural (hidden in the UI/CLI until a SOFA file ships)
# ---------------------------------------------------------------------------

_AZIMUTHS = [0.0, 30.0, 90.0, 270.0, 330.0]


@pytest.fixture
def sofa_file(tmp_path, monkeypatch):
    """A tiny HRTF set: each ear's gain encodes the source direction.

    SOFA azimuths run counter-clockwise, so 90 deg is the listener's left:
    the left ear gets ``1 + 0.8 sin(az)`` and the right ``1 - 0.8 sin(az)``.
    """
    import sofar

    import distill.audio.hrtf as hrtf

    az = np.array(_AZIMUTHS)
    ir = np.zeros((len(az), 2, 32))
    ir[:, 0, 0] = 1 + 0.8 * np.sin(np.radians(az))
    ir[:, 1, 0] = 1 - 0.8 * np.sin(np.radians(az))
    sofa = sofar.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = ir
    sofa.SourcePosition = np.stack([az, np.zeros(len(az)), np.ones(len(az))], axis=1)
    sofa.Data_SamplingRate = 48_000
    sofa.Data_Delay = np.zeros((1, 2))
    path = tmp_path / "test.sofa"
    sofar.write_sofa(str(path), sofa)

    hrtf.clear_hrtf_cache()
    monkeypatch.setattr(hrtf, "get_default_hrtf_path", lambda: path)
    yield path
    hrtf.clear_hrtf_cache()


def _source_azimuth(hrtf_data) -> float:
    return float(hrtf_data.source_azimuth)


def test_hrtf_lookup_wraps_around_360(sofa_file):
    from distill.audio.hrtf import load_hrtf

    # 358 deg is 2 deg from 0 deg, not 28 deg from 330 deg
    assert _source_azimuth(load_hrtf(sofa_file, azimuth=358.0)) == 0.0


def test_hrtf_cache_is_per_direction(sofa_file):
    from distill.audio.hrtf import load_hrtf

    left = load_hrtf(sofa_file, azimuth=30.0)
    right = load_hrtf(sofa_file, azimuth=330.0)
    assert _source_azimuth(left) == 30.0
    assert _source_azimuth(right) == 330.0
    assert left.hrir_left[0] > left.hrir_right[0]
    assert right.hrir_right[0] > right.hrir_left[0]


def test_binaural_keeps_a_mono_source_centred(sofa_file):
    t = np.arange(48_000) / 48_000
    mono = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    out = apply_spatial(
        mono, SpatialConfig(mode=SpatialMode.BINAURAL, width=1.0, depth=0.0)
    )

    assert out.shape == (2, mono.size)
    left_rms, right_rms = np.sqrt(np.mean(out**2, axis=1))
    # The old renderer used one HRIR at 90 deg, i.e. hard left
    assert left_rms == pytest.approx(right_rms, rel=0.02)
