"""Shared fixtures: a tiny synthetic corpus and one short CPU training run.

Everything runs on CPU in temp directories.  Griffin-Lim iterations and
the analyzer's sweep points are reduced for the whole session -- they only
affect audio quality and label suggestions, not the code paths under test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

SAMPLE_RATE = 48_000


@pytest.fixture(scope="session", autouse=True)
def _fast_audio():
    import distill.audio.spectrogram as spectrogram_mod
    import distill.controls.analyzer as analyzer_mod

    torch.set_num_threads(2)
    orig_spec_init = spectrogram_mod.AudioSpectrogram.__init__
    orig_analyzer_init = analyzer_mod.LatentSpaceAnalyzer.__init__

    def spec_init(self, config=None):
        orig_spec_init(self, config)
        self.griffin_lim.n_iter = 4

    def analyzer_init(self, variance_threshold=0.02, n_steps=21, n_sweep_points=3):
        orig_analyzer_init(self, variance_threshold, n_steps, n_sweep_points)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(spectrogram_mod.AudioSpectrogram, "__init__", spec_init)
        mp.setattr(analyzer_mod.LatentSpaceAnalyzer, "__init__", analyzer_init)
        yield


def write_corpus(directory: Path, n_files: int = 6, seconds: float = 2.0) -> list[Path]:
    """Write *n_files* distinct mono test tones (some with noise) as WAV."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    paths = []
    for i in range(n_files):
        sig = 0.5 * np.sin(2 * np.pi * 110 * 2 ** (i / 2) * t)
        sig += 0.1 * (i % 2) * rng.standard_normal(t.size)
        sig *= np.linspace(1.0, 0.2, t.size)
        path = directory / f"tone_{i}.wav"
        sf.write(path, sig.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
        paths.append(path)
    return paths


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> list[Path]:
    return write_corpus(tmp_path_factory.mktemp("corpus"))


def small_config(file_count: int, epochs: int = 3):
    from distill.training.config import get_adaptive_config

    config = get_adaptive_config(file_count)
    config.max_epochs = epochs
    config.regularization.augmentation_expansion = 0
    return config


@pytest.fixture(scope="session")
def trained(corpus, tmp_path_factory) -> dict:
    """One completed 3-epoch run saved to a model library."""
    from distill.training.loop import train

    work = tmp_path_factory.mktemp("trained")
    torch.manual_seed(0)
    result = train(
        small_config(len(corpus)),
        corpus,
        work / "run",
        torch.device("cpu"),
        models_dir=work / "models",
        dataset_name="tones",
        model_name="tones-model",
    )
    result["models_dir"] = work / "models"
    return result
