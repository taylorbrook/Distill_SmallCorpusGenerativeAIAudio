"""CLI parsing and the train -> generate round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from distill.cli import app
from distill.library.catalog import ModelLibrary

runner = CliRunner()


def test_options_after_the_positional_argument_are_accepted(tmp_path):
    config = tmp_path / "config.toml"

    result = runner.invoke(
        app,
        [
            "generate",
            "missing-model",
            "--duration",
            "2",
            "--device",
            "cpu",
            "--config",
            str(config),
        ],
    )
    assert "Missing argument" not in result.output
    assert "Model not found" in result.output

    result = runner.invoke(
        app,
        [
            "train",
            str(tmp_path),
            "--preset",
            "balanced",
            "--device",
            "cpu",
            "--config",
            str(config),
        ],
    )
    assert "Missing argument" not in result.output
    assert "No audio files found" in result.output


def test_cli_train_saves_a_model_with_sliders_and_it_generates(corpus, tmp_path):
    config = tmp_path / "config.toml"
    common = ["--device", "cpu", "--config", str(config)]

    result = runner.invoke(
        app,
        [
            "train",
            str(corpus[0].parent),
            "--epochs",
            "2",
            "--model-name",
            "cli-model",
            *common,
        ],
    )
    assert result.exit_code == 0, result.output

    entries = ModelLibrary(tmp_path / "data" / "models").list_all()
    assert [e.name for e in entries] == ["cli-model"]
    assert entries[0].has_analysis

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "generate",
            "cli-model",
            "--duration",
            "1",
            "--format",
            "ogg",
            "--slider",
            "0:5",
            "--output-dir",
            str(out_dir),
            *common,
        ],
    )
    assert result.exit_code == 0, result.output
    assert list(out_dir.glob("*.ogg"))


def _config_file(tmp_path, models_dir):
    path = tmp_path / "config.toml"
    path.write_text(f'[paths]\nmodels = "{models_dir}"\ngenerated = "out"\n')
    return path


def test_binaural_is_not_offered_until_an_hrtf_file_ships(tmp_path):
    result = runner.invoke(
        app,
        [
            "generate",
            "any-model",
            "--spatial-mode",
            "binaural",
            "--config",
            str(tmp_path / "config.toml"),
        ],
    )
    assert result.exit_code != 0
    assert "Invalid spatial mode" in result.output


def test_stereo_is_widened_before_normalisation(trained, tmp_path):
    config = _config_file(tmp_path, trained["models_dir"])
    out_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "generate",
            "tones-model",
            "--spatial-mode",
            "stereo",
            "--spatial-width",
            "1.5",
            "--seed",
            "3",
            "--duration",
            "2",
            "--output-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output

    [wav] = out_dir.glob("*.wav")
    audio, _sr = sf.read(str(wav))
    assert audio.ndim == 2 and audio.shape[1] == 2
    # Widening after the 0.891 peak normalisation pushed peaks past full
    # scale, where the WAV writer clipped them
    assert np.abs(audio).max() <= 0.9


def test_global_options_apply_to_subcommands(trained, tmp_path, monkeypatch):
    import distill.hardware.device as device_mod

    requested = []
    real_select = device_mod.select_device

    def select_device(preference="auto"):
        requested.append(preference)
        return real_select("cpu")

    monkeypatch.setattr(device_mod, "select_device", select_device)
    config = _config_file(tmp_path, trained["models_dir"])

    result = runner.invoke(
        app,
        ["--config", str(config), "--device", "cpu", "model", "list", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert [m["name"] for m in json.loads(result.stdout)] == ["tones-model"]
    assert requested == ["cpu"]


def test_gui_resolves_data_paths_against_the_config_file(tmp_path, monkeypatch):
    import distill.ui
    import distill.validation.startup
    from distill.cli import _launch_gui

    config = tmp_path / "elsewhere" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        '[general]\nfirst_run_complete = true\n[paths]\nmodels = "my_models"\n'
    )
    launched = {}
    monkeypatch.setattr(
        distill.validation.startup,
        "run_startup_validation",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        distill.ui,
        "launch_ui",
        lambda config, device: launched.update(config),
    )

    _launch_gui(device="cpu", config_path=config)

    # Same directory the CLI commands use for this config
    assert launched["paths"]["models"] == str(config.parent / "my_models")


def test_unavailable_mps_falls_back_to_cpu():
    import torch

    from distill.hardware.device import select_device

    if torch.backends.mps.is_available():
        pytest.skip("MPS is available here")
    assert select_device("mps").type == "cpu"
