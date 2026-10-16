"""CLI parsing and the train -> generate round trip."""

from __future__ import annotations

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
