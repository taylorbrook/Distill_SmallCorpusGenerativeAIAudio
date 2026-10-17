"""Gradio app wiring: training controls, data/library tabs, export and blending."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import gradio as gr
import pytest
import soundfile as sf
import torch

from distill.config.settings import load_config
from distill.inference.generation import GenerationConfig, GenerationPipeline
from distill.library.catalog import ModelLibrary
from distill.models.persistence import (
    ModelMetadata,
    load_model,
    save_model_from_checkpoint,
)
from distill.ui.app import create_app
from distill.ui.state import app_state, reset_metrics_buffer
from distill.ui.tabs import data_tab
from distill.ui.tabs.generate_tab import (
    MAX_SLIDERS,
    _export_audio,
    _generate_audio,
    _refresh_blend_model_choices,
)
from distill.ui.tabs.library_tab import _load_model_handler
from tests.conftest import write_corpus

# model name, max epochs, lr, dropout, weight decay, KL weight, preset
_TRAIN_ARGS = ("", 5, 1e-3, 0.2, 0.01, 0.01, "Balanced")


def _handler(app, name):
    for fn in app.fns.values():
        if getattr(fn.fn, "__name__", "") == name:
            return fn.fn
    raise LookupError(name)


class _FakeRunner:
    def __init__(self, is_running=True):
        self.is_running = is_running
        self.cancelled = False
        self.last_error = None
        self.calls = []

    def cancel(self):
        self.cancelled = True

    def start(self, **kwargs):
        self.calls.append(("start", kwargs))

    def resume(self, **kwargs):
        self.calls.append(("resume", kwargs))


def _reset_app_state():
    app_state.training_runner = None
    app_state.training_active = False
    app_state.current_dataset = None
    app_state.current_summary = None
    app_state.loaded_model = None
    app_state.pipeline = None
    app_state.preset_manager = None
    app_state.history_store = None
    app_state.blend_engine = None
    app_state.blend_models = {}
    reset_metrics_buffer()


def _tmp_config(tmp_path):
    config = load_config(tmp_path / "config.toml")
    config["paths"] = {
        key: str(tmp_path / value) for key, value in config["paths"].items()
    }
    return config


@pytest.fixture
def app(tmp_path):
    _reset_app_state()
    yield create_app(config=_tmp_config(tmp_path), device=torch.device("cpu"))
    _reset_app_state()


def _dataset(tmp_path):
    return SimpleNamespace(name="imported", valid_files=[tmp_path / "a.wav"])


# ---------------------------------------------------------------------------
# Train tab
# ---------------------------------------------------------------------------


def test_cancel_keeps_start_disabled_until_the_run_reports_completion(app):
    cancel = _handler(app, "_cancel_training")
    poll = _handler(app, "_poll_training")
    runner = _FakeRunner()
    app_state.training_runner = runner
    app_state.training_active = True

    timer, start_btn, cancel_btn, stats = cancel()
    assert runner.cancelled
    assert start_btn["interactive"] is False
    assert cancel_btn["interactive"] is False
    assert "Cancelling" in stats

    # Still finishing (checkpoint, analysis, model save): keep polling
    outputs = poll()
    assert outputs[3] == {"__type__": "update"}  # start_btn untouched
    assert "Cancelling" in outputs[1]

    # The training thread reports completion
    app_state.metrics_buffer["complete"] = True
    runner.is_running = False
    outputs = poll()
    assert outputs[3]["interactive"] is True
    assert "cancelled" in outputs[1].lower()


def test_start_and_resume_refuse_while_a_run_is_in_progress(app, tmp_path):
    runner = _FakeRunner(is_running=True)
    app_state.training_runner = runner
    app_state.current_dataset = _dataset(tmp_path)
    app_state.metrics_buffer["epoch_metrics"].append("the running run's metrics")

    for name in ("_start_training", "_resume_training"):
        outputs = _handler(app, name)(*_TRAIN_ARGS)
        assert "already running" in outputs[1]
        assert outputs[3]["interactive"] is False  # Start stays disabled

    assert app_state.training_runner is runner
    assert runner.calls == []
    assert app_state.metrics_buffer["epoch_metrics"] == ["the running run's metrics"]


def test_reloading_the_page_reattaches_to_a_running_run(app):
    app_state.training_runner = _FakeRunner(is_running=True)

    stats, timer, start_btn, cancel_btn, _resume = _handler(app, "_reattach_to_run")()

    assert "in progress" in stats
    assert timer.active is True
    assert start_btn["interactive"] is False
    assert cancel_btn["interactive"] is True


def test_each_run_gets_its_own_output_dir_outside_the_dataset(app, tmp_path):
    runner = _FakeRunner(is_running=False)
    app_state.training_runner = runner
    app_state.current_dataset = _dataset(tmp_path)

    start = _handler(app, "_start_training")
    start(*_TRAIN_ARGS)
    start(*_TRAIN_ARGS)

    dirs = [kwargs["output_dir"] for _, kwargs in runner.calls]
    assert len(dirs) == 2
    assert dirs[0] != dirs[1]
    for output_dir in dirs:
        # The dataset directory is scanned for audio: previews written
        # there would become training data
        assert app_state.datasets_dir not in output_dir.parents
        assert output_dir.parent == app_state.training_dir / "imported"


def _fake_run(run_dir, losses):
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    for epoch, val_loss in losses.items():
        path = checkpoints / f"checkpoint_epoch{epoch:04d}.pt"
        path.write_bytes(b"")
        meta = {"epoch": epoch, "train_loss": 1.0, "val_loss": val_loss}
        path.with_suffix(".pt.meta.json").write_text(json.dumps(meta))
    return checkpoints


def test_resume_continues_the_newest_run_from_its_latest_checkpoint(app, tmp_path):
    runner = _FakeRunner(is_running=False)
    app_state.training_runner = runner
    app_state.current_dataset = _dataset(tmp_path)
    root = app_state.training_dir / "imported"
    older = _fake_run(root / "run_a", {4: 1.0})
    for path in older.iterdir():
        os.utime(path, (1, 1))
    # Epoch 12 is a cancel checkpoint (val_loss inf); epoch 10 is "best"
    _fake_run(root / "run_b", {10: 1.0, 12: float("inf")})

    _handler(app, "_resume_training")(*_TRAIN_ARGS)

    [(kind, kwargs)] = runner.calls
    assert kind == "resume"
    assert (
        kwargs["checkpoint_path"]
        == root / "run_b" / "checkpoints" / "checkpoint_epoch0012.pt"
    )
    assert kwargs["output_dir"] == root / "run_b"


# ---------------------------------------------------------------------------
# Data tab
# ---------------------------------------------------------------------------


def test_data_tab_only_uses_files_imported_into_the_dataset_dir(app):
    import_dir = app_state.datasets_dir / "imported"
    write_corpus(import_dir, n_files=2, seconds=0.5)
    # Older versions wrote training output (preview WAVs) inside it
    write_corpus(import_dir / "training" / "previews", n_files=1, seconds=0.5)

    data_tab._load_existing_imports()

    files = app_state.current_dataset.valid_files
    assert len(files) == 2
    assert all(f.parent == import_dir.resolve() for f in files)

    update = data_tab._handle_thumbnail_click(
        gr.SelectData(None, {"index": 1, "value": None})
    )
    assert update["visible"] is True  # the player starts hidden
    assert update["value"] == str(files[1])


# ---------------------------------------------------------------------------
# Library tab wiring
# ---------------------------------------------------------------------------


def test_generate_tab_refresh_runs_after_the_library_load_finishes(app):
    loaders = {
        fn._id
        for fn in app.fns.values()
        if getattr(fn.fn, "__name__", "") == "_load_model_handler"
    }
    refreshes = [
        fn
        for fn in app.fns.values()
        if getattr(fn.fn, "__name__", "") == "_update_sliders_for_model"
    ]
    assert refreshes
    # A separate listener on the Load button races the load itself
    for fn in refreshes:
        assert fn.trigger_after in loaders


def test_clicking_a_model_card_selects_it(app):
    select = _handler(app, "_card_click_select")
    update = select(gr.EventData(None, {"model_name": "Pads v1"}))
    assert update["value"] == "Pads v1"

    # The card grid reports clicks itself (Gradio 6 does not mount a
    # visible=False textbox, so the old JS bridge had nothing to target)
    card_grids = [
        block
        for block in app.blocks.values()
        if isinstance(block, gr.HTML)
        and "trigger('click'" in (getattr(block, "js_on_load", None) or "")
    ]
    assert len(card_grids) == 1


def test_launch_allows_serving_files_from_the_data_dirs(tmp_path, monkeypatch):
    from distill.ui.app import launch_ui

    captured = {}
    monkeypatch.setattr(
        gr.Blocks, "launch", lambda self, **kwargs: captured.update(kwargs)
    )
    _reset_app_state()
    try:
        launch_ui(config=_tmp_config(tmp_path), device=torch.device("cpu"))
    finally:
        _reset_app_state()

    allowed = captured.get("allowed_paths") or []
    for name in ("history", "generated", "datasets", "training"):
        assert str((tmp_path / "data" / name).resolve()) in allowed


# ---------------------------------------------------------------------------
# Generate tab
# ---------------------------------------------------------------------------


def test_export_uses_the_chosen_sample_rate_and_bit_depth(app, trained):
    loaded = load_model(trained["saved_model_path"])
    pipeline = GenerationPipeline(loaded.model, loaded.spectrogram, torch.device("cpu"))
    result = pipeline.generate(GenerationConfig(duration_s=1.0, seed=1))
    app_state.metrics_buffer["last_result"] = result

    for name, pipe in (("via_pipeline", pipeline), ("fallback", None)):
        app_state.pipeline = pipe
        status = _export_audio("44100", "16-bit", name, "wav", "", "", "")
        assert "Exported" in status, status

        info = sf.info(str(app_state.generated_dir / f"{name}.wav"))
        assert info.samplerate == 44_100
        assert info.subtype == "PCM_16"
        # Resampled, not 48 kHz samples under a 44.1 kHz header
        expected = result.audio.shape[-1] * 44_100 / 48_000
        assert info.frames == pytest.approx(expected, abs=2)


def test_blend_panel_mixes_the_picked_models(app, trained, tmp_path):
    models_dir = tmp_path / "blend_models"
    for name in ("Model A", "Model B"):
        save_model_from_checkpoint(
            trained["final_checkpoint_path"],
            ModelMetadata(name=name),
            models_dir,
        )
    app_state.models_dir = models_dir
    app_state.model_library = ModelLibrary(models_dir)
    _load_model_handler("Model A")

    updates = _refresh_blend_model_choices()
    assert sorted(updates[0]["choices"]) == ["Model A", "Model B"]
    assert updates[0]["value"] == "Model A"  # blends with the loaded model

    sliders = [0] * MAX_SLIDERS
    # duration, output mode, width, depth, seed, evolution
    settings = [1.0, "stereo", 0.7, 0.5, 7, 0.0]
    blend_rows = ["Model A", "Model B", None, None] + [60, 40, 25, 25] + [2]
    audio, badge, *_ = _generate_audio(*sliders, *settings, *blend_rows)

    assert audio is not None, badge
    _sr, samples = audio
    assert samples.ndim == 2 and samples.shape[1] == 2  # stereo kept
    engine = app_state.blend_engine
    assert [s.metadata.name for s in engine.slots] == ["Model A", "Model B"]
    assert [s.weight for s in engine.slots] == [60, 40]
    assert (
        app_state.history_store.list_entries(limit=1)[0].model_name
        == "Model A + Model B"
    )

    # After "Remove Last" row 2 keeps its value but is hidden: no blend
    blend_rows[-1] = 1
    audio, badge, *_ = _generate_audio(*sliders, *settings, *blend_rows)
    assert audio is not None, badge
    assert app_state.history_store.list_entries(limit=1)[0].model_name == "Model A"


def test_binaural_is_not_offered_until_an_hrtf_file_ships(app):
    [output_mode] = [
        block
        for block in app.blocks.values()
        if isinstance(block, gr.Dropdown) and block.label == "Output Mode"
    ]
    assert [value for _label, value in output_mode.choices] == ["mono", "stereo"]
