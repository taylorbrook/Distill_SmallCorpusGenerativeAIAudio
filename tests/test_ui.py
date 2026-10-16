"""The Gradio app builds, and Train-tab cancellation waits for the run to finish."""

from __future__ import annotations

import pytest
import torch

from distill.config.settings import load_config
from distill.ui.app import create_app
from distill.ui.state import app_state


def _handler(app, name):
    for fn in app.fns.values():
        if getattr(fn.fn, "__name__", "") == name:
            return fn.fn
    raise LookupError(name)


class _FakeRunner:
    def __init__(self):
        self.is_running = True
        self.cancelled = False
        self.last_error = None

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def app(tmp_path):
    config = load_config(tmp_path / "config.toml")
    config["paths"] = {
        key: str(tmp_path / value) for key, value in config["paths"].items()
    }
    return create_app(config=config, device=torch.device("cpu"))


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
