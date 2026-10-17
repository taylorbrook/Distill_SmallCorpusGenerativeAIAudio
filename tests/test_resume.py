"""Resuming a run: where it restarts, which settings apply, and what it refuses."""

from __future__ import annotations

import json
import math
import shutil
import threading

import pytest
import torch

from distill.storage import safe_torch_load
from distill.training.checkpoint import get_best_checkpoint, manage_checkpoints
from distill.training.loop import train
from distill.training.metrics import EpochMetrics, MetricsHistory, StepMetrics
from tests.conftest import small_config


def _config(file_count: int, epochs: int, lr: float = 1e-3, weight_decay: float = 0.01):
    config = small_config(file_count, epochs=epochs)
    config.learning_rate = lr
    config.regularization.weight_decay = weight_decay
    return config


@pytest.fixture(scope="module")
def two_epoch_run(corpus, tmp_path_factory):
    """A completed 2-epoch run directory (copy it before resuming into it)."""
    run_dir = tmp_path_factory.mktemp("two_epoch") / "run"
    torch.manual_seed(0)
    train(_config(len(corpus), epochs=2), corpus, run_dir, torch.device("cpu"))
    return run_dir


def _latest(run_dir):
    return max((run_dir / "checkpoints").glob("checkpoint_epoch*.pt"))


def _copy_run(run_dir, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    return copy


def _epoch_recorder():
    epochs: list[EpochMetrics] = []

    def callback(event):
        if isinstance(event, EpochMetrics):
            epochs.append(event)

    return epochs, callback


def test_resume_follows_the_new_config_not_the_checkpoint(
    corpus, two_epoch_run, tmp_path
):
    run_dir = _copy_run(two_epoch_run, tmp_path)
    epochs, callback = _epoch_recorder()

    result = train(
        _config(len(corpus), epochs=4, lr=5e-4, weight_decay=0.05),
        corpus,
        run_dir,
        torch.device("cpu"),
        callback=callback,
        resume_checkpoint=_latest(run_dir),
    )

    assert [m.epoch for m in epochs] == [2, 3]
    # The lr logged after epoch 2 is the new cosine schedule's value for
    # epoch 3 of 4, starting from the new base lr
    expected = 5e-4 * (1 + math.cos(math.pi * 3 / 4)) / 2
    assert epochs[0].learning_rate == pytest.approx(expected, rel=1e-6)

    ckpt = safe_torch_load(result["final_checkpoint_path"])
    assert ckpt["scheduler_state_dict"]["T_max"] == 4
    assert ckpt["scheduler_state_dict"]["base_lrs"] == [5e-4]
    assert ckpt["optimizer_state_dict"]["param_groups"][0]["weight_decay"] == 0.05


def test_mid_epoch_cancel_is_repeated_on_resume(corpus, tmp_path):
    run_dir = tmp_path / "run"
    cancel = threading.Event()

    def cancel_during_second_epoch(event):
        if isinstance(event, StepMetrics) and event.epoch == 1:
            cancel.set()

    result = train(
        _config(len(corpus), epochs=3),
        corpus,
        run_dir,
        torch.device("cpu"),
        callback=cancel_during_second_epoch,
        cancel_event=cancel,
    )
    checkpoint = result["final_checkpoint_path"]
    assert checkpoint == _latest(run_dir)
    saved = safe_torch_load(checkpoint)
    assert saved["epoch"] == 1
    assert saved["epoch_complete"] is False

    epochs, callback = _epoch_recorder()
    train(
        _config(len(corpus), epochs=3),
        corpus,
        run_dir,
        torch.device("cpu"),
        callback=callback,
        resume_checkpoint=checkpoint,
    )
    # Epoch 1 was interrupted, so it runs again rather than being skipped
    assert [m.epoch for m in epochs] == [1, 2]


def test_a_checkpoint_that_cannot_be_loaded_is_an_error(corpus, tmp_path):
    bad = tmp_path / "run" / "checkpoints" / "checkpoint_epoch0001.pt"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a checkpoint")

    with pytest.raises(RuntimeError, match="Could not resume"):
        train(
            _config(len(corpus), epochs=1),
            corpus,
            tmp_path / "run",
            torch.device("cpu"),
            resume_checkpoint=bad,
        )


def test_a_fresh_run_refuses_a_directory_holding_another_runs_checkpoints(
    corpus,
    two_epoch_run,
    tmp_path,
):
    run_dir = _copy_run(two_epoch_run, tmp_path)
    before = sorted(p.name for p in (run_dir / "checkpoints").iterdir())

    with pytest.raises(ValueError, match="already holds checkpoints"):
        train(_config(len(corpus), epochs=1), corpus, run_dir, torch.device("cpu"))

    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == before


def _fake_checkpoint(directory, epoch, val_loss):
    path = directory / f"checkpoint_epoch{epoch:04d}.pt"
    path.write_bytes(b"")
    meta = {"epoch": epoch, "train_loss": 1.0, "val_loss": val_loss}
    path.with_suffix(".pt.meta.json").write_text(json.dumps(meta))
    return path


def test_nan_validation_loss_never_ranks_as_best(tmp_path):
    nan_ckpt = _fake_checkpoint(tmp_path, 1, float("nan"))
    good = _fake_checkpoint(tmp_path, 2, 5.0)
    for epoch in (3, 4, 5):
        _fake_checkpoint(tmp_path, epoch, 6.0)

    assert get_best_checkpoint(tmp_path) == good
    manage_checkpoints(tmp_path, max_recent=3)
    assert good.exists()
    assert not nan_ckpt.exists()

    history = MetricsHistory()
    for epoch, loss in enumerate([float("nan"), 3.0, 4.0]):
        history.add_epoch(
            EpochMetrics(
                epoch=epoch,
                total_epochs=3,
                train_loss=1.0,
                val_loss=loss,
                val_recon_loss=0.0,
                val_kl_loss=0.0,
                kl_divergence=1.0,
                overfitting_gap=0.0,
                learning_rate=1e-3,
                eta_seconds=0.0,
                elapsed_seconds=0.0,
            )
        )
    assert history.get_best_epoch() == 1
