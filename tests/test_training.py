"""Training runs leave behind a model whose sliders match its weights."""

from __future__ import annotations

import threading

import numpy as np
import torch

from distill.library.catalog import ModelLibrary
from distill.models.persistence import (
    ModelMetadata,
    load_model,
    save_model_from_checkpoint,
)
from distill.storage import safe_torch_load
from distill.training.loop import train
from distill.training.metrics import EpochMetrics
from tests.conftest import small_config


def test_completed_run_saves_model_with_matching_analysis(trained):
    loaded = load_model(trained["saved_model_path"])
    assert loaded.analysis is not None
    assert loaded.metadata.has_analysis
    assert loaded.metadata.training_epochs == 3
    assert loaded.metadata.training_date

    # The final checkpoint holds the same weights and the same analysis
    ckpt = safe_torch_load(trained["final_checkpoint_path"])
    assert ckpt["latent_analysis"] is not None
    for name, tensor in loaded.model.state_dict().items():
        torch.testing.assert_close(tensor, ckpt["model_state_dict"][name])
    np.testing.assert_allclose(
        loaded.analysis.pca_mean,
        ckpt["latent_analysis"]["pca_mean"],
    )

    entries = ModelLibrary(trained["models_dir"]).list_all()
    assert [e.name for e in entries] == ["tones-model"]
    assert entries[0].has_analysis


def test_save_from_final_checkpoint_keeps_sliders(trained, tmp_path):
    path = save_model_from_checkpoint(
        trained["final_checkpoint_path"],
        ModelMetadata(name="copy"),
        tmp_path,
    )
    loaded = load_model(path)
    assert loaded.analysis is not None
    assert loaded.metadata.training_epochs == 3  # epoch index 2 -> 3 epochs


def test_cancel_after_an_epoch_still_saves_a_model(corpus, tmp_path):
    cancel = threading.Event()

    def callback(event):
        if isinstance(event, EpochMetrics):
            cancel.set()

    result = train(
        small_config(len(corpus), epochs=5),
        corpus,
        tmp_path / "run",
        torch.device("cpu"),
        callback=callback,
        cancel_event=cancel,
        models_dir=tmp_path / "models",
        model_name="cancelled",
    )

    assert len(result["metrics_history"].epoch_metrics) == 1
    assert result["analysis"] is not None
    loaded = load_model(result["saved_model_path"])
    assert loaded.analysis is not None
    assert loaded.metadata.training_epochs == 1


def test_cancel_before_first_epoch_keeps_checkpoint_only(corpus, tmp_path):
    cancel = threading.Event()
    cancel.set()

    result = train(
        small_config(len(corpus)),
        corpus,
        tmp_path / "run",
        torch.device("cpu"),
        cancel_event=cancel,
        models_dir=tmp_path / "models",
    )

    assert result["saved_model_path"] is None
    assert result["analysis"] is None
    assert result["final_checkpoint_path"].exists()
    assert not (tmp_path / "models").exists()
