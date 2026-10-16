"""JSON index durability, tolerant loading, and safe model loading."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from distill.library.catalog import ModelEntry, ModelLibrary
from distill.models.persistence import load_model
from distill.storage import safe_torch_load, write_json_atomic
from distill.training.config import OverfittingPreset


def _entry(model_id: str, **overrides) -> ModelEntry:
    fields = dict(
        model_id=model_id,
        name=model_id,
        description="",
        file_path=f"{model_id}.distill",
        file_size_bytes=1,
        dataset_name="",
        dataset_file_count=0,
        dataset_total_duration_s=0.0,
        training_date="",
        save_date="",
        training_epochs=1,
        final_train_loss=0.0,
        final_val_loss=0.0,
        has_analysis=False,
        n_active_components=0,
    )
    fields.update(overrides)
    return ModelEntry(**fields)


def test_corrupt_index_recovers_from_backup_and_is_kept_aside(tmp_path):
    library = ModelLibrary(tmp_path)
    library.add_entry(_entry("a"))
    library.add_entry(_entry("b"))  # backup now holds {"a"}
    index = tmp_path / "model_library.json"
    index.write_text('{"version": 1, "models": {"a": {')  # truncated write

    recovered = ModelLibrary(tmp_path)
    assert [e.model_id for e in recovered.list_all()] == ["a"]
    assert list(tmp_path.glob("model_library.json.corrupt-*"))

    recovered.add_entry(_entry("c"))
    assert set(json.loads(index.read_text())["models"]) == {"a", "c"}
    backup = json.loads((tmp_path / "model_library.json.bak").read_text())
    assert set(backup["models"]) == {"a"}


def test_write_never_backs_up_an_unreadable_index(tmp_path):
    index = tmp_path / "presets.json"
    write_json_atomic(index, {"presets": {"keep": 1}})
    write_json_atomic(index, {"presets": {"keep": 1, "new": 2}})
    index.write_text("{not json")

    write_json_atomic(index, {"presets": {}})

    backup = json.loads((tmp_path / "presets.json.bak").read_text())
    assert backup == {"presets": {"keep": 1}}


def test_entries_with_unknown_fields_still_load(tmp_path):
    entry = asdict(_entry("a"))
    entry["field_from_a_newer_version"] = 42
    (tmp_path / "model_library.json").write_text(
        json.dumps({"version": 1, "models": {"a": entry}}),
    )
    assert [e.model_id for e in ModelLibrary(tmp_path).list_all()] == ["a"]


def test_model_with_unknown_metadata_field_loads(trained, tmp_path):
    saved = safe_torch_load(trained["saved_model_path"])
    saved["metadata"]["field_from_a_newer_version"] = "x"
    saved["spectrogram_config"]["another_new_field"] = 1
    path = tmp_path / "future.distill"
    torch.save(saved, path)

    assert load_model(path).metadata.name == "tones-model"


def test_existing_checkpoints_load_with_weights_only(trained):
    ckpt = safe_torch_load(trained["final_checkpoint_path"])
    assert isinstance(ckpt["training_config"]["preset"], OverfittingPreset)
    assert ckpt["latent_analysis"]["pca_components"].ndim == 2


class _RunsCode:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def test_model_file_carrying_code_is_refused(tmp_path):
    marker = tmp_path / "code_ran"
    path = tmp_path / "evil.distill"
    torch.save({"format": "distill_model", "version": 1, "x": _RunsCode(marker)}, path)

    with pytest.raises(ValueError, match="Refusing to load"):
        load_model(path)
    assert not marker.exists()
