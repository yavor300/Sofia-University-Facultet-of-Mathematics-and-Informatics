import pytest

from gutbrainie.ner.predict_token_classifier import _resolve_model_path


def test_resolve_model_path_uses_latest_complete_checkpoint(tmp_path):
    model_dir = tmp_path / "model"
    old_checkpoint = model_dir / "checkpoint-10"
    latest_checkpoint = model_dir / "checkpoint-20"
    incomplete_checkpoint = model_dir / "checkpoint-30"
    old_checkpoint.mkdir(parents=True)
    latest_checkpoint.mkdir()
    incomplete_checkpoint.mkdir()
    for checkpoint in (old_checkpoint, latest_checkpoint):
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_text("", encoding="utf-8")
    (incomplete_checkpoint / "config.json").write_text("{}", encoding="utf-8")

    assert _resolve_model_path(model_dir) == latest_checkpoint


def test_resolve_model_path_reports_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train it first"):
        _resolve_model_path(tmp_path / "missing")
