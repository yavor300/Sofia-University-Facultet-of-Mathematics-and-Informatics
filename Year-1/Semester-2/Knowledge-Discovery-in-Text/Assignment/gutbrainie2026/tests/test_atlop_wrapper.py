import json

from gutbrainie.re.atlop_wrapper import inspect_atlop_repository, run_atlop_action, write_atlop_notes


def test_write_atlop_notes_reports_required_files(tmp_path):
    official_repo = tmp_path / "official"
    _write_fake_atlop_repo(official_repo)
    output = tmp_path / "reports" / "atlop_notes.md"

    status = write_atlop_notes(official_repo, output)

    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "ATLOP Reproduction Notes" in text
    assert "annotations_to_atlop_format.ipynb" in text
    assert status["can_compose"] is True
    assert status["can_finetune"] is False


def test_run_atlop_dry_run_does_not_execute_script(tmp_path):
    official_repo = tmp_path / "official"
    _write_fake_atlop_repo(official_repo)

    result = run_atlop_action(official_repo, "compose", dry_run=True)

    assert result["dry_run"] is True
    assert result["command"][-1] == "compose_training_sets.py"
    assert not (official_repo / "Train" / "RE" / "executed.txt").exists()


def test_run_atlop_compose_executes_official_script(tmp_path):
    official_repo = tmp_path / "official"
    _write_fake_atlop_repo(official_repo)
    log_path = tmp_path / "logs" / "atlop_compose.log"

    result = run_atlop_action(official_repo, "compose", log_path=log_path)

    assert result["returncode"] == 0
    assert log_path.exists()
    assert (official_repo / "Train" / "RE" / "executed.txt").read_text(encoding="utf-8") == "compose"


def test_inspect_atlop_detects_lfs_pointer(tmp_path):
    official_repo = tmp_path / "official"
    _write_fake_atlop_repo(official_repo)
    archive = official_repo / "BaselineModels" / "RE.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")

    status = inspect_atlop_repository(official_repo)

    assert status["baseline_models"]["RE.zip"]["is_git_lfs_pointer"] is True


def _write_fake_atlop_repo(path):
    re_dir = path / "Train" / "RE"
    data_dir = re_dir / "data"
    meta_dir = data_dir / "meta"
    utils_dir = path / "Utils"
    meta_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)

    (path / "README.md").write_text("official baseline", encoding="utf-8")
    (utils_dir / "annotations_to_atlop_format.ipynb").write_text("{}", encoding="utf-8")
    (utils_dir / "NER_predictions_to_atlop_format.ipynb").write_text("{}", encoding="utf-8")
    (re_dir / "atlop_finetune.sh").write_text("echo finetune\n", encoding="utf-8")
    (re_dir / "atlop_generate_predictions.sh").write_text("echo predict\n", encoding="utf-8")
    (re_dir / "atlop_interface.py").write_text("print('interface')\n", encoding="utf-8")
    (meta_dir / "rel2id.json").write_text(json.dumps({"NA": 0}), encoding="utf-8")
    for name in ["train_gold.json", "train_silver.json", "train_silver_2025.json", "train_bronze.json", "dev.json"]:
        (data_dir / name).write_text("[]", encoding="utf-8")
    (re_dir / "compose_training_sets.py").write_text(
        "from pathlib import Path\nPath('executed.txt').write_text('compose', encoding='utf-8')\n",
        encoding="utf-8",
    )
