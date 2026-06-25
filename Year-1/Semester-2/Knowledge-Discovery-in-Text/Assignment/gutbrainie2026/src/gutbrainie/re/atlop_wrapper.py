"""Optional wrappers around the official GutBrainIE ATLOP baseline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


OFFICIAL_REPO_URL = "https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline"

REQUIRED_ATLOP_FILES = {
    "README": "README.md",
    "annotation conversion notebook": "Utils/annotations_to_atlop_format.ipynb",
    "NER prediction conversion notebook": "Utils/NER_predictions_to_atlop_format.ipynb",
    "training set composer": "Train/RE/compose_training_sets.py",
    "fine-tuning script": "Train/RE/atlop_finetune.sh",
    "prediction script": "Train/RE/atlop_generate_predictions.sh",
    "ATLOP interface": "Train/RE/atlop_interface.py",
    "relation label map": "Train/RE/data/meta/rel2id.json",
}

COMPOSE_INPUTS = (
    "Train/RE/data/train_gold.json",
    "Train/RE/data/train_silver.json",
    "Train/RE/data/train_silver_2025.json",
    "Train/RE/data/train_bronze.json",
    "Train/RE/data/dev.json",
)

FINETUNE_INPUTS = (
    "Train/RE/data/train_annotated.json",
    "Train/RE/data/dev.json",
)

PREDICT_INPUTS = (
    "Train/RE/outputs/best.ckpt",
    "Train/RE/data/predicted_entities_dev_atlop_format.json",
)

ATLOP_ACTIONS = ("compose", "finetune", "predict")


def inspect_atlop_repository(official_repo: str | Path) -> dict[str, Any]:
    """Inspect whether the official ATLOP baseline checkout is runnable."""
    repo = Path(official_repo)
    files = {
        name: {
            "path": relative_path,
            "exists": (repo / relative_path).exists(),
        }
        for name, relative_path in REQUIRED_ATLOP_FILES.items()
    }
    baseline_models = {
        "RE.zip": _model_archive_status(repo / "BaselineModels" / "RE.zip"),
        "NER.zip": _model_archive_status(repo / "BaselineModels" / "NER.zip"),
    }
    data_inputs = {relative_path: (repo / relative_path).exists() for relative_path in COMPOSE_INPUTS}
    finetune_inputs = {relative_path: (repo / relative_path).exists() for relative_path in FINETUNE_INPUTS}
    predict_inputs = {relative_path: (repo / relative_path).exists() for relative_path in PREDICT_INPUTS}
    return {
        "official_repo": str(repo),
        "official_repo_exists": repo.exists(),
        "required_files": files,
        "baseline_models": baseline_models,
        "compose_inputs": data_inputs,
        "finetune_inputs": finetune_inputs,
        "predict_inputs": predict_inputs,
        "can_compose": repo.exists() and all(data_inputs.values()) and files["training set composer"]["exists"],
        "can_finetune": repo.exists() and all(finetune_inputs.values()) and files["fine-tuning script"]["exists"],
        "can_predict": repo.exists() and all(predict_inputs.values()) and files["prediction script"]["exists"],
    }


def write_atlop_notes(
    official_repo: str | Path,
    output_path: str | Path,
    data_root: str | Path = "data/gutbrainie2026",
) -> dict[str, Any]:
    """Write a Markdown reproduction note for the optional ATLOP baseline."""
    status = inspect_atlop_repository(official_repo)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_notes(status, Path(data_root)), encoding="utf-8")
    return {"output": str(output), **status}


def run_atlop_action(
    official_repo: str | Path,
    action: str,
    output_path: str | Path | None = None,
    log_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one official ATLOP action from the external baseline checkout."""
    if action not in ATLOP_ACTIONS:
        raise ValueError(f"Unknown ATLOP action '{action}'. Expected one of {ATLOP_ACTIONS}.")

    repo = Path(official_repo).resolve()
    re_dir = repo / "Train" / "RE"
    if not repo.exists():
        raise FileNotFoundError(
            f"Official baseline repo not found: {repo}. Clone it with:\n"
            f"  git clone {OFFICIAL_REPO_URL} external/GutBrainIE_2026_Baseline"
        )
    if not re_dir.exists():
        raise FileNotFoundError(f"Official ATLOP directory not found: {re_dir}")

    command = _command_for_action(action)
    result: dict[str, Any] = {
        "action": action,
        "cwd": str(re_dir),
        "command": command,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    _assert_action_prerequisites(repo, action)
    completed = subprocess.run(command, cwd=re_dir, check=False, capture_output=True, text=True)
    result.update(
        {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if log_path is not None:
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(_format_run_log(result), encoding="utf-8")
        result["log_path"] = str(log)

    if action == "predict" and output_path is not None:
        official_output = repo / "Predictions" / "RE" / "predicted_relations.json"
        if completed.returncode == 0 and official_output.exists():
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(official_output, output)
            result["output_path"] = str(output)
        elif completed.returncode == 0:
            raise FileNotFoundError(f"ATLOP prediction command succeeded but did not create {official_output}")

    return result


def _command_for_action(action: str) -> list[str]:
    if action == "compose":
        return [sys.executable, "compose_training_sets.py"]
    if action == "finetune":
        return ["bash", "atlop_finetune.sh"]
    if action == "predict":
        return ["bash", "atlop_generate_predictions.sh"]
    raise ValueError(action)


def _assert_action_prerequisites(repo: Path, action: str) -> None:
    if action == "compose":
        required = COMPOSE_INPUTS
    elif action == "finetune":
        required = FINETUNE_INPUTS
    else:
        required = PREDICT_INPUTS
    missing = [relative_path for relative_path in required if not (repo / relative_path).exists()]
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"ATLOP action '{action}' is missing required official-baseline files:\n{rendered}\n"
            "Run the official conversion notebooks/scripts first, or use `make atlop-notes` for the checklist."
        )


def _model_archive_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    content = path.read_bytes()[:128]
    is_lfs_pointer = content.startswith(b"version https://git-lfs.github.com/spec/")
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "is_git_lfs_pointer": is_lfs_pointer,
    }


def _render_notes(status: dict[str, Any], data_root: Path) -> str:
    required = "\n".join(
        f"- [{'x' if details['exists'] else ' '}] {name}: `{details['path']}`"
        for name, details in status["required_files"].items()
    )
    compose_inputs = "\n".join(
        f"- [{'x' if exists else ' '}] `{path}`" for path, exists in status["compose_inputs"].items()
    )
    finetune_inputs = "\n".join(
        f"- [{'x' if exists else ' '}] `{path}`" for path, exists in status["finetune_inputs"].items()
    )
    predict_inputs = "\n".join(
        f"- [{'x' if exists else ' '}] `{path}`" for path, exists in status["predict_inputs"].items()
    )
    archives = "\n".join(
        "- `{name}`: exists={exists}, git_lfs_pointer={pointer}, size_bytes={size}".format(
            name=name,
            exists=details.get("exists"),
            pointer=details.get("is_git_lfs_pointer"),
            size=details.get("size_bytes"),
        )
        for name, details in status["baseline_models"].items()
    )
    return f"""# ATLOP Reproduction Notes

ATLOP is treated as an optional comparison. The local project implementation remains the PubMedBERT pair classifier.

## Official Repository

- Path: `{status['official_repo']}`
- Exists: `{status['official_repo_exists']}`
- Upstream: `{OFFICIAL_REPO_URL}`
- Local data root used by this project: `{data_root}`

## Required Official Files

{required}

## Baseline Model Archives

{archives}

If an archive is a Git LFS pointer, fetch it inside the official repo before using pretrained weights:

```bash
git -C {status['official_repo']} lfs pull
```

## Data Conversion Checklist

The official ATLOP scripts expect converted JSON files in `Train/RE/data`. Use the official notebooks as reference:

- `Utils/annotations_to_atlop_format.ipynb`
- `Utils/NER_predictions_to_atlop_format.ipynb`

Required before `compose`:

{compose_inputs}

Required before `finetune`:

{finetune_inputs}

Required before `predict`:

{predict_inputs}

## Local Wrapper Commands

Inspect and refresh this note:

```bash
make atlop-notes
```

Show the official command without running it:

```bash
make run-atlop ATLOP_ACTION=compose ATLOP_DRY_RUN=1
make run-atlop ATLOP_ACTION=finetune ATLOP_DRY_RUN=1
make run-atlop ATLOP_ACTION=predict ATLOP_DRY_RUN=1
```

Run the official steps after the converted files exist:

```bash
make run-atlop ATLOP_ACTION=compose
make run-atlop ATLOP_ACTION=finetune
make run-atlop ATLOP_ACTION=predict ATLOP_OUTPUT=outputs/predictions/atlop_predicted_relations_raw.json
```

## Current Status

- Can compose training sets now: `{status['can_compose']}`
- Can fine-tune now: `{status['can_finetune']}`
- Can predict now: `{status['can_predict']}`

Notes:

- The official ATLOP code is run from `external/GutBrainIE_2026_Baseline/Train/RE`.
- The wrapper does not edit official scripts.
- Prediction output copied by the wrapper is the official raw ATLOP relation output. Convert or merge it into challenge T621 submission format before internal T621 evaluation if needed.
"""


def _format_run_log(result: dict[str, Any]) -> str:
    payload = {
        "action": result.get("action"),
        "cwd": result.get("cwd"),
        "command": result.get("command"),
        "returncode": result.get("returncode"),
    }
    return (
        json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n\n# stdout\n"
        + str(result.get("stdout", ""))
        + "\n\n# stderr\n"
        + str(result.get("stderr", ""))
    )
