"""Wrapper hooks for official challenge evaluation scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_OFFICIAL_ENTRYPOINTS = (
    "baseline/evaluation/evaluate.py",
    "evaluation/evaluate.py",
    "baseline/evaluation/validate.py",
    "evaluation/validate.py",
)


def run_official_evaluation(
    official_repo: str | Path,
    prediction: str | Path,
    entrypoint: str | Path | None = None,
    gold: str | Path | None = None,
    task: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run an official evaluation script from a local baseline repository checkout."""
    repo_path = Path(official_repo).resolve()
    prediction_path = Path(prediction).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"Official repository not found: {repo_path}. "
            "Clone it with: git clone https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline "
            "external/GutBrainIE_2026_Baseline"
        )
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    script_path = _resolve_entrypoint(repo_path, entrypoint)
    command = [sys.executable, str(script_path), "--prediction", str(prediction_path)]
    if gold is not None:
        command.extend(["--gold", str(gold)])
    if task is not None:
        command.extend(["--task", task])
    if extra_args:
        command.extend(extra_args)

    completed = subprocess.run(
        command,
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _resolve_entrypoint(repo_path: Path, entrypoint: str | Path | None) -> Path:
    if entrypoint is not None:
        script_path = Path(entrypoint)
        if not script_path.is_absolute():
            script_path = repo_path / script_path
        if not script_path.exists():
            raise FileNotFoundError(f"Official evaluation entrypoint not found: {script_path}")
        return script_path

    for relative_path in DEFAULT_OFFICIAL_ENTRYPOINTS:
        script_path = repo_path / relative_path
        if script_path.exists():
            return script_path

    candidates = "\n".join(f"- {path}" for path in DEFAULT_OFFICIAL_ENTRYPOINTS)
    raise FileNotFoundError(
        "Could not find an official evaluation entrypoint. "
        "Pass --entrypoint explicitly. Tried:\n"
        f"{candidates}"
    )
