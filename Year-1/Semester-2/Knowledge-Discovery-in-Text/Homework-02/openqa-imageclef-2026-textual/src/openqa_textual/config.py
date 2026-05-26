"""Configuration helpers for local development and experiments."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand ${NAME:-default} placeholders in loaded config values."""

    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and expand simple environment placeholders."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load YAML config files.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {config_path}.")
    return expand_env_vars(data)


def project_root() -> Path:
    """Return the repository root for this project."""

    return Path(__file__).resolve().parents[2]

