"""Application configuration for NutriEvidence Agent."""

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(dotenv_path: str | None = None, *_args: object, **_kwargs: object) -> bool:
        path = Path(dotenv_path or ".env")
        if not path.exists():
            return False

        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)

        return True


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class Settings:
    ncbi_email: str
    ncbi_api_key: str | None
    use_llm: bool
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    use_openai_judge: bool
    openai_api_key: str | None
    openai_judge_model: str
    openai_judge_max_abstract_chars: int
    openai_judge_timeout: int


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    raise ValueError(
        f"Invalid boolean value for USE_LLM: {value!r}. "
        "Use one of true/false, yes/no, on/off, or 1/0."
    )


def load_settings(env_file: str | None = None) -> Settings:
    """Load settings from environment variables and an optional .env file."""
    if env_file:
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    use_llm = _parse_bool(os.getenv("USE_LLM"), default=True)
    use_openai_judge = _parse_bool(os.getenv("USE_OPENAI_JUDGE"), default=False)

    return Settings(
        ncbi_email=_optional_env("NCBI_EMAIL") or "",
        ncbi_api_key=_optional_env("NCBI_API_KEY"),
        use_llm=use_llm,
        llm_provider=(_optional_env("LLM_PROVIDER") or "ollama").lower(),
        ollama_base_url=_optional_env("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
        ollama_model=_optional_env("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
        use_openai_judge=use_openai_judge,
        openai_api_key=_optional_env("OPENAI_API_KEY"),
        openai_judge_model=_optional_env("OPENAI_JUDGE_MODEL") or "gpt-4o-mini",
        openai_judge_max_abstract_chars=_parse_int(
            os.getenv("OPENAI_JUDGE_MAX_ABSTRACT_CHARS"),
            default=1200,
            env_name="OPENAI_JUDGE_MAX_ABSTRACT_CHARS",
        ),
        openai_judge_timeout=_parse_int(
            os.getenv("OPENAI_JUDGE_TIMEOUT"),
            default=60,
            env_name="OPENAI_JUDGE_TIMEOUT",
        ),
    )


def _parse_int(value: str | None, default: int, env_name: str) -> int:
    if value is None or not value.strip():
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value for {env_name}: {value!r}") from exc

    return parsed


def require_ncbi_email(settings: Settings) -> str:
    """Return the configured NCBI email or fail for live PubMed retrieval."""
    if not settings.ncbi_email:
        raise RuntimeError(
            "NCBI_EMAIL is required for live PubMed retrieval. "
            "Set it in .env or use cached article data instead."
        )

    return settings.ncbi_email
