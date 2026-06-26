"""OpenAI GPT-based relation verification hooks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.data.articles import load_articles_csv
from gutbrainie.llm.ollama_relation_verifier import (
    NO_RELATION,
    RelationVerifierClient,
    article_context,
    parse_relation_decision,
    write_decisions_jsonl,
)
from gutbrainie.llm.prompts import build_relation_verification_prompt
from gutbrainie.re.candidates import generate_relation_candidates
from gutbrainie.re.relation_schema import VALID_RELATIONS, valid_predicates
from gutbrainie.re.rule_baseline import MENTION_RELATION_COLUMNS, deduplicate_mention_relations, load_entities
from gutbrainie.submission.export_t621 import mention_relations_to_t621_json

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ENV_PATH = ".env"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAIGPTClient:
    """Small OpenAI SDK adapter that exposes the local verifier protocol."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float = 120,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency 'openai'. Run: make install-gpt") from exc

        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, timeout=timeout)

    def generate(self, prompt: str) -> str:
        """Return raw model text for one prompt."""
        if hasattr(self.client, "responses"):
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=self.temperature,
            )
            output_text = getattr(response, "output_text", None)
            if output_text:
                return str(output_text)
            return _extract_responses_text(response)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only a JSON object with predicate and confidence.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return str(response.choices[0].message.content or "")


def predict_re_gpt_to_json(
    articles_path: str | Path,
    entities_path: str | Path,
    output_path: str | Path,
    model: str | None = None,
    env_path: str | Path | None = DEFAULT_ENV_PATH,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    api_key: str | None = None,
    threshold: float = 0.5,
    max_candidates: int | None = 50,
    max_distance: int | None = 0,
    timeout: float = 120,
    temperature: float = 0.0,
    decisions_output: str | Path | None = None,
    client: RelationVerifierClient | None = None,
) -> pd.DataFrame:
    """Verify relation candidates with OpenAI GPT and write T621 JSON predictions."""
    if env_path is not None:
        load_dotenv(env_path)

    resolved_model = model or os.environ.get("OPENAI_JUDGE_MODEL") or DEFAULT_OPENAI_MODEL
    resolved_key = api_key or os.environ.get(api_key_env)
    verifier = client or OpenAIGPTClient(
        model=resolved_model,
        api_key=resolved_key,
        timeout=timeout,
        temperature=temperature,
    )

    articles = load_articles_csv(articles_path)
    entities = load_entities(entities_path)
    candidates = generate_relation_candidates(
        articles=articles,
        entities=entities,
        valid_schema=VALID_RELATIONS,
        max_distance=max_distance,
    )
    if max_candidates is not None:
        candidates = candidates.head(max_candidates)

    article_lookup = {str(row["pmid"]): row for _, row in articles.iterrows()}
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for _, candidate_row in candidates.iterrows():
        candidate = candidate_row.to_dict()
        predicates = valid_predicates(candidate["subject_label"], candidate["object_label"])
        if not predicates:
            continue
        article = article_lookup.get(str(candidate["pmid"]))
        if article is None:
            continue
        article_text = article_context(article, candidate)
        prompt = build_relation_verification_prompt(article_text, candidate, predicates)
        raw_response = verifier.generate(prompt)
        decision = parse_relation_decision(raw_response, predicates)
        decisions.append(
            {
                "provider": "openai",
                "model": resolved_model,
                "pmid": str(candidate["pmid"]),
                "subject_text_span": str(candidate["subject_text_span"]),
                "subject_label": str(candidate["subject_label"]),
                "object_text_span": str(candidate["object_text_span"]),
                "object_label": str(candidate["object_label"]),
                "allowed_predicates": predicates,
                "raw_response": raw_response,
                **decision,
            }
        )
        if decision["predicate"] == NO_RELATION or decision["confidence"] < threshold:
            continue
        rows.append(
            {
                "pmid": str(candidate["pmid"]),
                "subject_text_span": str(candidate["subject_text_span"]),
                "subject_label": str(candidate["subject_label"]),
                "predicate": decision["predicate"],
                "object_text_span": str(candidate["object_text_span"]),
                "object_label": str(candidate["object_label"]),
            }
        )

    predictions = deduplicate_mention_relations(pd.DataFrame(rows, columns=MENTION_RELATION_COLUMNS))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(mention_relations_to_t621_json(predictions), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if decisions_output is not None:
        write_decisions_jsonl(decisions_output, decisions)
    return predictions


def load_dotenv(path: str | Path) -> dict[str, str]:
    """Load simple KEY=VALUE lines without overwriting existing environment variables."""
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_env_value(value.strip())
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def _strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _extract_responses_text(response: Any) -> str:
    parts: list[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)
