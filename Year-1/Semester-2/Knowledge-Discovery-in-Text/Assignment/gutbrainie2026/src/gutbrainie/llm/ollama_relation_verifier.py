"""Ollama-based relation verification hooks."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from gutbrainie.data.articles import load_articles_csv
from gutbrainie.llm.prompts import build_relation_verification_prompt
from gutbrainie.re.candidates import generate_relation_candidates
from gutbrainie.re.relation_schema import VALID_RELATIONS, valid_predicates
from gutbrainie.re.rule_baseline import MENTION_RELATION_COLUMNS, deduplicate_mention_relations, load_entities
from gutbrainie.submission.export_t621 import mention_relations_to_t621_json

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
NO_RELATION = "no_relation"


class RelationVerifierClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Return raw model text for one prompt."""


class OllamaClient:
    """Small stdlib client for Ollama's `/api/generate` endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Start Ollama and pull the model with: ollama pull {self.model}"
            ) from exc
        return str(body.get("response", ""))


def predict_re_ollama_to_json(
    articles_path: str | Path,
    entities_path: str | Path,
    output_path: str | Path,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    threshold: float = 0.5,
    max_candidates: int | None = 200,
    max_distance: int | None = 0,
    timeout: float = 120,
    temperature: float = 0.0,
    decisions_output: str | Path | None = None,
    client: RelationVerifierClient | None = None,
) -> pd.DataFrame:
    """Verify relation candidates with Ollama and write T621 JSON predictions."""
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
    verifier = client or OllamaClient(model=model, base_url=base_url, timeout=timeout, temperature=temperature)

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
        record = {
            "pmid": str(candidate["pmid"]),
            "subject_text_span": str(candidate["subject_text_span"]),
            "subject_label": str(candidate["subject_label"]),
            "object_text_span": str(candidate["object_text_span"]),
            "object_label": str(candidate["object_label"]),
            "allowed_predicates": predicates,
            "raw_response": raw_response,
            **decision,
        }
        decisions.append(record)
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


def article_context(article: pd.Series | dict[str, Any], candidate: dict[str, Any]) -> str:
    """Return title/abstract context for one candidate."""
    row = article if isinstance(article, dict) else article.to_dict()
    title = str(row.get("title", ""))
    abstract = str(row.get("abstract", ""))
    if candidate["subject_location"] == candidate["object_location"]:
        location = str(candidate["subject_location"])
        text = title if location == "title" else abstract
        return f"{location.title()}: {text}"
    return f"Title: {title}\nAbstract: {abstract}"


def parse_relation_decision(raw_response: str, allowed_predicates: list[str]) -> dict[str, Any]:
    """Parse and normalize a JSON relation decision from an LLM response."""
    payload = _extract_json_object(raw_response)
    predicate = normalize_predicate(str(payload.get("predicate", NO_RELATION)), allowed_predicates)
    confidence = _coerce_confidence(payload.get("confidence", 0.0))
    if predicate == NO_RELATION:
        confidence = 0.0 if "confidence" not in payload else confidence
    return {"predicate": predicate, "confidence": confidence}


def normalize_predicate(predicate: str, allowed_predicates: list[str]) -> str:
    normalized = predicate.strip().lower().replace("_", " ")
    allowed_lookup = {item.lower(): item for item in allowed_predicates}
    if normalized in {NO_RELATION, "no relation", "none", "null", "na"}:
        return NO_RELATION
    return allowed_lookup.get(normalized, NO_RELATION)


def write_decisions_jsonl(path: str | Path, decisions: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"predicate": NO_RELATION, "confidence": 0.0}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"predicate": NO_RELATION, "confidence": 0.0}
    return payload if isinstance(payload, dict) else {"predicate": NO_RELATION, "confidence": 0.0}


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
