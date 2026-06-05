"""LLM reranker that reorders existing recommendation candidates only."""

from __future__ import annotations

import json
from typing import Any

from src.preprocessing.article_preprocessor import normalize_text


SYSTEM_PROMPT = """You are a biomedical literature recommendation reranker.

Your task is to rerank candidate PubMed articles according to the user's research question.

Rules:
- Recommend only articles from the provided candidate list.
- Do not invent PubMed IDs, titles, journals, or articles.
- Do not provide medical advice.
- Do not answer as a clinician.
- Prefer articles that match the population, exposure/intervention, and outcome of the user question.
- Prefer systematic reviews, reviews, clinical trials, and recent articles when relevant.
- Use the abstract snippets and metadata only.
- Return strict JSON only."""


class LLMRecommendationReranker:
    def __init__(
        self,
        ollama_client,
        max_candidates: int = 10,
        abstract_char_limit: int = 900,
    ):
        self.ollama_client = ollama_client
        self.max_candidates = max(1, int(max_candidates))
        self.abstract_char_limit = max(0, int(abstract_char_limit))

    def prepare_candidates(self, candidate_articles: list[dict]) -> list[dict]:
        """Prepare the compact candidate payload sent to Ollama."""
        return prepare_candidates(
            candidate_articles,
            max_candidates=self.max_candidates,
            abstract_char_limit=self.abstract_char_limit,
        )

    def rerank(
        self,
        user_question: str,
        candidate_articles: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank existing candidates with Ollama, falling back to original order."""
        top_k = max(0, int(top_k))
        if top_k == 0:
            return []

        candidates = _dedupe_candidates(candidate_articles)[: self.max_candidates]
        compact_candidates = self.prepare_candidates(candidate_articles)
        if not candidates or not compact_candidates:
            return []

        if self.ollama_client is None:
            return _fallback_results(candidates, top_k, "Ollama client is unavailable.")

        response = self.ollama_client.generate_json(
            prompt=_build_prompt(
                user_question=normalize_text(user_question),
                candidates=compact_candidates,
                top_k=top_k,
            ),
            system_prompt=SYSTEM_PROMPT,
        )
        if response.get("error"):
            return _fallback_results(candidates, top_k, response.get("error"))

        parsed_rows = parse_recommendations_response(response)
        if not parsed_rows:
            return _fallback_results(candidates, top_k, "Ollama returned no usable reranking rows.")

        return validate_llm_recommendations(candidates=candidates, parsed_rows=parsed_rows, top_k=top_k)

    def preview_prompt(
        self,
        user_question: str,
        candidate_articles: list[dict],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Return the system prompt and user prompt without calling Ollama."""
        compact_candidates = self.prepare_candidates(candidate_articles)
        return {
            "system": SYSTEM_PROMPT,
            "user": _build_prompt(
                user_question=normalize_text(user_question),
                candidates=compact_candidates,
                top_k=max(0, int(top_k)),
            ),
            "candidate_count": len(compact_candidates),
            "candidate_pmids": [candidate.get("pmid", "") for candidate in compact_candidates],
        }


def prepare_candidates(
    candidate_articles: list[dict],
    max_candidates: int = 10,
    abstract_char_limit: int = 900,
) -> list[dict]:
    """Return deterministic compact candidates for Ollama without mutating originals."""
    max_candidates = max(0, int(max_candidates))
    abstract_char_limit = max(0, int(abstract_char_limit))
    if max_candidates == 0:
        return []

    prepared: list[dict] = []
    seen_pmids: set[str] = set()
    for article in candidate_articles:
        if not isinstance(article, dict):
            continue

        pmid = _pmid(article)
        if not pmid or pmid in seen_pmids:
            continue

        seen_pmids.add(pmid)
        prepared.append(
            {
                "pmid": pmid,
                "title": normalize_text(article.get("title", "")),
                "year": article.get("year"),
                "journal": normalize_text(article.get("journal", "")),
                "publication_types": _clean_list(article.get("publication_types", [])),
                "mesh_terms": _clean_list(article.get("mesh_terms", [])),
                "abstract_snippet": _truncate(article.get("abstract", ""), abstract_char_limit),
                "source_method": normalize_text(article.get("method", "")) or "algorithmic",
            }
        )
        if len(prepared) >= max_candidates:
            break

    return prepared


def _build_prompt(user_question: str, candidates: list[dict], top_k: int) -> str:
    return f"""You are a biomedical literature recommendation reranker.

Your task is to rerank candidate PubMed articles according to the user's research question.

Rules:
- Recommend only articles from the provided candidate list.
- Do not invent PubMed IDs, titles, journals, or articles.
- Do not provide medical advice.
- Do not answer as a clinician.
- Prefer articles that match the population, exposure/intervention, and outcome of the user question.
- Prefer systematic reviews, reviews, clinical trials, and recent articles when relevant.
- Use the abstract snippets and metadata only.
- Return strict JSON only.

User question:
{user_question}

Candidate articles:
{json.dumps(candidates, ensure_ascii=False)}

Select the final Top {top_k} articles by relevance to the user question.

Required JSON output schema:
{{
  "recommendations": [
    {{
      "pmid": "string",
      "rank": 1,
      "reason": "Short explanation why this article is recommended.",
      "matched_concepts": ["concept 1", "concept 2"],
      "evidence_type": "systematic review | review | clinical trial | observational study | unknown"
    }}
  ]
}}

Return only this JSON object. Do not include prose before or after the JSON."""


def parse_recommendations_response(response: Any) -> list[dict]:
    """Safely extract recommendation rows from an Ollama JSON object."""
    if not isinstance(response, dict):
        return []

    rows = response.get("recommendations")
    if not isinstance(rows, list):
        rows = response.get("reranked_articles")
    if not isinstance(rows, list):
        rows = response.get("articles")
    if not isinstance(rows, list):
        return []

    return [row for row in rows if isinstance(row, dict)]


def validate_llm_recommendations(candidates: list[dict], parsed_rows: list[dict], top_k: int) -> list[dict]:
    """Validate LLM rows, discard hallucinated PMIDs, sort by rank, and fill gaps."""
    top_k = max(0, int(top_k))
    if top_k == 0:
        return []

    candidate_by_pmid = {_pmid(article): article for article in candidates if _pmid(article)}
    selected: list[dict] = []
    selected_pmids: set[str] = set()
    valid_rows: list[tuple[int, int, dict]] = []

    for response_order, row in enumerate(parsed_rows):
        pmid = _pmid(row)
        if not pmid or pmid not in candidate_by_pmid or pmid in selected_pmids:
            continue

        valid_rows.append((_rank_value(row), response_order, row))
        selected_pmids.add(pmid)

    selected_pmids.clear()
    for _, _, row in sorted(valid_rows, key=lambda item: (item[0], item[1])):
        pmid = _pmid(row)
        if not pmid or pmid in selected_pmids:
            continue

        selected_pmids.add(pmid)
        selected.append(_enrich_result(candidate_by_pmid[pmid], row, len(selected) + 1, status="ok"))
        if len(selected) >= top_k:
            break

    for candidate in candidates:
        if len(selected) >= top_k:
            break

        pmid = _pmid(candidate)
        if not pmid or pmid in selected_pmids:
            continue

        selected_pmids.add(pmid)
        selected.append(
            _enrich_result(
                candidate,
                {
                    "reason": "Added from original algorithmic ranking because the LLM returned fewer valid candidates.",
                    "matched_concepts": [],
                    "evidence_type": "unknown",
                },
                len(selected) + 1,
                status="partial_fallback",
            )
        )

    return selected


def _rank_value(row: dict) -> int:
    raw_rank = row.get("rank", row.get("llm_rank", 10**9))
    try:
        rank = int(raw_rank)
    except (TypeError, ValueError):
        return 10**9

    return rank if rank > 0 else 10**9


def _fallback_results(candidates: list[dict], top_k: int, reason: Any) -> list[dict]:
    fallback_reason = normalize_text(reason) or "LLM reranker unavailable; original algorithmic order was preserved."
    return [
        _enrich_result(
            article,
            {
                "reason": fallback_reason,
                "matched_concepts": [],
                "evidence_type": "unknown",
            },
            rank,
            status="fallback",
        )
        for rank, article in enumerate(candidates[:top_k], start=1)
    ]


def _enrich_result(article: dict, llm_row: dict, rank: int, status: str) -> dict:
    result = dict(article)
    source_method = normalize_text(article.get("source_method", article.get("method", ""))) or "algorithmic"
    algorithmic_score = _algorithmic_score(article)
    result.update(
        {
            "pmid": _pmid(article),
            "title": normalize_text(article.get("title", "")),
            "abstract": normalize_text(article.get("abstract", "")),
            "year": article.get("year"),
            "journal": normalize_text(article.get("journal", "")),
            "publication_types": _clean_list(article.get("publication_types", [])),
            "mesh_terms": _clean_list(article.get("mesh_terms", [])),
            "score": algorithmic_score,
            "method": f"{source_method}+llm_rerank",
            "algorithmic_score": algorithmic_score,
            "source_method": source_method,
            "llm_rank": rank,
            "llm_reason": _reason(llm_row),
            "matched_concepts": _clean_list(llm_row.get("matched_concepts", [])),
            "evidence_type": normalize_text(llm_row.get("evidence_type", "unknown")) or "unknown",
            "llm_rerank_status": status,
        }
    )
    return result


def _reason(llm_row: dict) -> str:
    reason = normalize_text(llm_row.get("reason", ""))
    if reason:
        return reason

    return normalize_text(llm_row.get("llm_reason", "")) or "not_available"


def _dedupe_candidates(candidate_articles: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for article in candidate_articles:
        if not isinstance(article, dict):
            continue

        pmid = _pmid(article)
        if not pmid or pmid in seen:
            continue

        seen.add(pmid)
        deduped.append(article)

    return deduped


def _pmid(article: dict) -> str:
    return normalize_text(article.get("pmid", ""))


def _algorithmic_score(article: dict) -> float:
    raw_score = article.get("final_score", article.get("score", 0.0))
    try:
        return float(raw_score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]

    cleaned = []
    for item in values:
        text = normalize_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def _truncate(value: Any, max_chars: int) -> str:
    text = normalize_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    return f"{text[:max_chars].rstrip()}..."
