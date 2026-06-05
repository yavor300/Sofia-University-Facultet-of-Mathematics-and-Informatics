"""Annotate semantic evaluation candidates with the optional OpenAI judge."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.openai_judge import OpenAIRelevanceJudge
from src.retrieval.cache import load_articles
from src.utils.config import load_settings


DEFAULT_CANDIDATES_PATH = PROJECT_ROOT / "data" / "evaluation_candidates_semantic.csv"
DEFAULT_ARTICLES_PATH = PROJECT_ROOT / "data" / "pubmed_articles.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation_annotations_openai.csv"
FIELDNAMES = [
    "query_id",
    "query",
    "method",
    "rank",
    "pmid",
    "title",
    "year",
    "journal",
    "semantic_score",
    "judge_relevance",
    "judge_reason",
    "judge_model",
    "human_relevance",
    "human_notes",
]


def main() -> int:
    try:
        args = parse_args()
        settings = load_settings(str(args.env_file) if args.env_file else None)
        candidates = load_candidates(args.candidates)
        if not candidates:
            raise ValueError(f"No candidate rows found in {args.candidates}")

        article_by_pmid = load_article_index(args.articles)
        existing_by_key, orphan_rows = load_existing_output(args.output)
        api_key = args.api_key or settings.openai_api_key or ""
        judge = OpenAIRelevanceJudge(
            api_key=api_key,
            model=args.model or settings.openai_judge_model,
            max_abstract_chars=args.max_abstract_chars or settings.openai_judge_max_abstract_chars,
            timeout=args.timeout or settings.openai_judge_timeout,
        )

        if args.dry_run:
            print_dry_run_prompt(
                candidates,
                article_by_pmid,
                existing_by_key,
                judge,
                pmid=args.dry_run_pmid,
                require_abstract=args.dry_run_require_abstract,
            )
            return 0

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Set it in .env or pass --api-key.")

        rows, judged_count, skipped_count, failed_count = annotate_candidates(
            candidates=candidates,
            article_by_pmid=article_by_pmid,
            existing_by_key=existing_by_key,
            judge=judge,
            limit=args.limit,
        )
        current_keys = {row_key(candidate) for candidate in candidates}
        preserved_rows = [row for key, row in existing_by_key.items() if key not in current_keys]
        rows.extend(preserved_rows)
        rows.extend(orphan_rows)
        write_rows(rows, args.output)

        print(f"Wrote {len(rows)} rows to {args.output}")
        print(f"Judged: {judged_count}")
        print(f"Skipped existing/manual: {skipped_count}")
        print(f"Failed or missing judge labels: {failed_count}")
        return 0
    except Exception as exc:
        print(f"Error judging semantic recommendations: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--api-key", help="Overrides OPENAI_API_KEY from the environment.")
    parser.add_argument("--model", help="Overrides OPENAI_JUDGE_MODEL.")
    parser.add_argument("--max-abstract-chars", type=int, help="Overrides OPENAI_JUDGE_MAX_ABSTRACT_CHARS.")
    parser.add_argument("--timeout", type=int, help="Overrides OPENAI_JUDGE_TIMEOUT.")
    parser.add_argument("--limit", type=int, help="Judge at most this many new rows in this run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first unjudged prompt without calling OpenAI.")
    parser.add_argument("--dry-run-pmid", help="Preview the judge prompt for a specific PMID without calling OpenAI.")
    parser.add_argument(
        "--dry-run-require-abstract",
        action="store_true",
        help="In dry-run mode, preview the first unjudged row that has a cached abstract.",
    )
    return parser.parse_args()


def print_dry_run_prompt(
    candidates: list[dict],
    article_by_pmid: dict[str, dict],
    existing_by_key: dict[tuple[str, str, str, str], dict],
    judge: OpenAIRelevanceJudge,
    pmid: str | None = None,
    require_abstract: bool = False,
) -> None:
    candidate, row = first_unjudged_candidate(
        candidates,
        existing_by_key,
        article_by_pmid=article_by_pmid,
        pmid=pmid,
        require_abstract=require_abstract,
    )
    if not candidate or not row:
        print("No unjudged candidate rows found. Nothing to preview.")
        return

    article = article_by_pmid.get(row["pmid"], {})
    judge_article = {**candidate, **article, "score": candidate.get("semantic_score", "")}
    prompts = judge.preview_prompt(query=row["query"], article=judge_article)

    print("Dry run only. No OpenAI API call was made.")
    print(f"Candidate: PMID {row['pmid']} | method={row['method']} | rank={row['rank']}")
    if article.get("abstract"):
        print(f"Cached abstract: available ({len(str(article.get('abstract')))} characters)")
    else:
        print("Cached abstract: not available for this PMID in data/pubmed_articles.json")
    print("\n--- System Prompt ---")
    print(prompts["system"])
    print("\n--- User Prompt ---")
    print(prompts["user"])


def first_unjudged_candidate(
    candidates: list[dict],
    existing_by_key: dict[tuple[str, str, str, str], dict],
    article_by_pmid: dict[str, dict] | None = None,
    pmid: str | None = None,
    require_abstract: bool = False,
) -> tuple[dict | None, dict | None]:
    requested_pmid = clean(pmid)
    for candidate in candidates:
        if requested_pmid and clean(candidate.get("pmid")) != requested_pmid:
            continue

        existing = existing_by_key.get(row_key(candidate))
        row = build_output_row(candidate, existing)
        if row.get("human_relevance", "").strip() or row.get("judge_relevance", "").strip():
            continue
        if require_abstract:
            article = (article_by_pmid or {}).get(row["pmid"], {})
            if not clean(article.get("abstract")):
                continue
        return candidate, row

    return None, None


def annotate_candidates(
    candidates: list[dict],
    article_by_pmid: dict[str, dict],
    existing_by_key: dict[tuple[str, str, str, str], dict],
    judge: OpenAIRelevanceJudge,
    limit: int | None = None,
) -> tuple[list[dict], int, int, int]:
    output_rows: list[dict] = []
    judged_count = 0
    skipped_count = 0
    failed_count = 0
    remaining_limit = None if limit is None else max(0, int(limit))

    for index, candidate in enumerate(candidates, start=1):
        key = row_key(candidate)
        existing = existing_by_key.get(key)
        row = build_output_row(candidate, existing)

        if row.get("human_relevance", "").strip() or row.get("judge_relevance", "").strip():
            skipped_count += 1
            output_rows.append(row)
            print(f"[{index}/{len(candidates)}] Skipped already labeled PMID {row['pmid']}")
            continue

        if remaining_limit == 0:
            output_rows.append(row)
            continue

        article = article_by_pmid.get(row["pmid"], {})
        judge_article = {**candidate, **article, "score": candidate.get("semantic_score", "")}
        print(f"[{index}/{len(candidates)}] Judging PMID {row['pmid']} ({row['method']} rank {row['rank']})")
        result = judge.judge(query=row["query"], article=judge_article)

        relevance = result.get("relevance")
        if relevance is None:
            failed_count += 1
            row["judge_relevance"] = ""
        else:
            judged_count += 1
            row["judge_relevance"] = str(int(relevance))

        row["judge_reason"] = str(result.get("reason", "") or result.get("judge_error", "") or "").strip()
        row["judge_model"] = str(result.get("judge_model", judge.model) or judge.model).strip()
        output_rows.append(row)

        if remaining_limit is not None:
            remaining_limit -= 1

    return output_rows, judged_count, skipped_count, failed_count


def load_candidates(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        validate_candidate_columns(reader.fieldnames or [], path)
        return [normalize_candidate_row(row) for row in reader if row]


def validate_candidate_columns(fieldnames: list[str], path: Path) -> None:
    required = ["query_id", "query", "method", "rank", "pmid", "title", "year", "journal", "semantic_score"]
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f"Candidate file {path} is missing required columns: {', '.join(missing)}")


def normalize_candidate_row(row: dict[str, Any]) -> dict:
    return {
        "query_id": clean(row.get("query_id")),
        "query": clean(row.get("query")),
        "method": clean(row.get("method")),
        "rank": clean(row.get("rank")),
        "pmid": clean(row.get("pmid")),
        "title": clean(row.get("title")),
        "year": clean(row.get("year")),
        "journal": clean(row.get("journal")),
        "semantic_score": clean(row.get("semantic_score")),
        "relevance": clean(row.get("relevance")),
    }


def load_article_index(path: Path) -> dict[str, dict]:
    articles = load_articles(str(path))
    return {
        str(article.get("pmid", "")).strip(): article
        for article in articles
        if str(article.get("pmid", "")).strip()
    }


def load_existing_output(path: Path) -> tuple[dict[tuple[str, str, str, str], dict], list[dict]]:
    if not path.exists():
        return {}, []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        existing_rows = [normalize_output_row(row) for row in reader if row]

    by_key: dict[tuple[str, str, str, str], dict] = {}
    orphan_rows: list[dict] = []
    for row in existing_rows:
        key = row_key(row)
        if all(key):
            by_key[key] = row
        else:
            orphan_rows.append(row)

    return by_key, orphan_rows


def build_output_row(candidate: dict, existing: dict | None = None) -> dict:
    row = {field: "" for field in FIELDNAMES}
    row.update({field: clean(candidate.get(field)) for field in FIELDNAMES if field in candidate})

    if existing:
        for field in ["judge_relevance", "judge_reason", "judge_model", "human_relevance", "human_notes"]:
            row[field] = clean(existing.get(field))

    if not row["human_relevance"]:
        row["human_relevance"] = clean(candidate.get("relevance"))

    return row


def normalize_output_row(row: dict[str, Any]) -> dict:
    normalized = {field: clean(row.get(field)) for field in FIELDNAMES}
    return normalized


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDNAMES} for row in rows])


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean(row.get("query_id")),
        clean(row.get("method")),
        clean(row.get("rank")),
        clean(row.get("pmid")),
    )


def clean(value: Any) -> str:
    return str(value or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
