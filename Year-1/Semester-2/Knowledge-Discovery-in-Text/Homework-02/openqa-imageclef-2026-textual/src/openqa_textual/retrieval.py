"""Retrieval index construction utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
from typing import Any

from openqa_textual.data import get_sample_gold_answer, get_sample_id, get_sample_language
from openqa_textual.ocr import OCRResult, load_ocr_cache_record
from openqa_textual.ocr_postprocess import clean_ocr_question
from openqa_textual.prediction import read_jsonl, write_jsonl


@dataclass(slots=True)
class RetrievalIndexRecord:
    question_id: str
    language: str
    ocr_question: str
    gold_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_retrieval_index_from_ocr_rows(
    ocr_rows: list[dict[str, Any]],
    gold_by_id: dict[str, str] | None = None,
    text_field: str = "clean_question",
) -> list[dict[str, Any]]:
    """Build retrieval index records from OCR JSONL rows."""

    gold_by_id = gold_by_id or {}
    records = []
    for row in ocr_rows:
        question_id = str(row.get("question_id", ""))
        language = str(row.get("language") or "English")
        question = row.get(text_field)
        if question is None and text_field != "ocr_text":
            question = row.get("ocr_text", "")
        question = clean_ocr_question(str(question or ""), language=language)
        records.append(
            RetrievalIndexRecord(
                question_id=question_id,
                language=language,
                ocr_question=question,
                gold_answer=gold_by_id.get(question_id, str(row.get("gold_answer") or "")),
                metadata={
                    "source": row.get("source"),
                    "split": row.get("split"),
                    "ocr_engine": row.get("ocr_engine"),
                    "preprocess_variant": row.get("preprocess_variant"),
                    "confidence": row.get("confidence"),
                },
            ).to_dict()
        )
    return records


def build_retrieval_index_from_dataset(
    dataset_split: Any,
    split_name: str,
    cache_dir: str | Path,
    engine: str,
    preprocess_variant: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build retrieval records from a dataset split using existing OCR cache only."""

    records = []
    total = len(dataset_split) if limit is None else min(len(dataset_split), max(limit, 0))
    for index in range(total):
        sample = dataset_split[index]
        try:
            question_id = get_sample_id(sample)
        except KeyError:
            question_id = f"sample-{index:05d}"
        language = get_sample_language(sample)
        gold_answer = get_sample_gold_answer(sample)

        cache_record = load_ocr_cache_record(
            cache_dir=cache_dir,
            split=split_name,
            engine=engine,
            preprocess_variant=preprocess_variant,
            question_id=question_id,
        )
        ocr_result = cache_record.to_result() if cache_record else OCRResult("", None, engine, {})
        ocr_question = clean_ocr_question(ocr_result.text, language=language)

        records.append(
            RetrievalIndexRecord(
                question_id=question_id,
                language=language,
                ocr_question=ocr_question,
                gold_answer=gold_answer,
                metadata={
                    "source": f"{split_name}:{index}",
                    "split": split_name,
                    "ocr_engine": engine,
                    "preprocess_variant": preprocess_variant,
                    "confidence": ocr_result.confidence,
                    "cache_hit": cache_record is not None,
                },
            ).to_dict()
        )
    return records


def gold_answers_from_dataset_split(dataset_split: Any) -> dict[str, str]:
    """Return question_id -> gold answer for a split."""

    gold_by_id: dict[str, str] = {}
    for index in range(len(dataset_split)):
        sample = dataset_split[index]
        try:
            question_id = get_sample_id(sample)
        except KeyError:
            question_id = f"sample-{index:05d}"
        gold_by_id[question_id] = get_sample_gold_answer(sample)
    return gold_by_id


def load_ocr_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_retrieval_index(path: str | Path, records: list[dict[str, Any]]) -> None:
    write_jsonl(path, records)


def tokenize_for_retrieval(text: str) -> list[str]:
    """Simple multilingual tokenization for lexical retrieval."""

    return re.findall(r"[\w]+", str(text or "").casefold(), flags=re.UNICODE)


class BM25Retriever:
    """BM25 over retrieval index OCR question text."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.corpus = [tokenize_for_retrieval(record.get("ocr_question", "")) for record in records]
        self._ranker = self._build_ranker(self.corpus) if self.corpus else None

    @staticmethod
    def _build_ranker(corpus: list[list[str]]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is required for BM25 retrieval.") from exc
        return BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.records or self._ranker is None:
            return []
        query_tokens = tokenize_for_retrieval(query)
        bm25_scores = self._ranker.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        scores = [
            float(score) + _token_overlap_score(query_token_set, document_tokens)
            for score, document_tokens in zip(bm25_scores, self.corpus, strict=False)
        ]
        return _rank_records(self.records, scores, top_k=top_k, score_field="bm25_score")


class DenseRetriever:
    """SentenceTransformer dense retrieval over OCR question text."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        model_name: str = "intfloat/multilingual-e5-base",
        cache_dir: str | Path | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.records = records
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = self._load_model(model_name)
        self.embeddings = self._load_or_encode_embeddings(cache_dir)

    @staticmethod
    def _load_model(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for dense retrieval.") from exc
        return SentenceTransformer(model_name)

    def _load_or_encode_embeddings(self, cache_dir: str | Path | None):
        import numpy as np

        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / f"{_safe_model_name(self.model_name)}.npy"
            if cache_path.exists():
                cached = np.load(cache_path)
                if cached.shape[0] == len(self.records):
                    return cached

        texts = [_dense_text(record.get("ocr_question", ""), self.model_name) for record in self.records]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
        return embeddings

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.records:
            return []
        query_embedding = self.model.encode(
            [_dense_text(query, self.model_name)],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ query_embedding
        return _rank_records(self.records, scores, top_k=top_k, score_field="dense_score")


class HybridRetriever:
    """Hybrid retrieval combining normalized BM25 and dense scores."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        model_name: str = "intfloat/multilingual-e5-base",
        embedding_cache: str | Path | None = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.records = records
        self.bm25 = BM25Retriever(records)
        self.dense = DenseRetriever(records, model_name=model_name, cache_dir=embedding_cache)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.records:
            return []

        bm25_scores = list(self.bm25._ranker.get_scores(tokenize_for_retrieval(query)))
        query_embedding = self.dense.model.encode(
            [_dense_text(query, self.dense.model_name)],
            convert_to_numpy=True,
            normalize_embeddings=self.dense.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        dense_scores = list(self.dense.embeddings @ query_embedding)

        bm25_norm = _min_max_normalize(bm25_scores)
        dense_norm = _min_max_normalize(dense_scores)
        scores = [
            (self.bm25_weight * bm25_score) + (self.dense_weight * dense_score)
            for bm25_score, dense_score in zip(bm25_norm, dense_norm, strict=False)
        ]
        ranked = _rank_records(self.records, scores, top_k=top_k, score_field="hybrid_score")
        for item in ranked:
            index = item["rank_index"]
            item["bm25_score"] = bm25_scores[index]
            item["dense_score"] = dense_scores[index]
        return ranked


def build_retriever(
    method: str,
    records: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> BM25Retriever | DenseRetriever | HybridRetriever:
    config = config or {}
    method = method.lower()
    if method == "bm25":
        return BM25Retriever(records)
    if method == "dense":
        dense_config = config.get("dense", {})
        return DenseRetriever(
            records,
            model_name=dense_config.get("model_name", "intfloat/multilingual-e5-base"),
            cache_dir=config.get("index", {}).get("embedding_cache"),
        )
    if method == "hybrid":
        dense_config = config.get("dense", {})
        hybrid_config = config.get("hybrid", {})
        return HybridRetriever(
            records,
            model_name=dense_config.get("model_name", "intfloat/multilingual-e5-base"),
            embedding_cache=config.get("index", {}).get("embedding_cache"),
            bm25_weight=float(hybrid_config.get("bm25_weight", 0.5)),
            dense_weight=float(hybrid_config.get("dense_weight", 0.5)),
        )
    raise ValueError(f"Unknown retrieval method: {method}")


def load_retrieval_index(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _rank_records(
    records: list[dict[str, Any]],
    scores,
    top_k: int,
    score_field: str,
) -> list[dict[str, Any]]:
    ranked_indices = sorted(
        range(len(records)),
        key=lambda index: (float(scores[index]), records[index].get("question_id", "")),
        reverse=True,
    )[: max(top_k, 0)]
    results = []
    for rank, index in enumerate(ranked_indices, start=1):
        item = dict(records[index])
        item["rank"] = rank
        item["rank_index"] = index
        score = float(scores[index])
        item[score_field] = score if math.isfinite(score) else 0.0
        results.append(item)
    return results


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def _token_overlap_score(query_tokens: set[str], document_tokens: list[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens.intersection(document_tokens)) / len(query_tokens)


def _dense_text(text: str, model_name: str) -> str:
    if "e5" in model_name.lower():
        return f"query: {text}"
    return text


def _safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("._")
