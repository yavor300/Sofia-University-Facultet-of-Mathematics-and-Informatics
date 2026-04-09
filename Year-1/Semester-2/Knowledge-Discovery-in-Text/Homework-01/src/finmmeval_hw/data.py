from __future__ import annotations

import ast
import json
import numbers
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

HF_REPO = "Tomas08119993/finmmeval-cfa-cpa"
HF_REPO_TYPE = "dataset"
HF_ENGLISH_PARQUET = "data/train-00000-of-00001-en.parquet"
BBF_REPO = "bharatgenai/BhashaBench-Finance"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

OPTION_START_RE = re.compile(r"^\s*([A-Za-z])[\.\)\:]\s*(.*)$")
LETTER_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d+")
LETTER_ONLY_RE = re.compile(r"^\(?([a-zA-Z])[\)\.\:\-]?$")
TRAILING_LETTER_RE = re.compile(r"(?:option|choice)?[_\s-]*([a-z])$", re.IGNORECASE)
TRAILING_NUMBER_RE = re.compile(r"(\d+)$")


def ensure_english_parquet(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_copy = cache_dir / HF_ENGLISH_PARQUET
    if local_copy.exists():
        return local_copy
    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        filename=HF_ENGLISH_PARQUET,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def _normalize_label(value: str) -> str:
    return value.strip().lower()


def _label_from_index(idx: int) -> str:
    if 0 <= idx < len(ALPHABET):
        return ALPHABET[idx]
    return f"option_{idx + 1}"


def _candidate_to_label(candidate: object, fallback_idx: int) -> str:
    raw = str(candidate).strip()
    if not raw:
        return _label_from_index(fallback_idx)

    lowered = raw.lower()
    letter_only = LETTER_ONLY_RE.match(raw)
    if letter_only:
        return letter_only.group(1).lower()

    if lowered.isdigit():
        number = int(lowered)
        return _label_from_index(number - 1 if number > 0 else number)

    trailing_letter = TRAILING_LETTER_RE.search(lowered)
    if trailing_letter:
        return trailing_letter.group(1).lower()

    trailing_number = TRAILING_NUMBER_RE.search(lowered)
    if trailing_number:
        number = int(trailing_number.group(1))
        return _label_from_index(number - 1 if number > 0 else number)

    return _label_from_index(fallback_idx)


def _first_present(container: Mapping[str, object], keys: Sequence[str]) -> object:
    key_map = {str(key).lower(): key for key in container.keys()}
    for key in keys:
        if key in key_map:
            return container[key_map[key]]
    return None


def _find_column_name(container: Mapping[str, object], normalized_name: str) -> str | None:
    key_map = {
        re.sub(r"[^a-z0-9]+", "", str(key).lower()): str(key) for key in container.keys()
    }
    return key_map.get(normalized_name)


def _safe_list_parse(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, dict):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            return list(value)  # handles numpy arrays and other iterables
        except TypeError:
            pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                continue
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [value]


def parse_choices(value: object) -> List[str]:
    parsed = _safe_list_parse(value)
    out: List[str] = []
    for item in parsed:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        if len(text) == 1 and text.isalpha():
            out.append(_normalize_label(text))
        else:
            letter_match = LETTER_RE.search(text)
            if letter_match:
                out.append(_normalize_label(letter_match.group(0)))
    return out


def parse_answer_letters(value: object) -> List[str]:
    parsed = _safe_list_parse(value)
    letters: List[str] = []
    if parsed:
        for item in parsed:
            if item is None:
                continue
            matches = LETTER_RE.findall(str(item))
            for match in matches:
                letters.append(_normalize_label(match))
    else:
        text = "" if value is None else str(value)
        for match in LETTER_RE.findall(text):
            letters.append(_normalize_label(match))
    return deduplicate_keep_order(letters)


def parse_gold_indices(value: object) -> List[int]:
    if value is None:
        return []
    if isinstance(value, numbers.Integral):
        return [int(value)]
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return [int(value)]

    parsed = _safe_list_parse(value)
    indices: List[int] = []
    if parsed:
        for item in parsed:
            try:
                indices.append(int(item))
            except (TypeError, ValueError):
                continue
        return deduplicate_keep_order(indices)

    text = str(value)
    for match in DIGIT_RE.findall(text):
        indices.append(int(match))
    return deduplicate_keep_order(indices)


def deduplicate_keep_order(values: Sequence[object]) -> List[object]:
    seen = set()
    out = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_option_map(query: str) -> Dict[str, str]:
    option_map: Dict[str, str] = {}
    current_label: str | None = None
    current_lines: List[str] = []

    def flush_current() -> None:
        nonlocal current_label, current_lines
        if current_label is None:
            return
        content = " ".join(part.strip() for part in current_lines if part.strip()).strip()
        if content:
            option_map[current_label] = content
        current_label = None
        current_lines = []

    for line in query.splitlines():
        match = OPTION_START_RE.match(line)
        if match:
            flush_current()
            current_label = _normalize_label(match.group(1))
            content = match.group(2).strip()
            current_lines = [content] if content else []
            continue

        if current_label is not None and line.strip():
            current_lines.append(line.strip())

    flush_current()
    return option_map


def _select_question_text(row: pd.Series) -> str:
    text = str(row.get("text", "") or "").strip()
    if text:
        return text
    return str(row.get("query", "") or "").strip()


def _derive_gold_letters(choice_labels: List[str], row: pd.Series) -> List[str]:
    from_gold: List[str] = []
    for idx in parse_gold_indices(row.get("gold")):
        if 0 <= idx < len(choice_labels):
            from_gold.append(choice_labels[idx])

    from_answer = parse_answer_letters(row.get("answer"))
    if from_gold:
        return deduplicate_keep_order(from_gold)  # type: ignore[return-value]
    return [label for label in from_answer if label in set(choice_labels)]


def _build_option_texts(choice_labels: List[str], query: str) -> Dict[str, str]:
    option_map = parse_option_map(query)
    if not choice_labels:
        choice_labels = sorted(option_map.keys())

    texts: Dict[str, str] = {}
    for label in choice_labels:
        option_text = option_map.get(label, "").strip()
        if not option_text:
            option_text = f"Option {label.upper()}"
        texts[label] = option_text
    return texts


def _extract_choice_labels_and_texts(
    options_value: object,
    query: str,
    include_query_options: bool = True,
) -> Dict[str, object]:
    labels: List[str] = []
    option_texts: Dict[str, str] = {}

    def add_option(label_candidate: object, text_candidate: object, idx: int) -> None:
        label = _candidate_to_label(label_candidate, fallback_idx=idx)
        text = "" if text_candidate is None else str(text_candidate).strip()
        if label not in labels:
            labels.append(label)
        if text:
            option_texts[label] = text

    if include_query_options:
        parsed_query_map = parse_option_map(query)
        for label in sorted(parsed_query_map.keys()):
            if label not in labels:
                labels.append(label)
            option_texts[label] = parsed_query_map[label]

    if isinstance(options_value, Mapping):
        for idx, (key, value) in enumerate(options_value.items()):
            if isinstance(value, Mapping):
                text = _first_present(
                    value,
                    ["text", "value", "content", "option_text", "option", "choice"],
                )
                add_option(key, text, idx)
            else:
                add_option(key, value, idx)
    else:
        parsed_options = _safe_list_parse(options_value)
        for idx, item in enumerate(parsed_options):
            if isinstance(item, Mapping):
                text = _first_present(
                    item,
                    ["text", "value", "content", "option_text", "choice", "option"],
                )
                label_hint = _first_present(item, ["label", "key", "id", "option", "choice"])
                add_option(label_hint if label_hint is not None else idx + 1, text, idx)
                continue

            item_text = "" if item is None else str(item).strip()
            if not item_text:
                continue
            inline = OPTION_START_RE.match(item_text)
            if inline:
                add_option(inline.group(1), inline.group(2), idx)
            else:
                add_option(idx + 1, item_text, idx)

    if not labels:
        labels = sorted(option_texts.keys())

    for idx, label in enumerate(labels):
        if label not in option_texts or not option_texts[label].strip():
            option_texts[label] = f"Option {_label_from_index(idx).upper()}"

    return {"choice_labels": labels, "option_texts": option_texts}


def _map_indices_to_labels(indices: List[int], choice_labels: List[str]) -> List[str]:
    if not indices or not choice_labels:
        return []

    n = len(choice_labels)
    zero_based_valid = all(0 <= idx < n for idx in indices)
    if zero_based_valid:
        return [choice_labels[idx] for idx in indices]

    one_based_valid = all(1 <= idx <= n for idx in indices)
    if one_based_valid:
        return [choice_labels[idx - 1] for idx in indices]

    mapped: List[str] = []
    for idx in indices:
        if 0 <= idx < n:
            mapped.append(choice_labels[idx])
        elif 1 <= idx <= n:
            mapped.append(choice_labels[idx - 1])
    return mapped


def _derive_gold_letters_generic(
    row: Mapping[str, object],
    choice_labels: List[str],
    option_texts: Mapping[str, str],
) -> List[str]:
    answer_value = _first_present(
        row,
        [
            "answer",
            "answers",
            "correct_answer",
            "correct_answers",
            "label",
            "target",
        ],
    )
    gold_value = _first_present(
        row,
        [
            "gold",
            "gold_indices",
            "gold_index",
            "answer_index",
            "answer_indices",
            "correct_option_index",
            "correct_option_indices",
        ],
    )

    labels: List[str] = []
    gold_indices = parse_gold_indices(gold_value)
    labels.extend(_map_indices_to_labels(gold_indices, choice_labels))

    answer_indices = parse_gold_indices(answer_value)
    labels.extend(_map_indices_to_labels(answer_indices, choice_labels))

    by_letter = parse_answer_letters(answer_value)
    valid_labels = set(choice_labels)
    labels.extend([label for label in by_letter if label in valid_labels])

    answer_candidates = _safe_list_parse(answer_value)
    if not answer_candidates and answer_value is not None:
        answer_candidates = [answer_value]

    for candidate in answer_candidates:
        text = str(candidate).strip().lower()
        if text and text in valid_labels:
            labels.append(text)

    # Only match by raw option text when we still have no label-based signal.
    if not labels:
        option_text_lookup = {str(v).strip().lower(): str(k) for k, v in option_texts.items()}
        for candidate in answer_candidates:
            text = str(candidate).strip().lower()
            if not text:
                continue
            mapped_from_text = option_text_lookup.get(text)
            if mapped_from_text:
                labels.append(mapped_from_text)

    filtered = [label for label in labels if label in valid_labels]
    return deduplicate_keep_order(filtered)  # type: ignore[return-value]


def _matches_question_type(value: object, required_type: str) -> bool:
    text = str(value).strip().lower()
    required = required_type.strip().lower()
    if not text:
        return False
    if required == "mcq":
        return "mcq" in text or "multiple choice" in text
    return required in text


def _extract_option_columns(row_map: Mapping[str, object]) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for label in ["a", "b", "c", "d"]:
        col = _find_column_name(row_map, f"option{label}")
        if col is None:
            continue
        value = row_map.get(col)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        options[label] = text
    # Fallback for datasets using numbered option columns.
    if not options:
        for idx in range(1, 7):
            col = _find_column_name(row_map, f"option{idx}")
            if col is None:
                continue
            value = row_map.get(col)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            options[_label_from_index(idx - 1)] = text
    return options


def _empty_questions_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "id",
            "question",
            "query",
            "choice_labels",
            "option_texts",
            "gold_letters",
            "dataset_source",
        ]
    )


def load_finmmeval_questions(
    input_path: str | None,
    cache_dir: str | Path = "data/raw/hf_cache",
    english_only: bool = True,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    if input_path is None:
        source_path = ensure_english_parquet(cache_dir=cache_dir)
    else:
        source_path = Path(input_path)

    if source_path.suffix == ".parquet":
        raw = pd.read_parquet(source_path)
    elif source_path.suffix == ".csv":
        raw = pd.read_csv(source_path)
    elif source_path.suffix in {".jsonl", ".json"}:
        raw = pd.read_json(source_path, lines=source_path.suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported input format: {source_path.suffix}")

    rows = []
    for _, row in raw.iterrows():
        sample_id = str(row.get("id", "") or "").strip()
        source_sheet = str(row.get("source_sheet", "") or "").strip().lower()
        if english_only and source_sheet and source_sheet != "english":
            continue
        if english_only and not source_sheet and sample_id.startswith("zh_"):
            continue

        query = str(row.get("query", "") or "")
        question = _select_question_text(row)
        choice_labels = parse_choices(row.get("choices"))
        option_texts = _build_option_texts(choice_labels, query)
        if not choice_labels:
            choice_labels = sorted(option_texts.keys())

        gold_letters = _derive_gold_letters(choice_labels, row)

        rows.append(
            {
                "id": sample_id,
                "question": question,
                "query": query,
                "choice_labels": choice_labels,
                "option_texts": option_texts,
                "gold_letters": gold_letters,
                "dataset_source": "finmmeval",
            }
        )

    if not rows:
        return _empty_questions_df()

    cleaned = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)
    return cleaned


def load_bbf_questions(
    bbf_language: str = "English",
    bbf_split: str = "test",
    question_type: str = "MCQ",
    token: bool | str | None = None,
) -> pd.DataFrame:
    try:
        ds = load_dataset(
            BBF_REPO,
            data_dir=bbf_language,
            split=bbf_split,
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load BhashaBench-Finance. The dataset is gated: run "
            "`./.venv/bin/hf auth login` or set `HF_TOKEN`, then retry."
        ) from exc

    raw = ds.to_pandas()
    if raw.empty:
        return _empty_questions_df()

    normalized_columns = {
        re.sub(r"[^a-z0-9]+", "", col.lower()): col for col in raw.columns
    }
    question_type_col = None
    for key in ["questiontype", "type", "qtype"]:
        if key in normalized_columns:
            question_type_col = normalized_columns[key]
            break
    if question_type_col is None:
        raise ValueError(
            "BBF dataset does not expose a recognizable `question_type` column."
        )

    filtered = raw[raw[question_type_col].map(lambda v: _matches_question_type(v, question_type))]
    rows = []

    for row_idx, (_, row) in enumerate(filtered.iterrows()):
        row_map = row.to_dict()
        sample_id_value = _first_present(
            row_map,
            ["id", "question_id", "qid", "uid", "sample_id"],
        )
        sample_id_base = str(sample_id_value).strip() if sample_id_value is not None else ""
        sample_id = sample_id_base or f"bbf_{bbf_language.lower()}_{bbf_split}_{row_idx}"

        question_value = _first_present(
            row_map,
            ["question", "question_text", "prompt", "text", "query", "input"],
        )
        query_value = _first_present(row_map, ["query", "prompt", "question", "question_text", "text"])
        question = "" if question_value is None else str(question_value).strip()
        query = "" if query_value is None else str(query_value)

        options_value = _first_present(
            row_map,
            ["options", "choices", "option_list", "candidate_options", "answers_options"],
        )
        option_columns = _extract_option_columns(row_map)
        if option_columns:
            options_value = option_columns
        extracted = _extract_choice_labels_and_texts(
            options_value,
            query=query,
            include_query_options=not bool(option_columns),
        )
        choice_labels = extracted["choice_labels"]
        option_texts = extracted["option_texts"]
        if not choice_labels:
            continue

        gold_letters = _derive_gold_letters_generic(
            row=row_map,
            choice_labels=choice_labels,
            option_texts=option_texts,
        )

        rows.append(
            {
                "id": sample_id,
                "question": question,
                "query": query,
                "choice_labels": choice_labels,
                "option_texts": option_texts,
                "gold_letters": gold_letters,
                "dataset_source": "bbf",
            }
        )

    if not rows:
        return _empty_questions_df()
    return pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)


def load_questions(
    input_path: str | None,
    cache_dir: str | Path = "data/raw/hf_cache",
    english_only: bool = True,
    source: str = "finmmeval",
    bbf_language: str = "English",
    bbf_split: str = "test",
    bbf_question_type: str = "MCQ",
    bbf_token: bool | str | None = None,
) -> pd.DataFrame:
    if input_path is not None:
        return load_finmmeval_questions(
            input_path=input_path,
            cache_dir=cache_dir,
            english_only=english_only,
        )

    source_normalized = source.strip().lower()
    if source_normalized == "finmmeval":
        return load_finmmeval_questions(
            input_path=None,
            cache_dir=cache_dir,
            english_only=english_only,
        )
    if source_normalized == "bbf":
        return load_bbf_questions(
            bbf_language=bbf_language,
            bbf_split=bbf_split,
            question_type=bbf_question_type,
            token=bbf_token,
        )
    if source_normalized == "both":
        first = load_finmmeval_questions(
            input_path=None,
            cache_dir=cache_dir,
            english_only=english_only,
        )
        second = load_bbf_questions(
            bbf_language=bbf_language,
            bbf_split=bbf_split,
            question_type=bbf_question_type,
            token=bbf_token,
        )
        if first.empty and second.empty:
            return _empty_questions_df()
        combined = pd.concat([first, second], ignore_index=True)
        return combined.drop_duplicates(subset=["id"]).reset_index(drop=True)

    raise ValueError(f"Unsupported source: {source}. Use finmmeval|bbf|both.")


def build_option_level_frame(questions_df: pd.DataFrame, with_targets: bool = True) -> pd.DataFrame:
    rows = []
    for _, row in questions_df.iterrows():
        correct_set = set(row["gold_letters"]) if with_targets else set()
        for label in row["choice_labels"]:
            option_text = row["option_texts"][label]
            rows.append(
                {
                    "id": row["id"],
                    "label": label,
                    "question": row["question"],
                    "option_text": option_text,
                    "feature_text": (
                        f"question: {row['question']}\n"
                        f"option_{label}: {option_text}"
                    ),
                    "target": int(label in correct_set) if with_targets else None,
                }
            )
    return pd.DataFrame(rows)


def save_questions_jsonl(questions_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_rows: List[dict] = []
    for _, row in questions_df.iterrows():
        serializable_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "query": row["query"],
                "choice_labels": list(row["choice_labels"]),
                "option_texts": dict(row["option_texts"]),
                "gold_letters": list(row["gold_letters"]),
                "dataset_source": str(row.get("dataset_source", "unknown")),
            }
        )

    with output_path.open("w", encoding="utf-8") as fh:
        for item in serializable_rows:
            fh.write(json.dumps(item, ensure_ascii=True) + "\n")


def load_questions_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            item["choice_labels"] = [str(x) for x in item.get("choice_labels", [])]
            item["gold_letters"] = [str(x) for x in item.get("gold_letters", [])]
            item["option_texts"] = {
                str(k): str(v) for k, v in item.get("option_texts", {}).items()
            }
            item["dataset_source"] = str(item.get("dataset_source", "unknown"))
            rows.append(item)
    return pd.DataFrame(rows)


def labels_to_answer_string(labels: Iterable[str]) -> str:
    normalized = [str(label).lower() for label in labels if str(label).strip()]
    normalized = deduplicate_keep_order(normalized)  # type: ignore[assignment]
    if len(normalized) <= 1:
        return normalized[0] if normalized else ""
    return json.dumps(normalized, ensure_ascii=True)
