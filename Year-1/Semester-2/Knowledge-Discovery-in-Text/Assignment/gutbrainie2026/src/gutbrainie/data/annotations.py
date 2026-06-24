"""Annotation loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ENTITY_COLUMNS = ["pmid", "annotator", "start_idx", "end_idx", "location", "text_span", "label"]
MENTION_RELATION_COLUMNS = [
    "pmid",
    "annotator",
    "subject_text_span",
    "subject_label",
    "predicate",
    "object_text_span",
    "object_label",
]
FULL_RELATION_COLUMNS = [
    "pmid",
    "annotator",
    "subject_start_idx",
    "subject_end_idx",
    "subject_location",
    "subject_text_span",
    "subject_label",
    "predicate",
    "object_start_idx",
    "object_end_idx",
    "object_location",
    "object_text_span",
    "object_label",
]
ENTITY_DEDUP_COLUMNS = ["pmid", "location", "start_idx", "end_idx", "text_span", "label"]
ENTITY_LABELS = {
    "DDF",
    "anatomical location",
    "animal",
    "bacteria",
    "biomedical technique",
    "chemical",
    "dietary supplement",
    "drug",
    "food",
    "gene",
    "human",
    "microbiome",
    "statistical technique",
}
RELATION_PREDICATES = {
    "administered",
    "affect",
    "change abundance",
    "change effect",
    "change expression",
    "compared to",
    "impact",
    "influence",
    "interact",
    "is a",
    "is linked to",
    "located in",
    "part of",
    "produced by",
    "strike",
    "target",
    "used by",
}


def load_entities_csv(path: str | Path) -> pd.DataFrame:
    """Load entity annotations from a pipe-separated CSV file."""
    df = _load_entities_csv(path)
    return _coerce_int_columns(df, ["start_idx", "end_idx"])


def load_mention_relations_csv(path: str | Path) -> pd.DataFrame:
    """Load mention-level relation annotations from a pipe-separated CSV file."""
    return _load_mention_relations_csv(path)


def load_full_relations_csv(path: str | Path) -> pd.DataFrame:
    """Load offset-bearing relation annotations from a pipe-separated CSV file."""
    df = _load_full_relations_csv(path)
    return _coerce_int_columns(
        df,
        ["subject_start_idx", "subject_end_idx", "object_start_idx", "object_end_idx"],
    )


def deduplicate_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact entity duplicates according to the Phase 1 adjudication rule."""
    return df.drop_duplicates(subset=ENTITY_DEDUP_COLUMNS).reset_index(drop=True)


def _load_csv(path: str | Path, expected_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(Path(path), sep="|", dtype=str, keep_default_na=False)
    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing annotation columns in {path}: {missing}")

    df = df[expected_columns].copy()
    df["pmid"] = df["pmid"].astype(str)
    return df


def _load_entities_csv(path: str | Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r").split("|")
        if header != ENTITY_COLUMNS:
            missing = [column for column in ENTITY_COLUMNS if column not in header]
            raise ValueError(f"Unexpected entity header in {path}: missing {missing}")

        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n\r").split("|")
            if len(parts) < len(ENTITY_COLUMNS):
                raise ValueError(f"Too few entity fields in {path} line {line_number}: {line.rstrip()}")
            if len(parts) == len(ENTITY_COLUMNS):
                rows.append(parts)
                continue

            # Some bronze/silver spans contain raw pipe characters, e.g. "EE|MH group".
            rows.append([*parts[:5], "|".join(parts[5:-1]), parts[-1]])

    df = pd.DataFrame(rows, columns=ENTITY_COLUMNS)
    df["pmid"] = df["pmid"].astype(str)
    return df


def _load_mention_relations_csv(path: str | Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r").split("|")
        if header != MENTION_RELATION_COLUMNS:
            missing = [column for column in MENTION_RELATION_COLUMNS if column not in header]
            raise ValueError(f"Unexpected mention relation header in {path}: missing {missing}")

        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n\r").split("|")
            if len(parts) < len(MENTION_RELATION_COLUMNS):
                raise ValueError(f"Too few mention relation fields in {path} line {line_number}: {line.rstrip()}")
            rows.append(_parse_mention_relation_parts(parts, path, line_number))

    df = pd.DataFrame(rows, columns=MENTION_RELATION_COLUMNS)
    df["pmid"] = df["pmid"].astype(str)
    return df


def _load_full_relations_csv(path: str | Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r").split("|")
        if header != FULL_RELATION_COLUMNS:
            missing = [column for column in FULL_RELATION_COLUMNS if column not in header]
            raise ValueError(f"Unexpected full relation header in {path}: missing {missing}")

        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n\r").split("|")
            if len(parts) < len(FULL_RELATION_COLUMNS):
                raise ValueError(f"Too few full relation fields in {path} line {line_number}: {line.rstrip()}")
            rows.append(_parse_full_relation_parts(parts, path, line_number))

    df = pd.DataFrame(rows, columns=FULL_RELATION_COLUMNS)
    df["pmid"] = df["pmid"].astype(str)
    return df


def _parse_mention_relation_parts(parts: list[str], path: Path, line_number: int) -> list[str]:
    if len(parts) == len(MENTION_RELATION_COLUMNS):
        return parts

    if parts[-1] not in ENTITY_LABELS:
        raise ValueError(f"Unknown object label in {path} line {line_number}: {parts[-1]}")

    for subject_label_index in range(3, len(parts) - 3):
        subject_label = parts[subject_label_index]
        predicate = parts[subject_label_index + 1]
        if subject_label in ENTITY_LABELS and predicate in RELATION_PREDICATES:
            return [
                parts[0],
                parts[1],
                "|".join(parts[2:subject_label_index]),
                subject_label,
                predicate,
                "|".join(parts[subject_label_index + 2 : -1]),
                parts[-1],
            ]

    raise ValueError(f"Could not parse mention relation in {path} line {line_number}: {'|'.join(parts)}")


def _parse_full_relation_parts(parts: list[str], path: Path, line_number: int) -> list[str]:
    if len(parts) == len(FULL_RELATION_COLUMNS):
        return parts

    if parts[-1] not in ENTITY_LABELS:
        raise ValueError(f"Unknown object label in {path} line {line_number}: {parts[-1]}")

    for subject_label_index in range(6, len(parts) - 6):
        subject_label = parts[subject_label_index]
        predicate = parts[subject_label_index + 1]
        object_start = parts[subject_label_index + 2]
        object_end = parts[subject_label_index + 3]
        object_location = parts[subject_label_index + 4]
        if (
            subject_label in ENTITY_LABELS
            and predicate in RELATION_PREDICATES
            and object_start.isdigit()
            and object_end.isdigit()
            and object_location in {"title", "abstract"}
        ):
            return [
                parts[0],
                parts[1],
                parts[2],
                parts[3],
                parts[4],
                "|".join(parts[5:subject_label_index]),
                subject_label,
                predicate,
                object_start,
                object_end,
                object_location,
                "|".join(parts[subject_label_index + 5 : -1]),
                parts[-1],
            ]

    raise ValueError(f"Could not parse full relation in {path} line {line_number}: {'|'.join(parts)}")


def _coerce_int_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")
    return df
