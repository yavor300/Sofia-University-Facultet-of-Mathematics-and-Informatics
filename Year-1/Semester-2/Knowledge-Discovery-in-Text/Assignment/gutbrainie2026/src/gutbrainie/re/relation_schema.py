"""Relation schema definitions for GutBrainIE T621."""

from __future__ import annotations

from collections.abc import Mapping

VALID_RELATIONS: dict[tuple[str, str], list[str]] = {
    ("DDF", "DDF"): ["affect", "is a"],
    ("DDF", "anatomical location"): ["strike"],
    ("DDF", "animal"): ["target"],
    ("DDF", "bacteria"): ["change abundance"],
    ("DDF", "chemical"): ["interact"],
    ("DDF", "human"): ["target"],
    ("DDF", "microbiome"): ["change abundance"],
    ("anatomical location", "animal"): ["located in"],
    ("anatomical location", "human"): ["located in"],
    ("animal", "biomedical technique"): ["used by"],
    ("bacteria", "DDF"): ["influence"],
    ("bacteria", "animal"): ["located in"],
    ("bacteria", "bacteria"): ["interact"],
    ("bacteria", "chemical"): ["interact"],
    ("bacteria", "drug"): ["interact"],
    ("bacteria", "gene"): ["change expression"],
    ("bacteria", "human"): ["located in"],
    ("bacteria", "microbiome"): ["part of"],
    ("chemical", "DDF"): ["influence"],
    ("chemical", "anatomical location"): ["located in"],
    ("chemical", "animal"): ["administered", "located in"],
    ("chemical", "bacteria"): ["impact"],
    ("chemical", "chemical"): ["interact", "part of"],
    ("chemical", "gene"): ["change expression"],
    ("chemical", "human"): ["administered", "located in"],
    ("chemical", "microbiome"): ["impact", "produced by"],
    ("dietary supplement", "DDF"): ["influence"],
    ("dietary supplement", "animal"): ["administered"],
    ("dietary supplement", "bacteria"): ["impact"],
    ("dietary supplement", "gene"): ["change expression"],
    ("dietary supplement", "human"): ["administered"],
    ("dietary supplement", "microbiome"): ["impact"],
    ("drug", "DDF"): ["change effect"],
    ("drug", "animal"): ["administered"],
    ("drug", "bacteria"): ["impact"],
    ("drug", "chemical"): ["interact"],
    ("drug", "drug"): ["interact"],
    ("drug", "gene"): ["change expression"],
    ("drug", "human"): ["administered"],
    ("drug", "microbiome"): ["impact"],
    ("food", "DDF"): ["influence"],
    ("food", "animal"): ["administered"],
    ("food", "bacteria"): ["impact"],
    ("food", "gene"): ["change expression"],
    ("food", "human"): ["administered"],
    ("food", "microbiome"): ["impact"],
    ("human", "biomedical technique"): ["used by"],
    ("microbiome", "DDF"): ["is linked to"],
    ("microbiome", "anatomical location"): ["located in"],
    ("microbiome", "animal"): ["located in"],
    ("microbiome", "biomedical technique"): ["used by"],
    ("microbiome", "gene"): ["change expression"],
    ("microbiome", "human"): ["located in"],
    ("microbiome", "microbiome"): ["compared to"],
}


def valid_predicates(
    subject_label: str,
    object_label: str,
    schema: Mapping[tuple[str, str], list[str]] | None = None,
) -> list[str]:
    """Return valid predicates for an ordered subject/object label pair."""
    return list((schema or VALID_RELATIONS).get((str(subject_label), str(object_label)), []))


def is_valid_label_pair(
    subject_label: str,
    object_label: str,
    schema: Mapping[tuple[str, str], list[str]] | None = None,
) -> bool:
    """Return whether an ordered subject/object label pair is in the relation schema."""
    return bool(valid_predicates(subject_label, object_label, schema))
