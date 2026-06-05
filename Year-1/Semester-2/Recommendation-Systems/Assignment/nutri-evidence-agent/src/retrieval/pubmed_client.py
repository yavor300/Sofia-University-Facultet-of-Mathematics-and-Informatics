"""PubMed retrieval client built on Biopython Entrez."""

from __future__ import annotations

import re
from typing import Any

from Bio import Entrez


class PubMedClient:
    """Small wrapper around NCBI Entrez for normalized PubMed article records."""

    def __init__(self, email: str, api_key: str | None = None):
        if not email or not email.strip():
            raise ValueError("An NCBI email is required for live PubMed retrieval.")

        Entrez.email = email.strip()
        Entrez.tool = "NutriEvidenceAgent"
        if api_key:
            Entrez.api_key = api_key.strip()

    def search(self, query: str, max_results: int = 50) -> list[str]:
        """Search PubMed and return deduplicated PMIDs ordered by relevance."""
        if not query or not query.strip():
            return []

        max_results = max(0, int(max_results))
        if max_results == 0:
            return []

        with Entrez.esearch(
            db="pubmed",
            term=query.strip(),
            retmax=max_results,
            sort="relevance",
        ) as handle:
            result = Entrez.read(handle)

        return _dedupe_strings(result.get("IdList", []))

    def fetch_details(self, pmids: list[str]) -> list[dict]:
        """Fetch and normalize PubMed article metadata for a list of PMIDs."""
        unique_pmids = _dedupe_strings(pmids)
        if not unique_pmids:
            return []

        with Entrez.efetch(
            db="pubmed",
            id=",".join(unique_pmids),
            rettype="medline",
            retmode="xml",
        ) as handle:
            records = Entrez.read(handle)

        articles: list[dict] = []
        seen: set[str] = set()

        for record in records.get("PubmedArticle", []):
            try:
                article = _parse_pubmed_record(record)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue

            pmid = article.get("pmid")
            if not pmid or pmid in seen:
                continue

            seen.add(pmid)
            articles.append(article)

        return articles

    def search_and_fetch(self, query: str, max_results: int = 50) -> list[dict]:
        """Search PubMed, fetch article details, and attach the source query."""
        pmids = self.search(query=query, max_results=max_results)
        articles = self.fetch_details(pmids)

        for article in articles:
            article["source_query"] = query

        return articles


def _parse_pubmed_record(record: dict[str, Any]) -> dict:
    citation = record.get("MedlineCitation", {})
    article_data = citation.get("Article", {})
    pubmed_data = record.get("PubmedData", {})

    pmid = _text(citation.get("PMID"))
    if not pmid:
        raise ValueError("PubMed record is missing PMID.")

    return {
        "pmid": pmid,
        "title": _text(article_data.get("ArticleTitle")),
        "abstract": _extract_abstract(article_data),
        "year": _extract_year(article_data),
        "journal": _extract_journal(article_data),
        "authors": _extract_authors(article_data),
        "publication_types": _extract_publication_types(article_data),
        "mesh_terms": _extract_mesh_terms(citation),
        "doi": _extract_doi(article_data, pubmed_data),
        "source_query": "",
    }


def _extract_abstract(article_data: dict[str, Any]) -> str:
    abstract = article_data.get("Abstract", {})
    abstract_text = abstract.get("AbstractText", []) if isinstance(abstract, dict) else []
    parts: list[str] = []

    for item in _as_list(abstract_text):
        text = _text(item)
        if not text:
            continue

        label = _attribute(item, "Label")
        parts.append(f"{label}: {text}" if label else text)

    return " ".join(parts)


def _extract_year(article_data: dict[str, Any]) -> int | None:
    for article_date in _as_list(article_data.get("ArticleDate", [])):
        year = _to_int(_text(article_date.get("Year")) if isinstance(article_date, dict) else "")
        if year:
            return year

    journal = article_data.get("Journal", {})
    issue = journal.get("JournalIssue", {}) if isinstance(journal, dict) else {}
    pub_date = issue.get("PubDate", {}) if isinstance(issue, dict) else {}

    year = _to_int(_text(pub_date.get("Year")) if isinstance(pub_date, dict) else "")
    if year:
        return year

    medline_date = _text(pub_date.get("MedlineDate")) if isinstance(pub_date, dict) else ""
    match = re.search(r"(18|19|20)\d{2}", medline_date)
    return int(match.group(0)) if match else None


def _extract_journal(article_data: dict[str, Any]) -> str:
    journal = article_data.get("Journal", {})
    if not isinstance(journal, dict):
        return ""

    return _text(journal.get("Title")) or _text(journal.get("ISOAbbreviation"))


def _extract_authors(article_data: dict[str, Any]) -> list[str]:
    authors: list[str] = []

    for author in _as_list(article_data.get("AuthorList", [])):
        if not isinstance(author, dict):
            continue

        collective_name = _text(author.get("CollectiveName"))
        if collective_name:
            authors.append(collective_name)
            continue

        last_name = _text(author.get("LastName"))
        fore_name = _text(author.get("ForeName")) or _text(author.get("Initials"))
        name = " ".join(part for part in [fore_name, last_name] if part)
        if name:
            authors.append(name)

    return authors


def _extract_publication_types(article_data: dict[str, Any]) -> list[str]:
    return _dedupe_strings(_text(item) for item in _as_list(article_data.get("PublicationTypeList", [])))


def _extract_mesh_terms(citation: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    for heading in _as_list(citation.get("MeshHeadingList", [])):
        if not isinstance(heading, dict):
            continue

        descriptor = _text(heading.get("DescriptorName"))
        if descriptor:
            terms.append(descriptor)

    return _dedupe_strings(terms)


def _extract_doi(article_data: dict[str, Any], pubmed_data: dict[str, Any]) -> str:
    for article_id in _as_list(pubmed_data.get("ArticleIdList", [])):
        if _attribute(article_id, "IdType").lower() == "doi":
            return _text(article_id)

    for elocation_id in _as_list(article_data.get("ELocationID", [])):
        if _attribute(elocation_id, "EIdType").lower() == "doi":
            return _text(elocation_id)

    return ""


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _attribute(value: Any, name: str) -> str:
    attributes = getattr(value, "attributes", {})
    if not isinstance(attributes, dict):
        return ""

    return _text(attributes.get(name))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe_strings(values: Any) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)

    return deduped
