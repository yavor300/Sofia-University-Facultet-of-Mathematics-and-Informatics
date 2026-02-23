from elasticsearch import Elasticsearch
from config import ES_URL, ES_USER, ES_PASS, ES_INDEX


def client() -> Elasticsearch:
    if ES_USER and ES_PASS:
        return Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS))
    return Elasticsearch(ES_URL)


def _fields_for_lang(lang: str):
    """
    Returns:
      fields_with_boost: list[str] like ["title_en^3", "body_en", ...]
      fields_plain: list[str] like ["title_en", "body_en", ...] (no boosts)
      highlight_fields: list[str] like ["body_en", ...]
    """
    lang = (lang or "ALL").upper()

    if lang == "EN":
        return ["title_en^3", "body_en"], ["title_en", "body_en"], ["body_en"]
    if lang == "RU":
        return ["title_ru^3", "body_ru"], ["title_ru", "body_ru"], ["body_ru"]

    return (
        ["title_en^3", "body_en", "title_ru^3", "body_ru"],
        ["title_en", "body_en", "title_ru", "body_ru"],
        ["body_en", "body_ru"],
    )


def search(
    q: str,
    lang: str = "ALL",
    size: int = 10,
    fuzzy: bool = False,
    exact: bool = False
):
    """
    Multilingual search with optional fuzzy typos and exact phrase mode.

    lang:
      - EN  -> title_en/body_en
      - RU  -> title_ru/body_ru
      - ALL -> both

    fuzzy:
      - Applies only when exact=False
      - Uses fuzziness=AUTO for typo tolerance

    exact:
      - Uses match_phrase over the selected fields
      - When exact=True, fuzzy is ignored
    """
    es = client()

    fields_boosted, fields_plain, highlight_fields = _fields_for_lang(lang)

    if exact:
        query = {
            "bool": {
                "should": [{"match_phrase": {f: q}} for f in fields_plain],
                "minimum_should_match": 1
            }
        }
    else:
        multi_match = {
            "query": q,
            "fields": fields_boosted,
            "type": "best_fields",
            "minimum_should_match": "70%",
        }

        if fuzzy:
            multi_match.update({
                "fuzziness": "AUTO",
                "prefix_length": 1,
                "max_expansions": 50,
            })

        query = {"multi_match": multi_match}

    query_body = {
        "query": query,
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {}
        }
    }

    for field in highlight_fields:
        query_body["highlight"]["fields"][field] = {
            "fragment_size": 160,
            "number_of_fragments": 2
        }

    resp = es.search(index=ES_INDEX, body=query_body, size=size)

    results = []
    for hit in resp["hits"]["hits"]:
        src = hit.get("_source", {})
        hl = hit.get("highlight", {})

        snippet = ""
        for f in highlight_fields:
            if f in hl:
                snippet = " ... ".join(hl[f])
                break

        title = (src.get("title_en") or src.get("title_ru") or "")

        results.append({
            "id": hit.get("_id", ""),
            "score": hit.get("_score", 0.0),
            "title": title,
            "path": src.get("path", ""),
            "language": src.get("language", ""),
            "snippet": snippet
        })

    return results

def more_like_this(doc_id: str, size: int = 10):
    """
    Find documents similar to a given document id using Elasticsearch More Like This query.

    Parameters:
      doc_id: Elasticsearch _id of the reference document
      size: number of similar docs to return

    Returns:
      List of dicts: {id, score, title, path, language, snippet}
    """
    if not doc_id:
        return []

    es = client()

    query_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "more_like_this": {
                            "fields": ["body_en", "body_ru"],
                            "like": [
                                {"_index": ES_INDEX, "_id": doc_id}
                            ],
                            "min_term_freq": 2,
                            "min_doc_freq": 5,
                            "max_query_terms": 25
                        }
                    }
                ],
                "must_not": [
                    {"ids": {"values": [doc_id]}}
                ]
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {
                "body_en": {"fragment_size": 160, "number_of_fragments": 2},
                "body_ru": {"fragment_size": 160, "number_of_fragments": 2},
            }
        }
    }

    resp = es.search(index=ES_INDEX, body=query_body, size=size)

    results = []
    for hit in resp["hits"]["hits"]:
        src = hit.get("_source", {})
        hl = hit.get("highlight", {})

        snippet = ""
        if "body_en" in hl:
            snippet = " ... ".join(hl["body_en"])
        elif "body_ru" in hl:
            snippet = " ... ".join(hl["body_ru"])

        title = (src.get("title_en") or src.get("title_ru") or "")

        results.append({
            "id": hit.get("_id", ""),
            "score": hit.get("_score", 0.0),
            "title": title,
            "path": src.get("path", ""),
            "language": src.get("language", ""),
            "snippet": snippet
        })

    return results
