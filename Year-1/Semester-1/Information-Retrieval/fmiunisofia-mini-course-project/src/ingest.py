import argparse
from pathlib import Path
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from config import ES_URL, ES_USER, ES_PASS, ES_INDEX


def client() -> Elasticsearch:
    if ES_USER and ES_PASS:
        return Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS))
    return Elasticsearch(ES_URL)


def read_text_file(fp: Path, lang: str) -> tuple[str, str]:
    """
    Returns (title, body) from a .txt file.

    - title: first non-empty line, else filename stem
    - body: full text
    - encoding: for RU try utf-8 then cp1251; for EN use utf-8
    """
    encodings = ["utf-8"]
    if lang.upper() == "RU":
        encodings = ["utf-8", "cp1251"]

    text = ""
    for enc in encodings:
        try:
            text = fp.read_text(encoding=enc, errors="strict")
            break
        except UnicodeDecodeError:
            continue

    if not text:
        text = fp.read_text(encoding="utf-8", errors="ignore")

    text = text.strip()
    if not text:
        return fp.stem, ""

    lines = [ln.strip() for ln in text.splitlines()]
    title = next((ln for ln in lines if ln), fp.stem)
    return title, text


def to_doc(fp: Path, root_dir: Path, lang: str) -> dict:
    """
    Converts a single .txt file to an Elasticsearch document.

    Writes content into language-specific fields:
      EN -> title_en/body_en
      RU -> title_ru/body_ru

    Keeps:
      id: stable identifier based on relative path
      path: relative path for display in ui
      language: EN/RU
    """
    lang = lang.upper()
    title, body = read_text_file(fp, lang)

    rel_path = str(fp.relative_to(root_dir))
    doc_id = f"{lang}:{rel_path}"

    doc = {
        "id": doc_id,
        "path": rel_path,
        "language": lang,
        "title_en": "",
        "body_en": "",
        "title_ru": "",
        "body_ru": ""
    }

    if lang == "EN":
        doc["title_en"] = title
        doc["body_en"] = body
    elif lang == "RU":
        doc["title_ru"] = title
        doc["body_ru"] = body
    else:
        raise ValueError("Unsupported --lang. Use EN or RU.")

    return doc


def iter_docs(root_dir: Path, lang: str):
    for fp in root_dir.rglob("*.txt"):
        yield to_doc(fp, root_dir, lang)


def to_actions(docs):
    for d in docs:
        yield {
            "_op_type": "index",
            "_index": ES_INDEX,
            "_id": d["id"],
            "_source": d
        }


def main():
    ap = argparse.ArgumentParser(description="Ingest TXT corpus into Elasticsearch")
    ap.add_argument("--dir", required=True, help="Directory containing .txt files (recursive)")
    ap.add_argument("--lang", required=True, choices=["EN", "RU"], help="Language of the corpus")
    ap.add_argument("--max_docs", type=int, default=0, help="Limit number of docs for quick testing (0 = no limit)")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    docs_iter = iter_docs(root, args.lang)

    if args.max_docs > 0:
        docs = []
        for i, d in enumerate(docs_iter, start=1):
            docs.append(d)
            if i >= args.max_docs:
                break
    else:
        docs = list(docs_iter)

    print(f"Found {len(docs)} .txt documents under: {root} (lang={args.lang})")

    es = client()
    helpers.bulk(es, to_actions(tqdm(docs)))
    es.indices.refresh(index=ES_INDEX)

    print("Ingestion done.")


if __name__ == "__main__":
    main()
