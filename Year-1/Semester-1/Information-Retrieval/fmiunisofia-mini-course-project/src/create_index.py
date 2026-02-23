from elasticsearch import Elasticsearch
from config import ES_URL, ES_USER, ES_PASS, ES_INDEX

def client() -> Elasticsearch:
    if ES_USER and ES_PASS:
        return Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASS))
    return Elasticsearch(ES_URL)

MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "en_text": {"type": "english"},
                "ru_text": {"type": "russian"},
            }
        }
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "path": {"type": "keyword"},
            "language": {"type": "keyword"},

            "title_en": {"type": "text", "analyzer": "en_text"},
            "body_en": {"type": "text", "analyzer": "en_text"},

            "title_ru": {"type": "text", "analyzer": "ru_text"},
            "body_ru": {"type": "text", "analyzer": "ru_text"},
        }
    }
}

if __name__ == "__main__":
    es = client()

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)

    es.indices.create(index=ES_INDEX, body=MAPPING)

    print(f"Created index: {ES_INDEX}")
