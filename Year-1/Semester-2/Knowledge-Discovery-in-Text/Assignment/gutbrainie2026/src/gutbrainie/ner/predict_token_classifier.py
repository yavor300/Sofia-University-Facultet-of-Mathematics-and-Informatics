"""Token-classification NER prediction entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from gutbrainie.data.articles import load_articles_csv
from gutbrainie.ner.bio_tags import decode_bio_spans
from gutbrainie.ner.tokenizers import load_fast_tokenizer
from gutbrainie.submission.export_t611 import entities_to_t611_json


def predict_token_classifier_to_json(
    model_path: str | Path,
    articles_path: str | Path,
    output_path: str | Path,
    batch_size: int = 8,
    max_length: int = 512,
    use_cpu: bool = False,
) -> pd.DataFrame:
    """Run a trained token-classification model and write T611 JSON."""
    transformers, torch = _import_runtime()
    model_path = Path(model_path)
    tokenizer = load_fast_tokenizer(transformers, str(model_path))

    model = transformers.AutoModelForTokenClassification.from_pretrained(str(model_path))
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()
    id_to_label = _load_id_to_label(model_path, model)

    articles = load_articles_csv(articles_path)
    examples = [
        (str(article["pmid"]), location, str(article[location]))
        for _, article in articles.iterrows()
        for location in ("title", "abstract")
    ]

    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        texts = [text for _, _, text in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = tokenized.pop("offset_mapping").cpu().tolist()
        inputs = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            predictions = model(**inputs).logits.argmax(dim=-1).cpu().tolist()

        for (pmid, location, text), example_offsets, label_ids in zip(batch, offsets, predictions, strict=False):
            for span in decode_bio_spans(text, _offsets_as_tuples(example_offsets), label_ids, id_to_label):
                rows.append(
                    {
                        "pmid": pmid,
                        "start_idx": span["start_idx"],
                        "end_idx": span["end_idx"],
                        "location": location,
                        "text_span": span["text_span"],
                        "label": span["label"],
                    }
                )

    prediction_df = pd.DataFrame(rows, columns=["pmid", "start_idx", "end_idx", "location", "text_span", "label"])
    payload = entities_to_t611_json(prediction_df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return prediction_df


def _load_id_to_label(model_path: Path, model: Any) -> dict[int, str]:
    label_map_path = model_path / "label_map.json"
    if label_map_path.exists():
        payload = json.loads(label_map_path.read_text(encoding="utf-8"))
        return {int(key): value for key, value in payload["id_to_label"].items()}
    return {int(key): value for key, value in model.config.id2label.items()}


def _offsets_as_tuples(offsets: list[list[int]]) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in offsets]


def _import_runtime():
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency '{exc.name}'. Install project dependencies with: make install"
        ) from exc
    return transformers, torch
