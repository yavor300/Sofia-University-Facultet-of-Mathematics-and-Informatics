# Experiment Log

This log records the current train-split diagnostic experiments. The train split is the only released split with usable gold answers; dev and test answers are hidden as `HIDDEN`. These scores are useful for checking task fit, fine-tuning behavior, copying, and repetition, but they are not final generalization scores.

Source report: `data/reports/train_finetune_comparison.json`

## Train Results

| ID | System | OCR | Preprocess | OCR correction | Retrieval | LLM | Fine-tuned | n | EM | norm_EM | token_F1 | char_sim | non_empty | train_copy | repeated | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E01 | base | tesseract | resize_only | clean OCR only | none | Qwen2.5-7B-Instruct | no | 300 | 0.0267 | 0.0367 | 0.1342 | 0.3510 | 1.0000 | 0.0567 | 0.0268 | Prompt-only baseline is weak; outputs are non-empty but rarely match expected short answers. |
| E02 | base_rag | tesseract | resize_only | clean OCR only | BM25 k=3 over train index | Qwen2.5-7B-Instruct | no | 300 | 0.0967 | 0.1133 | 0.2230 | 0.4003 | 1.0000 | 0.1533 | 0.1137 | RAG improves the base model, but train-copy and repetition rise. |
| E03 | finetuned | tesseract | resize_only | clean OCR only | none | Qwen2.5-7B-Instruct + LoRA | yes | 300 | 0.2967 | 0.3100 | 0.4718 | 0.6100 | 1.0000 | 0.3267 | 0.1267 | Fine-tuning gives the strongest clean improvement; best token_F1. |
| E04 | finetuned_rag | tesseract | resize_only | clean OCR only | BM25 k=3 over train index | Qwen2.5-7B-Instruct + LoRA | yes | 300 | 0.3000 | 0.3300 | 0.4683 | 0.6116 | 1.0000 | 0.3500 | 0.1839 | Best train EM/norm_EM, but most optimistic because train RAG can retrieve training answers. |

## Interpretation

Fine-tuning is the main improvement. The base model reaches only `0.0367` normalized EM, while the fine-tuned model reaches `0.3100`. RAG also helps the base model, improving normalized EM from `0.0367` to `0.1133`, but it increases copying from train answers.

The fine-tuned RAG system has the highest train normalized EM (`0.3300`), but its token F1 is slightly below the non-RAG fine-tuned system (`0.4683` vs `0.4718`) and it has the highest train-copy/repetition rates. For a cleaner model choice, `finetuned` is the safer diagnostic winner; `finetuned_rag` is a candidate only if leaderboard or manual inspection confirms that retrieval helps on hidden data.

## Planned Next Experiments

| ID | OCR | Preprocess | OCR correction | Retrieval | LLM | Fine-tuned | Score | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E05 | best-of-ocr | ensemble | no | BM25 k=3 | Qwen2.5-7B-Instruct | no | TBD | Compare stronger OCR with base RAG. |
| E06 | best-of-ocr | ensemble | no | none | Qwen2.5-7B-Instruct + LoRA | yes | TBD | Compare fine-tuned model with stronger OCR. |
| E07 | best-of-ocr | ensemble | no | BM25/hybrid k=3 | Qwen2.5-7B-Instruct + LoRA | yes | TBD | Final candidate for leaderboard/manual inspection. |

## Useful Artifacts

- Train comparison report: `data/reports/train_finetune_comparison.json`
- Train OCR outputs: `data/processed/train_ocr_outputs_cleaned.jsonl`
- Train retrieval index: `data/processed/train_retrieval_index.jsonl`
- Base predictions: `data/processed/train_base_llm.json`
- Base RAG predictions: `data/processed/train_base_llm_rag.json`
- Fine-tuned predictions: `data/processed/train_ft_llm.json`
- Fine-tuned RAG predictions: `data/processed/train_ft_llm_rag.json`
