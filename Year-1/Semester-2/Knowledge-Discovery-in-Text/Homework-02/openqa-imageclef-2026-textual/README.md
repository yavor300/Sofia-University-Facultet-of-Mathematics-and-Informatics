# ImageCLEF 2026 OpenQA Textual

This repository contains a local solution pipeline for the **ImageCLEF-MR2026 OpenQA Textual** task. The input is an image containing a textual question; the output is a short answer in the required JSON submission format.

```text
question image
  -> OCR
  -> OCR cleanup / question normalization
  -> optional retrieval over train examples
  -> local instruction-tuned LLM answer generation
  -> answer postprocessing
  -> submission JSON / ZIP
```

The current implementation supports OCR diagnostics, LLM prediction, BM25/dense/hybrid retrieval, QLoRA fine-tuning, local evaluation, experiment logging, submission validation, and ZIP creation.

## Current Status

The implemented final candidate is based on:

- dataset: `SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual`
- OCR: Tesseract by default, with language-specific settings
- preprocessing: `resize_only` by default, with `raw`, `contrast`, and `binarized` variants available
- base LLM: `Qwen/Qwen2.5-7B-Instruct`
- optional fine-tuning: LoRA/QLoRA adapter
- optional RAG: BM25, dense, or hybrid retrieval over train examples

Important evaluation caveat: released `dev` and `test` answers are hidden as `HIDDEN`, so real local metric evaluation is only possible on the `train` split. Train metrics are diagnostic and should not be reported as final hidden-test generalization scores.

Best train diagnostic result so far:

| System | n | EM | norm_EM | token_F1 | char_sim |
| --- | ---: | ---: | ---: | ---: | ---: |
| fine-tuned + BM25 RAG | 300 | 0.3000 | 0.3300 | 0.4683 | 0.6116 |
| fine-tuned | 300 | 0.2967 | 0.3100 | 0.4718 | 0.6100 |
| base + BM25 RAG | 300 | 0.0967 | 0.1133 | 0.2230 | 0.4003 |
| base | 300 | 0.0267 | 0.0367 | 0.1342 | 0.3510 |

See:

- `experiments/experiment_log.md`
- `FINAL_SYSTEM_DESCRIPTION_DRAFT.md`
- `data/reports/train_finetune_comparison.json` if generated locally

## Setup

Use Python 3.10+ or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest
```

Optional OCR and GPU dependencies are separated because OCR and CUDA stacks can conflict across platforms:

```bash
pip install -r requirements-ocr.txt
pip install -r requirements-gpu.txt
```

Tesseract also requires a system binary. On Ubuntu/WSL:

```bash
sudo apt-get install tesseract-ocr
```

For multilingual OCR, install the relevant Tesseract language packs, for example `tesseract-ocr-eng`, `tesseract-ocr-bul`, `tesseract-ocr-hrv`, `tesseract-ocr-ita`, `tesseract-ocr-srp`, and `tesseract-ocr-chi-sim` where available.

Copy `.env.example` to `.env` if you need custom cache locations or Hugging Face settings.

## Project Layout

```text
configs/          YAML configuration for data, OCR, retrieval, generation, fine-tuning, RunPod
data/             local raw data cache, processed files, reports, submissions
experiments/      experiment log and run metadata
notebooks/        exploratory notebooks
src/openqa_textual/ core package: OCR, generation, retrieval, evaluation, submission utilities
src/scripts/      command-line entry points
tests/            unit and smoke tests
```

## Configuration

Main config files:

- `configs/data.yaml` - dataset name, cache path, split aliases
- `configs/ocr.yaml` - OCR engines, preprocessing variants, language mappings
- `configs/generation.yaml` - LLM name and decoding settings
- `configs/retrieval.yaml` - retrieval index and RAG settings
- `configs/finetune.yaml` - QLoRA/LoRA training settings
- `configs/runpod.yaml` - remote training/inference helper settings

The default dataset config maps:

```text
train -> train
dev   -> validation
test  -> test
```

## Common Commands

Inspect a split:

```bash
python -m src.scripts.inspect_dataset --split train --n 5
python -m src.scripts.inspect_dataset --split dev --n 5
python -m src.scripts.inspect_dataset --split test --n 5
```

Run OCR:

```bash
python -m src.scripts.run_ocr \
  --split train \
  --n 30 \
  --output data/processed/train_ocr_outputs.jsonl
```

Run OCR on a full split:

```bash
python -m src.scripts.run_ocr \
  --split train \
  --all \
  --output data/processed/train_ocr_outputs.jsonl
```

Clean OCR output:

```bash
python -m src.scripts.postprocess_ocr \
  --input data/processed/train_ocr_outputs.jsonl \
  --output data/processed/train_ocr_outputs_cleaned.jsonl
```

Build train retrieval index:

```bash
python -m src.scripts.build_retrieval_index \
  --ocr-jsonl data/processed/train_ocr_outputs_cleaned.jsonl \
  --output data/processed/train_retrieval_index.jsonl
```

Run end-to-end OCR + LLM prediction:

```bash
python -m src.scripts.predict \
  --split test \
  --output data/processed/test_predictions.json \
  --submission
```

Run end-to-end prediction with RAG:

```bash
python -m src.scripts.predict \
  --split test \
  --retrieval-index data/processed/train_retrieval_index.jsonl \
  --retrieval-method bm25 \
  --rag-k 3 \
  --output data/processed/test_predictions.json \
  --submission
```

Run generation from an existing OCR JSONL file:

```bash
python -m src.scripts.predict_llm \
  --ocr-jsonl data/processed/test_ocr_outputs_cleaned.jsonl \
  --output data/processed/test_predictions.json
```

Run generation with a LoRA adapter:

```bash
python -m src.scripts.predict \
  --split test \
  --adapter-path data/models/qwen25_7b_lora \
  --load-in-4bit \
  --output data/processed/test_predictions.json \
  --submission
```

## Fine-Tuning

Build chat-format SFT data from OCR and gold answers:

```bash
python -m src.scripts.build_training_data \
  --ocr-jsonl data/processed/train_ocr_outputs_cleaned.jsonl \
  --output data/processed/train_sft_ocr.jsonl
```

Validate the fine-tuning config without loading the model:

```bash
python -m src.scripts.train_qlora \
  --config configs/finetune.yaml \
  --dry-run
```

Run QLoRA training:

```bash
python -m src.scripts.train_qlora \
  --config configs/finetune.yaml
```

## Evaluation

Evaluate a single prediction file against a gold file:

```bash
python -m src.scripts.evaluate_dev \
  --pred data/processed/dev_predictions.json \
  --gold data/processed/dev_gold.json \
  --output data/reports/dev_metrics.json
```

Compare multiple systems against a dataset split with gold answers:

```bash
python -m src.scripts.evaluate_predictions \
  --split train \
  --system base=data/processed/train_base_llm.json \
  --system base_rag=data/processed/train_base_llm_rag.json \
  --system finetuned=data/processed/train_ft_llm.json \
  --system finetuned_rag=data/processed/train_ft_llm_rag.json \
  --output data/reports/train_finetune_comparison.json
```

The comparison report includes:

- exact match
- normalized exact match
- token F1
- character similarity
- non-empty answer rate
- train-answer copy rate
- repeated-answer rate
- per-language breakdown

## Experiment Logging

Record experiment metadata manually:

```bash
python -m src.scripts.log_experiment \
  --id E04 \
  --ocr tesseract \
  --preprocess resize_only \
  --ocr-correction "clean OCR only" \
  --retrieval "BM25 k=3" \
  --llm "Qwen2.5-7B-Instruct + LoRA" \
  --fine-tuned yes \
  --dev-score "train norm_EM=0.3300" \
  --notes "Best train normalized EM; higher train-copy rate."
```

The main experiment notes are kept in:

```text
experiments/experiment_log.md
```

## Submission

Create the final submission JSON from internal prediction records:

```bash
python -m src.scripts.make_submission \
  --pred data/processed/test_predictions.json \
  --output data/submissions/predictions.json
```

Validate the JSON against the expected test split:

```bash
python -m src.scripts.validate_submission \
  --submission data/submissions/predictions.json \
  --expected-size-from-split test
```

Create the ZIP file expected by the submission workflow:

```bash
python -m src.scripts.zip_submission \
  --submission data/submissions/predictions.json \
  --output data/submissions/openqa_textual_submission.zip
```

Validate the ZIP:

```bash
python -m src.scripts.validate_submission \
  --submission data/submissions/openqa_textual_submission.zip \
  --expected-size-from-split test
```

Final submission object format:

```json
[
  {
    "question_id": "...",
    "answers": ["..."],
    "language": "English"
  }
]
```

The final submission file must not contain debug fields.

## Main Artifacts

Generated locally during experiments:

- `data/processed/train_ocr_outputs_cleaned.jsonl`
- `data/processed/train_retrieval_index.jsonl`
- `data/processed/test_predictions.json`
- `data/submissions/predictions.json`
- `data/submissions/openqa_textual_submission.zip`
- `data/reports/train_finetune_comparison.json`

Repository documents:

- `experiments/experiment_log.md`
- `FINAL_SYSTEM_DESCRIPTION_DRAFT.md`

Most generated data, caches, models, reports, and submission files are intentionally ignored by Git.
