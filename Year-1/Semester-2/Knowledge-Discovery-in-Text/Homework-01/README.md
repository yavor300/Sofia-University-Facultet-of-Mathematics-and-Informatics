# FinMMEval Homework 01 Solution

This project implements an English-only solution for:
- Course homework: **Automatic question answering**
- Competition task: **FinMMEval Task 1 - Financial Exam Q&A**

The code builds a full local pipeline:
1. Download and normalize the English part of the dataset.
2. Train and compare two approaches.
3. Generate a submission-ready CSV file.

It now also supports loading a second dataset:
- `bharatgenai/BhashaBench-Finance` (gated), with `MCQ` `question_type` filtering.

## Implemented approaches

- `lexical_baseline`: unsupervised lexical overlap ranker.
- `supervised_option_pair_classifier`: TF-IDF + Logistic Regression on `(question, option)` pairs.
- `transformer_cross_encoder`: pretrained Transformer fine-tuned on `(question, option)` binary relevance.

## Project structure

```text
Homework-01/
├── assignment.md
├── homework.md
├── requirements.txt
├── README.md
├── configs/
│   └── default.yaml
├── docs/
│   └── system_description.md
├── scripts/
│   └── run_pipeline.sh
├── src/
│   └── finmmeval_hw/
│       ├── __init__.py
│       ├── cli.py
│       ├── data.py
│       ├── evaluation.py
│       └── modeling.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── results/
```

## Environment setup

Use the already-created virtual environment:

```bash
cd Year-1/Semester-2/Knowledge-Discovery-in-Text/Homework-01
./.venv/bin/pip install -r requirements.txt
```

## Run end-to-end

```bash
scripts/run_pipeline.sh
```

This executes:
- `prepare`
- `train`
- `predict`

## Manual commands

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli prepare
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli train
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli evaluate
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli predict
```

Train with transformer cross-encoder:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli train \
  --input data/processed/english_questions_finmmeval.jsonl \
  --model-type transformer \
  --transformer-model-name distilbert-base-uncased \
  --model-out models/option_pair_transformer_finmmeval \
  --metrics-out results/dev_metrics_finmmeval_transformer_compare.json
```

## BhashaBench-Finance (English MCQ only)

This dataset is gated. First authenticate:

```bash
./.venv/bin/hf auth login
```

Then prepare only BBF English MCQ data:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli prepare \
  --source bbf \
  --bbf-language English \
  --bbf-split test \
  --bbf-question-type MCQ \
  --bbf-use-token \
  --output data/processed/english_questions_bbf_mcq.jsonl
```

Prepare a combined dataset (FinMMEval + BBF MCQ):

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli prepare \
  --source both \
  --bbf-language English \
  --bbf-split test \
  --bbf-question-type MCQ \
  --bbf-use-token \
  --output data/processed/english_questions_combined.jsonl
```

Observed BBF English schema (test split):
- columns: `id`, `question`, `correct_answer`, `option_a`, `option_b`, `option_c`, `option_d`, `language`, `question_type`, `question_level`, `topic`, `subject_domain`
- total English rows: `13,451`
- filtered MCQ rows: `12,440`

## Main outputs

- `data/processed/english_questions.jsonl` - normalized English dataset
- `models/option_pair_classifier.joblib` - trained model
- `results/dev_metrics.json` - local dev scores for both approaches
- `results/submission.csv` - prediction file (`id,answer`)

## Notes

- The code uses only English samples.
- For BBF, the loader keeps only rows with `question_type` matching `MCQ`/`Multiple Choice`.
- The loader is robust to schema inconsistencies in the public dataset by reading parquet directly when needed.
- `results/submission.csv` is generated in a standard simple format. If the competition portal requires a different column name, only the header needs adaptation.
