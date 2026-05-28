# FinMMEval Homework 01 Solution

This project implements an English-only automatic question answering pipeline for **FinMMEval Task 1 - Financial Exam Q&A**. The task is multiple-choice financial exam QA: for each question, the system predicts the correct answer option.

The repository contains:

- dataset loaders and normalization code;
- classical baseline and supervised models;
- a structured benchmark runner for several model families;
- a RunPod-oriented Llama 3 8B QLoRA fine-tuning workflow;
- CSV prediction generation;
- a Bulgarian final system description.

## Current Best Result

The strongest experiment so far is:

| Model | Dataset | Dev questions | Exact match / Top-1 |
| --- | --- | ---: | ---: |
| Llama 3 8B Instruct + QLoRA | BBF English MCQ | 2488 | 0.4964 |

Classical benchmark results on the same BBF English MCQ split:

| Model | Dev questions | Exact match / Top-1 |
| --- | ---: | ---: |
| SVM pair | 2488 | 0.3131 |
| TF-IDF + Logistic Regression pair | 2488 | 0.3127 |
| Word2Vec + MLP pair | 2488 | 0.3031 |
| 4D multiclass SVM | 2488 | 0.2850 |
| Transformer cross-encoder | 2488 | 0.2834 |
| MLP pair | 2488 | 0.2769 |
| most common letter baseline | 2488 | 0.2641 |
| lexical overlap | 2488 | 0.2408 |

Results are stored in:

- `results/runpod/extended_benchmarks_bbf_mcq.json`
- `results/runpod/llama3_8b_qlora_bbf_mcq_metrics.json`

The final written description is:

- `FINAL_SYSTEM_DESCRIPTION.md`

## Data

The code supports two sources:

1. **FinMMEval**
   - Hugging Face repo: `Tomas08119993/finmmeval-cfa-cpa`
   - English-only subset
   - robust loader for schema inconsistencies in the public parquet files

2. **BhashaBench-Finance**
   - Hugging Face repo: `bharatgenai/BhashaBench-Finance`
   - gated dataset, requires Hugging Face authentication
   - English split
   - filtered to `question_type=MCQ`

Observed BBF English test split:

- total English rows: `13,451`
- filtered MCQ rows: `12,440`
- local split used in experiments: `9,952` train / `2,488` dev
- split settings: `seed=42`, `dev_size=0.2`, `sample_ratio=1.0`

The normalized project format is JSONL. Each row contains the question text, options, answer labels, gold letters when available, and source metadata.

## Implemented Approaches

Classical and supervised approaches:

- `most_common_letter_baseline` - predicts the most frequent option letter.
- `lexical_overlap` - chooses the option with the strongest token overlap with the question.
- `tfidf_logreg_pair` - turns each `(question, option)` pair into a binary classification example and trains TF-IDF + Logistic Regression.
- `svm_pair` - pairwise TF-IDF + SVM.
- `mlp_pair` - MLP over reduced TF-IDF features.
- `word2vec_mlp_pair` - Word2Vec representations followed by MLP.
- `multiclass_4d_svm_summary` - multiclass SVM for four-option single-answer questions.
- `transformer_cross_encoder` - Transformer-based option-pair scorer.

LLM approach:

- `meta-llama/Meta-Llama-3-8B-Instruct`
- QLoRA fine-tuning
- 4-bit quantized loading
- LoRA rank `16`, alpha `32`, dropout `0.05`
- greedy generation with `max_new_tokens=8`

## Project Structure

```text
Homework-01/
├── README.md
├── FINAL_SYSTEM_DESCRIPTION.md
├── requirements.txt
├── requirements-runpod-llama.txt
├── configs/
│   ├── default.yaml
│   ├── benchmarks.yaml
│   └── llama_qlora.yaml
├── scripts/
│   ├── run_pipeline.sh
│   └── run_llama_qlora.sh
├── src/finmmeval_hw/
│   ├── cli.py
│   ├── data.py
│   ├── evaluation.py
│   ├── modeling.py
│   ├── llama_qlora.py
│   ├── extended_benchmarks.py
│   └── benchmarks/
│       ├── runner.py
│       ├── experiment_config.py
│       └── models/
├── data/
│   └── processed/
├── models/
└── results/
```

## Environment Setup

Local classical-model setup:

```bash
cd Year-1/Semester-2/Knowledge-Discovery-in-Text/Homework-01
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

RunPod / Llama QLoRA setup:

```bash
pip install -r requirements-runpod-llama.txt
```

For gated BBF access, authenticate with Hugging Face:

```bash
./.venv/bin/hf auth login
```

All module commands use `PYTHONPATH=src`.

## Basic Pipeline

Run the default FinMMEval prepare -> train -> predict pipeline:

```bash
scripts/run_pipeline.sh
```

Equivalent manual commands:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli prepare
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli train
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli evaluate
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli predict
```

Default outputs:

- `data/processed/english_questions.jsonl`
- `models/option_pair_classifier.joblib`
- `results/dev_metrics.json`
- `results/split.json`
- `results/submission.csv`

## Prepare Data

Prepare FinMMEval English data:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli prepare \
  --source finmmeval \
  --output data/processed/english_questions_finmmeval.jsonl
```

Prepare BBF English MCQ data:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli prepare \
  --source bbf \
  --bbf-language English \
  --bbf-split test \
  --bbf-question-type MCQ \
  --bbf-use-token \
  --output data/processed/english_questions_bbf_mcq.jsonl
```

Prepare a combined FinMMEval + BBF dataset:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli prepare \
  --source both \
  --bbf-language English \
  --bbf-split test \
  --bbf-question-type MCQ \
  --bbf-use-token \
  --output data/processed/english_questions_combined.jsonl
```

## Train and Evaluate Classical Models

Train the default TF-IDF + Logistic Regression pair classifier:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli train \
  --input data/processed/english_questions_finmmeval.jsonl \
  --model-type linear \
  --model-out models/option_pair_classifier.joblib \
  --metrics-out results/dev_metrics_finmmeval_linear.json \
  --split-out results/split_finmmeval_linear.json
```

Train a Transformer cross-encoder:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli train \
  --input data/processed/english_questions_finmmeval.jsonl \
  --model-type transformer \
  --transformer-model-name distilbert-base-uncased \
  --model-out models/option_pair_transformer_finmmeval \
  --metrics-out results/dev_metrics_finmmeval_transformer.json \
  --split-out results/split_finmmeval_transformer.json
```

Evaluate an existing model:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli evaluate \
  --input data/processed/english_questions_bbf_mcq.jsonl \
  --model-type linear \
  --model models/option_pair_classifier.joblib \
  --output results/evaluation_bbf_linear.json
```

Generate a CSV prediction file:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli predict \
  --input data/processed/english_questions_finmmeval.jsonl \
  --model-type linear \
  --model models/option_pair_classifier.joblib \
  --output results/submission.csv
```

The CSV format is:

```text
id,answer
...
```

## Structured Benchmarks

Run all configured classical benchmarks:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.benchmarks.runner \
  --config-yaml configs/benchmarks.yaml \
  --input-jsonl data/processed/english_questions_bbf_mcq.jsonl \
  --output-json results/extended_benchmarks_bbf_mcq.json \
  --output-md results/extended_benchmarks_bbf_mcq.md \
  --seed 42 \
  --dev-size 0.2 \
  --sample-ratio 1.0 \
  --transformer-model-dir models/option_pair_transformer_finmmeval
```

Run a smaller benchmark sample:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.benchmarks.runner \
  --config-yaml configs/benchmarks.yaml \
  --input-jsonl data/processed/english_questions_combined.jsonl \
  --output-json results/extended_benchmarks_combined_10pct.json \
  --output-md results/extended_benchmarks_combined_10pct.md \
  --seed 42 \
  --dev-size 0.2 \
  --sample-ratio 0.1 \
  --transformer-model-dir models/option_pair_transformer_finmmeval
```

Benchmark configuration lives in:

- `configs/benchmarks.yaml`
- `src/finmmeval_hw/benchmarks/experiment_config.py`
- `src/finmmeval_hw/benchmarks/models/`

## Llama 3 8B QLoRA

The best result uses Llama 3 8B Instruct fine-tuned with QLoRA on BBF English MCQ.

Config:

- `configs/llama_qlora.yaml`

Run:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.llama_qlora \
  --config configs/llama_qlora.yaml
```

or:

```bash
scripts/run_llama_qlora.sh
```

Configured outputs:

- `models/llama3_8b_qlora_bbf_mcq`
- `results/llama3_8b_qlora_bbf_mcq_metrics.json`
- `results/llama3_8b_qlora_bbf_mcq_predictions.csv`
- `results/llama3_8b_qlora_bbf_mcq_split.json`

The RunPod result copied into this project is:

- `results/runpod/llama3_8b_qlora_bbf_mcq_metrics.json`

## Main Artifacts

Important project files:

- `FINAL_SYSTEM_DESCRIPTION.md` - Bulgarian final system description.
- `configs/default.yaml` - default local pipeline settings.
- `configs/benchmarks.yaml` - benchmark model settings.
- `configs/llama_qlora.yaml` - Llama 3 QLoRA settings.
- `src/finmmeval_hw/cli.py` - main prepare/train/evaluate/predict CLI.
- `src/finmmeval_hw/benchmarks/runner.py` - structured benchmark runner.
- `src/finmmeval_hw/llama_qlora.py` - Llama QLoRA training/evaluation script.

Generated or local artifacts:

- `data/processed/english_questions_finmmeval.jsonl`
- `data/processed/english_questions_bbf_mcq.jsonl`
- `data/processed/english_questions_combined.jsonl`
- `models/option_pair_classifier.joblib`
- `models/option_pair_transformer_finmmeval`
- `models/llama3_8b_qlora_bbf_mcq`
- `results/submission.csv`
- `results/runpod/extended_benchmarks_bbf_mcq.json`
- `results/runpod/llama3_8b_qlora_bbf_mcq_metrics.json`

Some generated data, model directories, and RunPod outputs may be ignored by Git depending on the local `.gitignore`.

