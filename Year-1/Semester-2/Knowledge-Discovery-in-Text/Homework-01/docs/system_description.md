# System Description (Homework 01)

## Task

I participated in **FinMMEval Task 1 (Financial Exam Q&A)** and focused on the **English** part of the data only.  
The goal is to predict the correct option letter for each finance multiple-choice question.

## Data processing

I used the dataset `Tomas08119993/finmmeval-cfa-cpa` from Hugging Face and kept only English samples.  
The public dataset has a schema mismatch between Chinese and English parquet files for the `gold` field, so I implemented a robust loader that reads the English parquet file directly and normalizes all fields.

I also added support for `bharatgenai/BhashaBench-Finance` (BBF), English split, with strict filtering to `MCQ` `question_type` only.
In the current BBF release, English has 13,451 rows and 12,440 rows are MCQ.

For each question, I extracted:
- question text
- option labels (`a`, `b`, `c`, ...)
- option texts parsed from the `query` field
- gold answer letters (for local validation)

The normalized data is stored as JSONL to make the pipeline reproducible.

## Approaches

I tested two approaches:

1. **Lexical overlap baseline**  
   Unsupervised method that selects the option with the highest token overlap with the question.

2. **Supervised option-pair classifier**  
   I converted each question into several `(question, option)` pairs.  
   Each pair is labeled as correct/incorrect, then a TF-IDF + Logistic Regression classifier is trained to estimate correctness probability.  
   At inference time, the option with the highest probability is selected.

## Training and evaluation

Because no separate labeled test split is provided in this dataset release, I used a train/dev split on English samples (`dev_size=0.2`, `seed=42`) for local comparison.

Metrics:
- exact match accuracy
- top-1 accuracy

The supervised approach outperformed the lexical baseline on local validation and was used for final prediction generation.

## Reproducibility

Commands:

```bash
PYTHONPATH=src ./.venv/bin/python -m finmmeval_hw.cli run-all
```

Produced artifacts:
- processed dataset: `data/processed/english_questions.jsonl`
- trained model: `models/option_pair_classifier.joblib`
- local metrics: `results/dev_metrics.json`
- submission file: `results/submission.csv`
