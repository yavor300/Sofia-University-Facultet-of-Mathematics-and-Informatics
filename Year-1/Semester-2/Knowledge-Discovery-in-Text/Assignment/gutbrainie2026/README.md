# GutBrainIE 2026 T611/T621

Course project for the GutBrainIE 2026 challenge.

Focused subtasks:

- `T611`: Named Entity Recognition
- `T621`: Mention-Level Relation Extraction

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The official data should be placed at:

```text
data/gutbrainie2026/
```

The data directory is ignored by Git.

## Smoke Checks

```bash
pytest
python -m gutbrainie.cli --help
```

## Makefile Commands

Most project commands are wrapped by `make`:

```bash
make help
make cli-help
make install-gliner
make test
make prepare-data QUALITY=gold
make validate-all
make eda
make eda-all
make run-ner-baseline
make predict-ner-dictionary
make prepare-gliner-data
make prepare-gliner-all
make train-gliner
make train-gliner-cpu
make predict-gliner
make train-token-classifier
make predict-token-classifier
make run-ner-transformer
make run-re-baseline
make run-re-transformer
make evaluate
make evaluate-official
make export-t611
make export-t621
```

The wrappers work from the source tree by setting `PYTHONPATH=src`. Common overrides:

```bash
make prepare-data QUALITY=dev
make prepare-data DATA_ROOT=data/gutbrainie2026 OUTPUTS_ROOT=outputs
make eda EDA_QUALITIES="gold dev"
make eda-all
make predict-ner-dictionary \
  TRAIN_ENTITIES=data/gutbrainie2026/Annotations/Train/gold_quality/csv_format/train_gold_entities.csv \
  ARTICLES=data/gutbrainie2026/Articles/csv_format/articles_dev.csv \
  NER_DICTIONARY_OUTPUT=outputs/predictions/dev_t611_dictionary.json
make prepare-gliner-data GLINER_EXPERIMENT=gold_silver
make prepare-gliner-all
make predict-gliner GLINER_MODEL=urchade/gliner_medium-v2.1 \
  ARTICLES=data/gutbrainie2026/Articles/csv_format/articles_dev.csv
make train-gliner-cpu GLINER_EXPERIMENT=gold GLINER_MODEL=urchade/gliner_medium-v2.1
make train-token-classifier TOKEN_EXPERIMENT=gold
make train-token-classifier TOKEN_EXPERIMENT=gold NER_TRANSFORMER_CONFIG=configs/ner_transformer_cpu.yaml
make train-pubmedbert-token-classifier TOKEN_EXPERIMENT=gold
make train-biomedbert-token-classifier TOKEN_EXPERIMENT=gold
make train-scibert-token-classifier TOKEN_EXPERIMENT=gold
make train-biobert-token-classifier TOKEN_EXPERIMENT=gold
make predict-token-classifier TOKEN_MODEL_DIR=outputs/models/token_classifier_gold
make predict-re-rule
```

## NER Dictionary Baseline

```bash
python -m gutbrainie.cli predict-ner-dictionary \
  --train-entities data/gutbrainie2026/Annotations/Train/gold_quality/csv_format/train_gold_entities.csv \
  --articles data/gutbrainie2026/Articles/csv_format/articles_dev.csv \
  --output outputs/predictions/dev_t611_dictionary.json
```

Makefile shortcut:

```bash
make predict-ner-dictionary
make run-ner-baseline
```

## GLiNER NER

Install the optional GLiNER dependency:

```bash
make install-gliner
```

Prepare GLiNER train/validation data:

```bash
python -m gutbrainie.cli prepare-gliner-data \
  --data-root data/gutbrainie2026 \
  --experiment gold \
  --output-dir outputs/gliner
```

Supported experiments are `gold`, `gold_silver`, and `gold_silver_silver_2025`.

Fine-tune GLiNER, if the installed `gliner` package exposes a compatible training API:

```bash
python -m gutbrainie.cli train-gliner \
  --config configs/ner_gliner.yaml \
  --model urchade/gliner_medium-v2.1 \
  --train-data outputs/gliner/gliner_gold_train.jsonl \
  --validation-data outputs/gliner/gliner_gold_validation.jsonl \
  --output-dir outputs/models/gliner_gold
```

Predict with a fine-tuned model directory or a Hugging Face model name:

```bash
python -m gutbrainie.cli predict-gliner \
  --config configs/ner_gliner.yaml \
  --model outputs/models/gliner_gold \
  --articles data/gutbrainie2026/Articles/csv_format/articles_dev.csv \
  --output outputs/predictions/dev_t611_gliner_gold.json
```

Makefile equivalents:

```bash
make prepare-gliner-data GLINER_EXPERIMENT=gold
make prepare-gliner-all
make train-gliner GLINER_EXPERIMENT=gold GLINER_MODEL=urchade/gliner_medium-v2.1
make train-gliner-cpu GLINER_EXPERIMENT=gold GLINER_MODEL=urchade/gliner_medium-v2.1
make predict-gliner GLINER_MODEL=outputs/models/gliner_gold
make run-ner-transformer GLINER_MODEL=outputs/models/gliner_gold
```

If training exits with code `137`, the OS killed the process due to memory pressure. Use the CPU-safe config:

```bash
make train-gliner-cpu GLINER_EXPERIMENT=gold GLINER_MODEL=urchade/gliner_medium-v2.1
```

The CPU-safe config uses `configs/ner_gliner_cpu.yaml`: batch size 1, shorter max length, no pinned memory, and `max_steps: 100` for a bounded smoke-sized run. Increase `max_steps` or remove it after the command is stable on your machine.

## PubMedBERT/BioBERT/SciBERT Token Classifier

The classical token-classification NER path converts title and abstract annotations into BIO tags with Hugging Face tokenizer offset mappings.

Train the default BiomedBERT/PubMedBERT-style model on gold:

```bash
python -m gutbrainie.cli train-token-classifier \
  --config configs/ner_transformer.yaml \
  --data-root data/gutbrainie2026 \
  --experiment gold \
  --output-dir outputs/models/token_classifier_gold
```

The same Microsoft BiomedBERT model is also available as an explicit named experiment:

```bash
python -m gutbrainie.cli train-token-classifier \
  --config configs/ner_pubmedbert.yaml \
  --data-root data/gutbrainie2026 \
  --experiment gold \
  --output-dir outputs/models/pubmedbert_gold

python -m gutbrainie.cli train-token-classifier \
  --config configs/ner_biomedbert.yaml \
  --data-root data/gutbrainie2026 \
  --experiment gold \
  --output-dir outputs/models/biomedbert_gold
```

Supported experiments match the GLiNER setup: `gold`, `gold_silver`, and `gold_silver_silver_2025`.

For a bounded CPU smoke run:

```bash
python -m gutbrainie.cli train-token-classifier \
  --config configs/ner_transformer_cpu.yaml \
  --data-root data/gutbrainie2026 \
  --experiment gold \
  --output-dir outputs/models/token_classifier_gold_smoke
```

To try SciBERT or BioBERT, use the dedicated configs:

```bash
python -m gutbrainie.cli train-token-classifier \
  --config configs/ner_scibert.yaml \
  --data-root data/gutbrainie2026 \
  --experiment gold \
  --output-dir outputs/models/scibert_gold

python -m gutbrainie.cli train-token-classifier \
  --config configs/ner_biobert.yaml \
  --data-root data/gutbrainie2026 \
  --experiment gold \
  --output-dir outputs/models/biobert_gold
```

Predict T611 JSON with a trained model:

```bash
python -m gutbrainie.cli predict-token-classifier \
  --model outputs/models/token_classifier_gold \
  --articles data/gutbrainie2026/Articles/csv_format/articles_dev.csv \
  --output outputs/predictions/dev_t611_token_classifier_gold.json
```

Makefile equivalents:

```bash
make train-token-classifier TOKEN_EXPERIMENT=gold
make train-token-classifier TOKEN_EXPERIMENT=gold_silver
make train-token-classifier TOKEN_EXPERIMENT=gold NER_TRANSFORMER_CONFIG=configs/ner_transformer_cpu.yaml
make train-pubmedbert-token-classifier TOKEN_EXPERIMENT=gold
make train-pubmedbert-token-classifier TOKEN_EXPERIMENT=gold_silver
make train-biomedbert-token-classifier TOKEN_EXPERIMENT=gold
make train-biomedbert-token-classifier TOKEN_EXPERIMENT=gold_silver
make train-scibert-token-classifier TOKEN_EXPERIMENT=gold
make train-scibert-token-classifier TOKEN_EXPERIMENT=gold_silver
make train-biobert-token-classifier TOKEN_EXPERIMENT=gold
make train-biobert-token-classifier TOKEN_EXPERIMENT=gold_silver
make predict-token-classifier TOKEN_MODEL_DIR=outputs/models/token_classifier_gold
```

For the full comparison grid:

```bash
make train-pubmedbert-token-classifier TOKEN_EXPERIMENT=gold
make train-pubmedbert-token-classifier TOKEN_EXPERIMENT=gold_silver
make train-pubmedbert-token-classifier TOKEN_EXPERIMENT=gold_silver_silver_2025

make train-biomedbert-token-classifier TOKEN_EXPERIMENT=gold
make train-biomedbert-token-classifier TOKEN_EXPERIMENT=gold_silver
make train-biomedbert-token-classifier TOKEN_EXPERIMENT=gold_silver_silver_2025

make train-scibert-token-classifier TOKEN_EXPERIMENT=gold
make train-scibert-token-classifier TOKEN_EXPERIMENT=gold_silver
make train-scibert-token-classifier TOKEN_EXPERIMENT=gold_silver_silver_2025

make train-biobert-token-classifier TOKEN_EXPERIMENT=gold
make train-biobert-token-classifier TOKEN_EXPERIMENT=gold_silver
make train-biobert-token-classifier TOKEN_EXPERIMENT=gold_silver_silver_2025
```

Predict and evaluate one of those runs by pointing `TOKEN_MODEL_DIR` at the matching model directory:

```bash
make predict-token-classifier \
  TOKEN_MODEL_DIR=outputs/models/scibert_gold_silver_silver_2025 \
  TOKEN_OUTPUT=outputs/predictions/dev_t611_scibert_gold_silver_silver_2025.json

make evaluate EVAL_TASK=ner \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  PREDICTION=outputs/predictions/dev_t611_scibert_gold_silver_silver_2025.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_ner_scibert_gold_silver_silver_2025.json
```

PubMedBERT prediction and evaluation example:

```bash
make predict-token-classifier \
  TOKEN_MODEL_DIR=outputs/models/pubmedbert_gold_silver_silver_2025 \
  TOKEN_OUTPUT=outputs/predictions/dev_t611_pubmedbert_gold_silver_silver_2025.json

make evaluate EVAL_TASK=ner \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  PREDICTION=outputs/predictions/dev_t611_pubmedbert_gold_silver_silver_2025.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_ner_pubmedbert_gold_silver_silver_2025.json
```

BiomedBERT prediction and evaluation example:

```bash
make predict-token-classifier \
  TOKEN_MODEL_DIR=outputs/models/biomedbert_gold_silver_silver_2025 \
  TOKEN_OUTPUT=outputs/predictions/dev_t611_biomedbert_gold_silver_silver_2025.json

make evaluate EVAL_TASK=ner \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  PREDICTION=outputs/predictions/dev_t611_biomedbert_gold_silver_silver_2025.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_ner_biomedbert_gold_silver_silver_2025.json
```

## Mention-Level Relation Rule Baseline

The rule baseline generates ordered subject-object mention pairs, filters them by the T621 relation schema, and predicts the most frequent training predicate for each `(subject_label, object_label)` pair when its prior is above a threshold.

Gold-entity mode evaluates RE independently of NER:

```bash
python -m gutbrainie.cli predict-re-rule \
  --articles data/gutbrainie2026/Articles/csv_format/articles_dev.csv \
  --entities data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  --train-relations data/gutbrainie2026/Annotations/Train/gold_quality/csv_format/train_gold_mention_level_relations.csv \
  --output outputs/predictions/dev_t621_rule_gold_entities.json

python -m gutbrainie.cli evaluate \
  --task re \
  --gold data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  --prediction outputs/predictions/dev_t621_rule_gold_entities.json \
  --output outputs/reports/metrics_dev_re_rule_gold_entities.json
```

Predicted-entity mode evaluates the full NER-to-RE pipeline:

```bash
python -m gutbrainie.cli predict-re-rule \
  --articles data/gutbrainie2026/Articles/csv_format/articles_dev.csv \
  --entities outputs/predictions/dev_t611_token_classifier_gold_silver_silver_2025.json \
  --train-relations data/gutbrainie2026/Annotations/Train/gold_quality/csv_format/train_gold_mention_level_relations.csv \
  --output outputs/predictions/dev_t621_rule_predicted_entities.json
```

Makefile equivalents:

```bash
make predict-re-rule \
  RE_ENTITIES=data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  RE_RULE_OUTPUT=outputs/predictions/dev_t621_rule_gold_entities.json

make predict-re-rule \
  RE_ENTITIES=outputs/predictions/dev_t611_token_classifier_gold_silver_silver_2025.json \
  RE_RULE_OUTPUT=outputs/predictions/dev_t621_rule_predicted_entities.json

make evaluate EVAL_TASK=re \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  PREDICTION=outputs/predictions/dev_t621_rule_gold_entities.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_re_rule_gold_entities.json
```

Use `RE_RULE_THRESHOLD` and `RE_RULE_MAX_DISTANCE` to make the baseline stricter:

```bash
make predict-re-rule RE_RULE_THRESHOLD=0.9 RE_RULE_MAX_DISTANCE=0
```

## Evaluation

Internal exact-match NER evaluation:

```bash
python -m gutbrainie.cli evaluate \
  --task ner \
  --gold data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  --prediction outputs/predictions/dev_t611_predictions.csv \
  --output outputs/reports/metrics_ner.json
```

Internal mention-level relation evaluation:

```bash
python -m gutbrainie.cli evaluate \
  --task re \
  --gold data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  --prediction outputs/predictions/dev_t621_predictions.csv \
  --output outputs/reports/metrics_re.json
```

Makefile equivalents:

```bash
make evaluate EVAL_TASK=ner \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  PREDICTION=outputs/predictions/dev_t611_predictions.csv \
  METRICS_OUTPUT=outputs/reports/metrics_ner.json

make evaluate EVAL_TASK=re \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  PREDICTION=outputs/predictions/dev_t621_predictions.csv \
  METRICS_OUTPUT=outputs/reports/metrics_re.json
```

Official evaluator wrapper, after cloning the external baseline repository:

```bash
git clone https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline external/GutBrainIE_2026_Baseline

python -m gutbrainie.cli evaluate-official \
  --official-repo external/GutBrainIE_2026_Baseline \
  --prediction outputs/predictions/dev_t611_gliner.json
```

If the official repository entrypoint differs, pass it explicitly:

```bash
python -m gutbrainie.cli evaluate-official \
  --official-repo external/GutBrainIE_2026_Baseline \
  --entrypoint evaluation/evaluate.py \
  --prediction outputs/predictions/dev_t611_gliner.json \
  -- --any-official-arg value
```
