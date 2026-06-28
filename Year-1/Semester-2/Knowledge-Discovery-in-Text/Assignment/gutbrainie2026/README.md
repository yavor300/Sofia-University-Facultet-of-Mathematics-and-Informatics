# GutBrainIE 2026 T611/T621 Information Extraction

Course project for the GutBrainIE 2026 challenge.

Focused subtasks:

- `T611`: Named Entity Recognition
- `T621`: Mention-Level Relation Extraction

The repository contains a complete local pipeline for loading the GutBrainIE data, validating annotations, running exploratory analysis, training/evaluating several NER and RE models, generating end-to-end predictions, and visualizing outputs in a Streamlit demo.

## Project Status

Implemented components:

- data loaders and offset validation for articles, entities, and mention-level relations;
- exploratory data analysis and imbalance reports;
- internal exact-match evaluation for T611 and T621;
- T611 dictionary baseline, GLiNER experiments, and transformer token classifiers;
- T621 relation-prior baseline, transformer pair classifier, BioLinkBERT experiments, and optional LLM-assisted verifiers;
- optional official ATLOP baseline wrapper and notes;
- one-command end-to-end prediction pipeline;
- Streamlit frontend for dataset browsing, entity highlighting, relation inspection, metrics, and error analysis;
- final experiment summary table for report writing.

Main final artifacts:

```text
outputs/experiment_results.csv
outputs/experiment_results_gutbrainie.csv
outputs/reports/
outputs/predictions/
app/streamlit_app.py
notebooks/01_data_exploration.ipynb
```

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
make train-re-pair-classifier
make predict-re-pair-classifier
make predict-re-ollama
make predict-re-gpt
make run-pipeline
make app
make experiment-results
make atlop-notes
make prepare-atlop-data
make run-atlop
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
make train-re-pair-classifier RE_PAIR_EXPERIMENT=gold
make predict-re-pair-classifier RE_PAIR_MODEL_DIR=outputs/models/re_pair_classifier_gold
make atlop-notes
make run-atlop ATLOP_ACTION=compose ATLOP_DRY_RUN=1
make predict-re-gpt RE_ENTITIES=outputs/predictions/dev_t611_pubmedbert_gold_silver_silver_2025.json GPT_MAX_CANDIDATES=50
make run-pipeline \
  PIPELINE_NER_MODEL=outputs/models/pubmedbert_gold_silver_silver_2025 \
  PIPELINE_RE_MODEL=outputs/models/re_pair_classifier_gold_silver_silver_2025
make app
make experiment-results
```

## Final Dev Results

All final experiment rows are collected with:

```bash
make experiment-results
```

This reads `outputs/reports/metrics_dev_*.json` and writes:

```text
outputs/experiment_results.csv
outputs/experiment_results_gutbrainie.csv
```

The table includes canonical experiments, legacy/ambiguous filenames, self-check rows, prediction paths, metric paths, and report filters. Use `include_in_report=true` and `status=canonical` for the clean report comparison.

### Best T611 NER Runs

| Experiment | Model | Precision | Recall | Micro-F1 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| `ner_pubmedbert_gold_silver_silver_2025` | PubMedBERT token classifier | 0.7892 | 0.8302 | 0.8092 | 0.7472 |
| `ner_pubmedbert_gold_silver` | PubMedBERT token classifier | 0.7836 | 0.8243 | 0.8034 | 0.7311 |
| `ner_scibert_gold_silver_silver_2025` | SciBERT token classifier | 0.7599 | 0.8183 | 0.7880 | 0.7198 |
| `ner_scibert_gold_silver` | SciBERT token classifier | 0.7547 | 0.8152 | 0.7838 | 0.7158 |
| `ner_biobert_gold_silver` | BioBERT token classifier | 0.7478 | 0.7906 | 0.7686 | 0.6975 |
| `ner_pubmedbert_gold` | PubMedBERT token classifier | 0.7443 | 0.7898 | 0.7664 | 0.6270 |

Best standalone NER model: `ner_pubmedbert_gold_silver_silver_2025`.

### Best T621 RE Runs

| Experiment | Entity Input | Precision | Recall | Micro-F1 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| `re_pair_classifier_biolinkbert_gold_silver_gold_entities` | gold entities | 0.4457 | 0.6409 | 0.5257 | 0.4978 |
| `re_pair_classifier_biolinkbert_gold_gold_entities` | gold entities | 0.4629 | 0.6032 | 0.5238 | 0.4516 |
| `re_pair_classifier_gold_silver_gold_entities` | gold entities | 0.4396 | 0.6198 | 0.5144 | 0.4462 |
| `re_pair_classifier_gold_gold_entities` | gold entities | 0.4390 | 0.5970 | 0.5060 | 0.4810 |
| `re_pair_classifier_biolinkbert_gold_silver_pubmedbert_gold_silver_entities` | predicted PubMedBERT entities | 0.3101 | 0.5048 | 0.3842 | 0.3117 |
| `re_pair_classifier_biolinkbert_gold_silver_pubmedbert_gold_silver_silver_2025_entities` | predicted PubMedBERT entities | 0.3050 | 0.5075 | 0.3810 | 0.3125 |

Best RE model with gold entities: `re_pair_classifier_biolinkbert_gold_silver_gold_entities`.

Best end-to-end RE model: `re_pair_classifier_biolinkbert_gold_silver_pubmedbert_gold_silver_entities`.

The end-to-end score is lower than gold-entity RE because T621 depends directly on upstream T611 entity mentions. This is expected and useful for the final report: the relation classifier is stronger when entity boundaries and labels are correct, while real pipeline performance is capped by NER errors.

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
  --entities outputs/predictions/dev_t611_pubmedbert_gold_silver_silver_2025.json \
  --train-relations data/gutbrainie2026/Annotations/Train/gold_quality/csv_format/train_gold_mention_level_relations.csv \
  --output outputs/predictions/dev_t621_rule_ner_pubmedbert_gold_silver_silver_2025_entities.json
```

Makefile equivalents:

```bash
make predict-re-rule \
  RE_ENTITIES=data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  RE_RULE_OUTPUT=outputs/predictions/dev_t621_rule_gold_entities.json

make predict-re-rule \
  RE_ENTITIES=outputs/predictions/dev_t611_pubmedbert_gold_silver_silver_2025.json \
  RE_RULE_OUTPUT=outputs/predictions/dev_t621_rule_ner_pubmedbert_gold_silver_silver_2025_entities.json

make evaluate EVAL_TASK=re \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  PREDICTION=outputs/predictions/dev_t621_rule_gold_entities.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_re_rule_gold_entities.json

make evaluate EVAL_TASK=re \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  PREDICTION=outputs/predictions/dev_t621_rule_ner_pubmedbert_gold_silver_silver_2025_entities.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_re_rule_ner_pubmedbert_gold_silver_silver_2025_entities.json
```

Use `RE_RULE_THRESHOLD` and `RE_RULE_MAX_DISTANCE` to make the baseline stricter:

```bash
make predict-re-rule RE_RULE_THRESHOLD=0.9 RE_RULE_MAX_DISTANCE=0
```

## Mention-Level Relation Pair Classifier

The trainable RE model creates one sequence-classification example per candidate mention pair. It inserts typed subject/object markers into the title, abstract, or title-plus-abstract context, samples negative `no_relation` examples, and fine-tunes the encoder configured in `configs/re_transformer.yaml`.

Train on gold only:

```bash
python -m gutbrainie.cli train-re-pair-classifier \
  --config configs/re_transformer.yaml \
  --data-root data/gutbrainie2026 \
  --experiment gold \
  --output-dir outputs/models/re_pair_classifier_gold
```

Supported experiments are `gold`, `gold_silver`, and `gold_silver_silver_2025`:

```bash
make train-re-pair-classifier RE_PAIR_EXPERIMENT=gold
make train-re-pair-classifier RE_PAIR_EXPERIMENT=gold_silver
make train-re-pair-classifier RE_PAIR_EXPERIMENT=gold_silver_silver_2025
```

Use `RE_PAIR_MODEL` to try another encoder, for example BioLinkBERT:

```bash
make train-re-pair-classifier \
  RE_PAIR_EXPERIMENT=gold \
  RE_PAIR_MODEL=michiyasunaga/BioLinkBERT-base \
  RE_PAIR_MODEL_DIR=outputs/models/re_pair_classifier_biolinkbert_gold
```

Gold-entity mode evaluates the relation classifier independently from NER:

```bash
make predict-re-pair-classifier \
  RE_PAIR_MODEL_DIR=outputs/models/re_pair_classifier_biolinkbert_gold_silver \
  ARTICLES=data/gutbrainie2026/Articles/csv_format/articles_dev.csv \
  RE_ENTITIES=data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  RE_PAIR_OUTPUT=outputs/predictions/dev_t621_pair_classifier_biolinkbert_gold_silver_gold_entities.json

make evaluate EVAL_TASK=re \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  PREDICTION=outputs/predictions/dev_t621_pair_classifier_biolinkbert_gold_silver_gold_entities.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_re_pair_classifier_biolinkbert_gold_silver_gold_entities.json
```

Predicted-entity mode evaluates the full NER-to-RE pipeline:

```bash
make predict-re-pair-classifier \
  RE_PAIR_MODEL_DIR=outputs/models/re_pair_classifier_biolinkbert_gold_silver \
  ARTICLES=data/gutbrainie2026/Articles/csv_format/articles_dev.csv \
  RE_ENTITIES=outputs/predictions/dev_t611_pubmedbert_gold_silver.json \
  RE_PAIR_OUTPUT=outputs/predictions/dev_t621_pair_classifier_biolinkbert_gold_silver_pubmedbert_gold_silver_entities.json

make evaluate EVAL_TASK=re \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  PREDICTION=outputs/predictions/dev_t621_pair_classifier_biolinkbert_gold_silver_pubmedbert_gold_silver_entities.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_re_pair_classifier_biolinkbert_gold_silver_pubmedbert_gold_silver_entities.json
```

## End-to-End Prediction Pipeline

The pipeline command runs NER, feeds the predicted T611 entities into RE candidate generation, runs the RE model, writes T611/T621 JSON files, logs the full run configuration, and evaluates automatically when `--split dev` has gold annotations.

The final report path uses the best standalone T611 model, `pubmedbert_gold_silver_silver_2025`, and the best end-to-end T621 pair classifier, `re_pair_classifier_biolinkbert_gold_silver`.

```bash
make run-pipeline \
  PIPELINE_SPLIT=dev \
  PIPELINE_NER_MODEL=outputs/models/pubmedbert_gold_silver_silver_2025 \
  PIPELINE_RE_MODEL=outputs/models/re_pair_classifier_biolinkbert_gold_silver \
  PIPELINE_OUTPUT_DIR=outputs/predictions/pipeline_dev \
  PIPELINE_METRICS_OUTPUT=outputs/reports/pipeline_dev_metrics.json
```

Equivalent CLI:

```bash
python -m gutbrainie.cli run-pipeline \
  --data-root data/gutbrainie2026 \
  --split dev \
  --ner-backend token-classifier \
  --re-backend pair-classifier \
  --ner-model outputs/models/pubmedbert_gold_silver_silver_2025 \
  --re-model outputs/models/re_pair_classifier_biolinkbert_gold_silver \
  --output-dir outputs/predictions/pipeline_dev \
  --metrics-output outputs/reports/pipeline_dev_metrics.json
```

Outputs:

```text
outputs/predictions/pipeline_dev/dev_t611_entities.json
outputs/predictions/pipeline_dev/dev_t621_mention_relations.json
outputs/predictions/pipeline_dev/pipeline_config.json
outputs/reports/pipeline_dev_metrics.json
```

For test articles, switch the split. The command writes predictions and config, but skips evaluation because test gold labels are not available locally:

```bash
make run-pipeline \
  PIPELINE_SPLIT=test \
  PIPELINE_NER_MODEL=outputs/models/pubmedbert_gold_silver_silver_2025 \
  PIPELINE_RE_MODEL=outputs/models/re_pair_classifier_biolinkbert_gold_silver \
  PIPELINE_OUTPUT_DIR=outputs/predictions/pipeline_test \
  PIPELINE_METRICS_OUTPUT=outputs/reports/pipeline_test_metrics.json
```

Useful controls:

- `PIPELINE_NER_BACKEND`: `token-classifier`, `gliner`, or `dictionary`.
- `PIPELINE_RE_BACKEND`: `pair-classifier` or `rule`.
- `PIPELINE_USE_CPU=1`: force CPU inference.
- `PIPELINE_RE_THRESHOLD`: pair-classifier confidence threshold.

## Streamlit Visualization Demo

The frontend demo lives in `app/` and is intended for course presentation, error analysis, and model interpretability.

Install dependencies, then start the app:

```bash
pip install -r requirements.txt
make app
```

Equivalent command:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

The app includes:

- project overview;
- dataset explorer with label/predicate distributions;
- T611 entity highlighting with true-positive, false-positive, and false-negative comparison on dev;
- T621 relation cards and tables;
- metrics dashboard from `outputs/reports/*.json`;
- error-analysis views;
- custom-text dictionary NER demo.

## Optional Ollama Relation Verification

LLM experiments are optional and are used here for mention-level relation verification, not as the main project solution. The default local Ollama model tag is `llama3.1:8b`.

Pull and serve the model with Ollama:

```bash
ollama pull llama3.1:8b
ollama serve
```

Run a bounded dev experiment over the first candidate pairs:

```bash
make predict-re-ollama \
  RE_ENTITIES=outputs/predictions/dev_t611_pubmedbert_gold_silver_silver_2025.json \
  OLLAMA_MODEL=llama3.1:8b \
  OLLAMA_MAX_CANDIDATES=200 \
  OLLAMA_OUTPUT=outputs/predictions/dev_t621_ollama_llama31.json \
  OLLAMA_DECISIONS_OUTPUT=outputs/reports/dev_t621_ollama_llama31_decisions.jsonl
```

Evaluate it like any other T621 prediction:

```bash
make evaluate EVAL_TASK=re \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  PREDICTION=outputs/predictions/dev_t621_ollama_llama31.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_re_ollama_llama31.json
```

Useful controls:

- `OLLAMA_MAX_CANDIDATES`: caps the number of LLM calls.
- `OLLAMA_MAX_DISTANCE`: defaults to same-sentence candidates only.
- `OLLAMA_THRESHOLD`: minimum returned confidence for keeping a relation.
- `OLLAMA_DECISIONS_OUTPUT`: JSONL audit log of prompts/results.

## Optional GPT Relation Verification

GPT experiments use the same candidate generation, schema filtering, prompt format, confidence threshold, T621 export, and JSONL audit log as the Ollama verifier. This makes the local Llama and OpenAI runs directly comparable.

Install the optional OpenAI dependency:

```bash
make install-gpt
```

Create `.env` from `.env.example` and set at least:

```text
OPENAI_API_KEY=
OPENAI_JUDGE_MODEL=gpt-4o-mini
```

Run a bounded dev experiment over predicted PubMedBERT entities:

```bash
make predict-re-gpt \
  RE_ENTITIES=outputs/predictions/dev_t611_pubmedbert_gold_silver_silver_2025.json \
  GPT_MAX_CANDIDATES=50 \
  GPT_OUTPUT=outputs/predictions/dev_t621_gpt_pubmedbert_entities.json \
  GPT_DECISIONS_OUTPUT=outputs/reports/dev_t621_gpt_pubmedbert_entities_decisions.jsonl
```

Evaluate it like any other T621 prediction:

```bash
make evaluate EVAL_TASK=re \
  GOLD=data/gutbrainie2026/Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  PREDICTION=outputs/predictions/dev_t621_gpt_pubmedbert_entities.json \
  METRICS_OUTPUT=outputs/reports/metrics_dev_re_gpt_pubmedbert_entities.json
```

Useful controls:

- `GPT_MODEL`: overrides `OPENAI_JUDGE_MODEL` from `.env`.
- `GPT_MAX_CANDIDATES`: caps paid API calls.
- `GPT_MAX_DISTANCE`: defaults to same-sentence candidates only.
- `GPT_THRESHOLD`: minimum returned confidence for keeping a relation.
- `GPT_DECISIONS_OUTPUT`: JSONL audit log of prompts/results.

## Optional ATLOP Reproduction

The official baseline uses ATLOP for relation extraction. In this project it is treated as a bonus comparison; the main implemented RE model is the PubMedBERT pair classifier above.

The official repository should live at:

```text
external/GutBrainIE_2026_Baseline
```

Refresh the local reproduction checklist:

```bash
make atlop-notes
```

This writes:

```text
outputs/reports/atlop_notes.md
```

The wrapper does not edit official scripts. It runs the official files under `external/GutBrainIE_2026_Baseline/Train/RE` only when explicitly called.

Install the optional ATLOP dependencies before running official training:

```bash
make install-atlop
```

Dry-run the official commands first:

```bash
make run-atlop ATLOP_ACTION=compose ATLOP_DRY_RUN=1
make run-atlop ATLOP_ACTION=finetune ATLOP_DRY_RUN=1
make run-atlop ATLOP_ACTION=predict ATLOP_DRY_RUN=1
```

Create the ATLOP JSON files in `external/GutBrainIE_2026_Baseline/Train/RE/data`, compose the official training files, then fine-tune:

```bash
make prepare-atlop-data
make run-atlop ATLOP_ACTION=compose
make run-atlop ATLOP_ACTION=finetune
```

For prediction, also convert the dev NER predictions into ATLOP input format:

```bash
make prepare-atlop-data \
  ATLOP_PREDICTED_ENTITIES=outputs/predictions/dev_t611_pubmedbert_gold_silver_silver_2025.json

make run-atlop ATLOP_ACTION=predict \
  ATLOP_OUTPUT=outputs/predictions/atlop_predicted_relations_raw.json
```

Notes:

- `BaselineModels/RE.zip` may be a Git LFS pointer until `git lfs pull` is run inside the official repo.
- The copied ATLOP prediction is the official raw relation output; merge or convert it before internal T621 evaluation if needed.
- ATLOP setup is heavier than the pair classifier and expects CUDA in the official script.

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
