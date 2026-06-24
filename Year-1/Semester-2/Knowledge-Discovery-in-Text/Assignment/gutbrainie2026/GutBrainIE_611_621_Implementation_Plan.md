# GutBrainIE 2026 Course Project Implementation Plan

**Project focus:** Subtask **6.1.1 Named Entity Recognition (NER)** and Subtask **6.2.1 Mention-Level Relation Extraction (M-RE)** for the GutBrainIE @ CLEF 2026 challenge.

This document is intended to be shared with a Codex agent as an implementation roadmap. It describes the project scope, repository structure, implementation phases, exact development steps, model options, evaluation flow, and deliverables.

---

## 1. Project Goal

Build a reproducible biomedical information extraction pipeline over PubMed titles and abstracts related to the gut-brain axis.

The system must support two tasks:

### Task T611 — Subtask 6.1.1: Named Entity Recognition

Given an article title and abstract, detect biomedical entity mentions and classify them into one of the challenge entity labels.

Expected output per PMID:

```json
{
  "34870091": {
    "entities": [
      {
        "start_idx": 75,
        "end_idx": 82,
        "location": "title",
        "text_span": "patients",
        "label": "human"
      }
    ]
  }
}
```

### Task T621 — Subtask 6.2.1: Mention-Level Relation Extraction

Given an article and entity mentions, detect relations between textual entity mentions.

Expected output per PMID:

```json
{
  "34870091": {
    "mention_level_relations": [
      {
        "subject_text_span": "intestinal microbiome",
        "subject_label": "microbiome",
        "predicate": "located in",
        "object_text_span": "patients",
        "object_label": "human"
      }
    ]
  }
}
```

The final course project should include:

- a working data loading pipeline;
- a baseline NER model;
- a stronger NER model;
- a baseline relation extraction model;
- optionally a stronger RE model;
- evaluation on the development set;
- generated prediction files in challenge-compatible JSON format;
- a model report with metrics, experiments, and error analysis;
- a README explaining how to reproduce the results.

---

## 2. Input Data Structure

The local dataset is expected to have the following high-level structure:

```text
data/gutbrainie2026/
├── Annotations/
│   ├── Dev/
│   │   ├── csv_format/
│   │   ├── json_format/dev.json
│   │   └── txt_format/
│   ├── Train/
│   │   ├── bronze_quality/
│   │   ├── gold_quality/
│   │   └── silver_quality/
│   ├── GutBrainIE_2026_Annotation_Guidelines.pdf
│   └── uris.csv
├── Articles/
│   ├── csv_format/
│   ├── json_format/
│   └── txt_format/
└── Test_Data/
    ├── articles_test.csv
    ├── articles_test.json
    └── articles_test.txt
```

Important files:

```text
Articles/csv_format/articles_train_gold.csv
Articles/csv_format/articles_train_silver.csv
Articles/csv_format/articles_train_silver_2025.csv
Articles/csv_format/articles_train_bronze.csv
Articles/csv_format/articles_dev.csv
Articles/csv_format/articles_test.csv

Annotations/Train/gold_quality/csv_format/train_gold_entities.csv
Annotations/Train/gold_quality/csv_format/train_gold_mention_level_relations.csv
Annotations/Train/gold_quality/csv_format/train_gold_relations.csv

Annotations/Train/silver_quality/csv_format/train_silver_entities.csv
Annotations/Train/silver_quality/csv_format/train_silver_mention_level_relations.csv
Annotations/Train/silver_quality/csv_format/train_silver_relations.csv

Annotations/Train/silver_quality/csv_format/train_silver_2025_entities.csv
Annotations/Train/silver_quality/csv_format/train_silver_2025_mention_level_relations.csv
Annotations/Train/silver_quality/csv_format/train_silver_2025_relations.csv

Annotations/Train/bronze_quality/csv_format/train_bronze_entities.csv
Annotations/Train/bronze_quality/csv_format/train_bronze_mention_level_relations.csv
Annotations/Train/bronze_quality/csv_format/train_bronze_relations.csv

Annotations/Dev/csv_format/dev_entities.csv
Annotations/Dev/csv_format/dev_mention_level_relations.csv
Annotations/Dev/csv_format/dev_relations.csv
```

CSV files use pipe separators:

```text
pmid|title|authors|journal|year|abstract
pmid|annotator|start_idx|end_idx|location|text_span|label
pmid|annotator|subject_text_span|subject_label|predicate|object_text_span|object_label
```

---

## 3. Recommended Scope for the Course Assignment

### Required implementation

1. Data loader for articles, entities, and mention-level relations.
2. Dataset validation, including offset checks.
3. NER baseline.
4. Transformer or GLiNER-based NER model.
5. Mention-level RE candidate generation.
6. RE baseline.
7. Optional transformer-based RE classifier.
8. Evaluation on the development set.
9. JSON prediction export for T611 and T621.
10. Model report with experiment table and error analysis.

### Recommended minimum viable project

The minimum acceptable project should implement:

```text
NER:
  - dictionary/rule baseline
  - fine-tuned or prompted GLiNER model

RE:
  - candidate-pair generation from entity mentions
  - typed rule/prior baseline
  - optional BERT-style pair classifier

Evaluation:
  - micro/macro precision, recall, F1
  - official evaluation script integration where possible
```

### Recommended advanced version

The stronger course project should implement:

```text
NER:
  - GLiNER baseline reproduction
  - PubMedBERT/BioBERT/SciBERT token-classification experiment

RE:
  - type-constrained candidate generation
  - PubMedBERT/BioLinkBERT pair classifier with entity markers
  - optional ATLOP baseline reproduction from official repo

Analysis:
  - comparison of gold-only vs gold+silver training
  - impact of predicted entities vs gold entities for relation extraction
  - error analysis by entity label and relation predicate
```

---

## 4. Suggested Repository Structure

Implement the project with a clean Python package structure:

```text
.
├── README.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── paths.yaml
│   ├── ner_gliner.yaml
│   ├── ner_transformer.yaml
│   ├── re_baseline.yaml
│   └── re_transformer.yaml
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_error_analysis_ner.ipynb
│   └── 03_error_analysis_re.ipynb
├── src/
│   └── gutbrainie/
│       ├── __init__.py
│       ├── constants.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── articles.py
│       │   ├── annotations.py
│       │   ├── dataset.py
│       │   ├── offsets.py
│       │   └── splits.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── ner_metrics.py
│       │   ├── re_metrics.py
│       │   ├── official_eval_wrapper.py
│       │   └── report.py
│       ├── ner/
│       │   ├── __init__.py
│       │   ├── dictionary_baseline.py
│       │   ├── bio_tags.py
│       │   ├── train_token_classifier.py
│       │   ├── predict_token_classifier.py
│       │   ├── gliner_runner.py
│       │   └── postprocess.py
│       ├── re/
│       │   ├── __init__.py
│       │   ├── relation_schema.py
│       │   ├── candidates.py
│       │   ├── rule_baseline.py
│       │   ├── train_pair_classifier.py
│       │   ├── predict_pair_classifier.py
│       │   └── postprocess.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── ollama_relation_verifier.py
│       │   ├── gpt_relation_verifier.py
│       │   └── prompts.py
│       ├── submission/
│       │   ├── __init__.py
│       │   ├── export_t611.py
│       │   ├── export_t621.py
│       │   ├── validate.py
│       │   └── zip_submission.py
│       └── cli.py
├── scripts/
│   ├── prepare_data.sh
│   ├── run_ner_baseline.sh
│   ├── run_ner_transformer.sh
│   ├── run_re_baseline.sh
│   ├── run_re_transformer.sh
│   └── evaluate_all.sh
├── outputs/
│   ├── predictions/
│   ├── models/
│   ├── reports/
│   └── submissions/
└── tests/
    ├── test_data_loading.py
    ├── test_offsets.py
    ├── test_bio_conversion.py
    ├── test_re_candidates.py
    ├── test_submission_format.py
    └── test_metrics.py
```

Do not commit the official data files. Add them to `.gitignore`:

```gitignore
data/gutbrainie2026/
outputs/models/
outputs/predictions/
outputs/submissions/
*.pt
*.bin
*.safetensors
```

---

## 5. Implementation Phases

## Phase 0 — Project Bootstrap

### Goal

Create the repository skeleton and basic reproducibility setup.

### Steps for Codex

1. Create the repository structure shown above.
2. Add `requirements.txt` with dependencies:

```text
pandas
numpy
scikit-learn
tqdm
pyyaml
pytest
seqeval
transformers
datasets
accelerate
torch
sentencepiece
protobuf
rich
matplotlib
jupyter
```

Optional dependencies:

```text
gliner
spacy
ollama
openai
```

3. Add `pyproject.toml` with package configuration for `src/gutbrainie`.
4. Add a basic CLI entry point:

```bash
python -m gutbrainie.cli --help
```

5. Add `configs/paths.yaml`:

```yaml
data_root: data/gutbrainie2026
outputs_root: outputs
train_quality: gold
use_silver: false
use_bronze: false
```

### Acceptance criteria

- `pip install -r requirements.txt` works.
- `pytest` runs successfully with placeholder tests.
- `python -m gutbrainie.cli --help` prints available commands.

---

## Phase 1 — Data Loading and Validation

### Goal

Read articles, entities, and mention-level relations from the provided CSV/JSON files and normalize them into internal Python objects or DataFrames.

### Key implementation decisions

Use CSV as the primary format because it is explicit and easy to inspect. Use JSON loading later as an optional alternative.

Offsets are location-specific. This means:

- if `location == "title"`, `start_idx` and `end_idx` refer to the title string;
- if `location == "abstract"`, offsets refer to the abstract string;
- do not concatenate title and abstract unless you also maintain an offset mapping.

### Steps for Codex

1. Implement `src/gutbrainie/data/articles.py`:

```python
def load_articles_csv(path: str | Path) -> pd.DataFrame:
    ...
```

Expected columns:

```text
pmid, title, authors, journal, year, abstract
```

2. Implement `src/gutbrainie/data/annotations.py`:

```python
def load_entities_csv(path: str | Path) -> pd.DataFrame:
    ...

def load_mention_relations_csv(path: str | Path) -> pd.DataFrame:
    ...

def load_full_relations_csv(path: str | Path) -> pd.DataFrame:
    ...
```

3. Convert all PMIDs to string to avoid mismatch between article and annotation files.

4. Implement `src/gutbrainie/data/offsets.py`:

```python
def validate_entity_offsets(article_row, entity_row) -> bool:
    """Check whether text_span equals text[start_idx:end_idx]."""
```

5. Implement a data preparation command:

```bash
python -m gutbrainie.cli prepare-data \
  --data-root data/gutbrainie2026 \
  --quality gold \
  --output outputs/reports/data_validation_gold.json
```

6. Add validation report fields:

```json
{
  "articles": 639,
  "entities": 20530,
  "relations": 8556,
  "offset_checks_passed": 20300,
  "offset_checks_failed": 230,
  "missing_articles": 0
}
```

7. Implement deduplication logic:

```python
def deduplicate_entities(df: pd.DataFrame) -> pd.DataFrame:
    # unique by pmid, location, start_idx, end_idx, text_span, label
```

For the first implementation, deduplicate exact duplicates only. Do not attempt complex adjudication between annotators.

### Acceptance criteria

- Gold, silver, silver_2025, bronze, and dev article files can be loaded.
- Entity and relation CSV files can be loaded.
- Offset validation report is generated.
- Duplicates are handled consistently.

---

## Phase 2 — Exploratory Data Analysis

### Goal

Generate dataset statistics for the report and help decide model design.

### Steps for Codex

1. Implement `src/gutbrainie/evaluation/report.py` or `scripts/eda.py`.
2. Generate statistics:

- number of documents per split;
- number of entities per label;
- number of relations per predicate;
- number of relations per `(subject_label, predicate, object_label)` triple;
- average title length;
- average abstract length;
- average entities per article;
- average relations per article;
- label imbalance.

3. Save outputs:

```text
outputs/reports/data_stats_gold.csv
outputs/reports/data_stats_dev.csv
outputs/reports/entity_label_distribution.csv
outputs/reports/relation_label_distribution.csv
```

4. Create at least two plots:

```text
outputs/reports/entity_distribution.png
outputs/reports/relation_distribution.png
```

### Acceptance criteria

- The report can include real numbers from the local dataset.
- The project can explain why micro-F1 is important because of class imbalance.

---

## Phase 3 — Evaluation Layer

### Goal

Implement internal evaluation and integrate the official evaluation script where possible.

### Internal NER evaluation

A NER prediction is correct only when all of the following match:

```text
pmid, location, start_idx, end_idx, label
```

The `text_span` should be validated but should not be the only matching key.

Implement:

```python
def evaluate_ner(gold_entities: pd.DataFrame, pred_entities: pd.DataFrame) -> dict:
    ...
```

Return:

```text
micro_precision, micro_recall, micro_f1
macro_precision, macro_recall, macro_f1
per_label_precision, per_label_recall, per_label_f1
```

### Internal mention-level RE evaluation

A mention-level relation prediction is correct when all fields match:

```text
pmid,
subject_text_span,
subject_label,
predicate,
object_text_span,
object_label
```

Because mention-level relations do not include offsets in the official T621 format, use the official field set for evaluation. If there are duplicate text spans within the same article, optionally keep an internal offset-aware evaluation for debugging.

Implement:

```python
def evaluate_mention_relations(gold_relations: pd.DataFrame, pred_relations: pd.DataFrame) -> dict:
    ...
```

For macro labels, use relation classes as triples:

```text
subject_label + predicate + object_label
```

### Official evaluation integration

Clone or reference the official baseline repository:

```bash
git clone https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline external/GutBrainIE_2026_Baseline
```

The official repository contains baseline/evaluation folders and supports reproducing baselines, generating predictions, and validating/evaluating submissions. Use it as an external reference, not as the only implementation.

Implement a wrapper:

```bash
python -m gutbrainie.cli evaluate-official \
  --official-repo external/GutBrainIE_2026_Baseline \
  --prediction outputs/predictions/dev_t611_gliner.json
```

### Acceptance criteria

- Internal metrics work on small synthetic tests.
- Internal metrics work on dev predictions.
- Official evaluation can be run manually or through a documented wrapper.

---

## Phase 4 — NER Baseline

### Goal

Create a transparent, fast baseline for entity extraction.

### Approach A: Dictionary baseline

Build a label-specific dictionary from training annotations.

Example:

```text
"patients" -> human
"intestinal microbiome" -> microbiome
"Alzheimer's disease" -> DDF
```

### Steps for Codex

1. Implement `src/gutbrainie/ner/dictionary_baseline.py`.
2. Build dictionary from train entities:

```python
def build_entity_dictionary(train_entities: pd.DataFrame) -> dict[str, set[str]]:
    ...
```

3. Normalize dictionary keys:

- lowercase;
- strip whitespace;
- optionally remove trailing punctuation;
- keep original label.

4. Predict by exact case-insensitive matching over title and abstract separately.
5. Avoid overlapping predictions by preferring:

1. longer span;
2. higher frequency in training;
3. deterministic label order.

6. Export predictions to T611 JSON:

```bash
python -m gutbrainie.cli predict-ner-dictionary \
  --train-entities Annotations/Train/gold_quality/csv_format/train_gold_entities.csv \
  --articles Articles/csv_format/articles_dev.csv \
  --output outputs/predictions/dev_t611_dictionary.json
```

### Acceptance criteria

- Produces valid `entities` JSON.
- Runs quickly on dev.
- Establishes a lower-bound baseline.

---

## Phase 5 — Stronger NER Models

Implement at least one stronger NER approach. The recommended first choice is GLiNER because the official baseline uses GLiNER/NuNerZero for NER.

## Option 1: GLiNER / NuNerZero

### Why

- Strong zero-shot/few-shot entity extraction behavior.
- Official baseline uses this family of model.
- Easier than writing BIO tokenization from scratch.

### Steps for Codex

1. Implement `src/gutbrainie/ner/gliner_runner.py`.
2. Convert training data to GLiNER format:

```json
{
  "text": "Title or abstract text here",
  "label": [[0, 8, "human"], [20, 35, "microbiome"]]
}
```

3. Treat title and abstract as separate examples to preserve offsets.
4. Fine-tune on:

```text
Experiment 1: train_gold only
Train: Gold 85%
Validation: Gold 15%
Final evaluation: Official Dev

Experiment 2: train_gold + train_silver
Train: Gold 85% + Silver
Validation: Gold 15%
Final evaluation: Official Dev

Experiment 3: train_gold + train_silver + train_silver_2025
Train: Gold 85% + Silver + Silver_2025
Validation: Gold 15%
Final evaluation: Official Dev
```

5. Use dev for validation.
6. Predict separately for title and abstract.
7. Convert GLiNER spans back to challenge format.
8. Save predictions:

```text
outputs/predictions/dev_t611_gliner_gold.json
outputs/predictions/dev_t611_gliner_gold_silver.json
outputs/predictions/test_t611_gliner_best.json
```

### Recommended configuration

```yaml
model_name: urchade/gliner_medium-v2.1
labels:
  - anatomical location
  - animal
  - biomedical technique
  - bacteria
  - chemical
  - dietary supplement
  - DDF
  - drug
  - food
  - gene
  - human
  - microbiome
  - statistical technique
threshold: 0.5
max_len: 384
batch_size: 8
learning_rate: 2e-5
epochs: 5
```

If `NuNerZero` is available from the baseline repo or Hugging Face, add it as an experiment.

## Option 2: PubMedBERT / BioBERT / SciBERT token classification

### Why

This is a classical and easy-to-explain NER architecture for a course report.

### Steps for Codex

1. Implement BIO conversion in `src/gutbrainie/ner/bio_tags.py`.
2. Convert each title and abstract into token-level labels:

```text
B-DDF, I-DDF, B-chemical, I-chemical, O, ...
```

3. Use Hugging Face `AutoTokenizer` with offset mappings.
4. Align character-level entity spans to token labels.
5. Train token classification model:

```python
AutoModelForTokenClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
```

Alternative models:

```text
allenai/scibert_scivocab_uncased
dmis-lab/biobert-base-cased-v1.1
microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

6. Decode token predictions back to character spans.
7. Postprocess invalid spans.
8. Export to T611 JSON.

### Recommended experiments

```text
NER-1: dictionary baseline
NER-2: GLiNER zero-shot / baseline
NER-3: GLiNER fine-tuned on gold
NER-4: GLiNER fine-tuned on gold+silver
NER-5: PubMedBERT token classification on gold
NER-6: PubMedBERT token classification on gold+silver
NER-7: SciBERT token classification on gold
```

### Acceptance criteria

- At least one stronger NER model beats the dictionary baseline on dev micro-F1.
- The best NER model is saved and reusable for RE prediction.

---

## Phase 6 — Mention-Level Relation Extraction Baseline

### Goal

Build a relation extraction pipeline using entity mentions.

### Important design choice

Relation extraction depends on entity mentions. Implement two modes:

```text
Gold-entity mode:
  Use gold dev entities to evaluate the RE classifier independently.

Predicted-entity mode:
  Use NER predictions as input to evaluate the full pipeline.
```

Gold-entity mode is useful for understanding RE quality. Predicted-entity mode reflects the real end-to-end system.

### Candidate generation

For each article:

1. collect entity mentions;
2. generate ordered subject-object pairs;
3. filter by valid entity-type combinations from the relation schema;
4. assign label if the pair exists in gold relations;
5. otherwise assign `no_relation`.

Implement `src/gutbrainie/re/relation_schema.py`:

```python
VALID_RELATIONS = {
    ("anatomical location", "human"): ["located in"],
    ("anatomical location", "animal"): ["located in"],
    ("bacteria", "DDF"): ["influence"],
    ("bacteria", "gene"): ["change expression"],
    ("bacteria", "microbiome"): ["part of"],
    ("chemical", "microbiome"): ["impact", "produced by"],
    ...
}
```

Use the full schema from the challenge guidelines.

### Rule baseline

Implement a simple prior-based baseline:

1. Estimate the most frequent predicate for each `(subject_label, object_label)` pair from train.
2. At prediction time, generate valid candidate pairs.
3. Predict the most frequent predicate if its prior probability is above a threshold.
4. Otherwise predict no relation.

Example:

```text
(microbiome, DDF) -> is linked to
(bacteria, DDF) -> influence
(DDF, human) -> target
```

### Steps for Codex

1. Implement `src/gutbrainie/re/candidates.py`:

```python
def generate_relation_candidates(articles, entities, valid_schema, max_distance=None) -> pd.DataFrame:
    ...
```

2. Candidate features:

```text
pmid
subject_text_span
subject_label
object_text_span
object_label
subject_location
object_location
subject_start_idx
subject_end_idx
object_start_idx
object_end_idx
text_between
sentence_distance
candidate_key
```

3. Implement `src/gutbrainie/re/rule_baseline.py`:

```python
class RelationPriorBaseline:
    def fit(self, train_relations): ...
    def predict(self, candidates): ...
```

4. Export to T621 JSON:

```bash
python -m gutbrainie.cli predict-re-rule \
  --articles Articles/csv_format/articles_dev.csv \
  --entities outputs/predictions/dev_t611_gliner_best.json \
  --train-relations Annotations/Train/gold_quality/csv_format/train_gold_mention_level_relations.csv \
  --output outputs/predictions/dev_t621_rule.json
```

### Acceptance criteria

- Candidate generation works on gold and predicted entities.
- Rule baseline produces valid `mention_level_relations` JSON.
- Evaluation reports micro/macro F1.

---

## Phase 7 — Stronger Mention-Level RE Model

Implement at least one trainable RE model if time allows.

## Option 1: PubMedBERT / BioLinkBERT pair classifier

### Why

This is easier to implement and explain than ATLOP. It is suitable for a course project.

### Input format

Create one classification example per candidate pair. Insert entity markers into the article text:

```text
[SUBJ_BACTERIA] Lactobacillus [/SUBJ_BACTERIA] reduced symptoms of [OBJ_DDF] depression [/OBJ_DDF] in mice.
```

Target label:

```text
influence
```

For negative examples:

```text
no_relation
```

### Candidate context

Use one of these context strategies:

1. title only if both mentions are in title;
2. abstract only if both mentions are in abstract;
3. title + separator + abstract if mentions are in different locations;
4. sentence window around both mentions if sentence splitting is implemented.

Start with option 3 for simplicity.

### Steps for Codex

1. Implement `src/gutbrainie/re/train_pair_classifier.py`.
2. Generate candidate examples from train gold.
3. Add negative sampling because `no_relation` candidates will dominate.
4. Train a multi-class classifier:

```python
AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    num_labels=num_relation_labels
)
```

5. Alternative encoders:

```text
michiyasunaga/BioLinkBERT-base
allenai/scibert_scivocab_uncased
dmis-lab/biobert-base-cased-v1.1
microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

6. Recommended training parameters:

```yaml
max_length: 512
learning_rate: 2e-5
batch_size: 8
epochs: 5
negative_sampling_ratio: 3
weight_decay: 0.01
warmup_ratio: 0.1
metric_for_best_model: micro_f1_without_no_relation
```

7. Predict relations by:

- generating candidates from predicted NER entities;
- classifying each candidate;
- removing `no_relation`;
- applying confidence threshold;
- exporting T621 JSON.

### Recommended experiments

```text
RE-1: relation prior baseline using gold entities
RE-2: relation prior baseline using predicted entities
RE-3: PubMedBERT pair classifier using gold entities
RE-4: PubMedBERT pair classifier using predicted entities
RE-5: BioLinkBERT pair classifier using predicted entities
RE-6: train on gold only vs gold+silver
```

## Option 2: ATLOP baseline reproduction

### Why

The official baseline uses ATLOP for relation extraction. It is likely stronger, but setup may be heavier.

### Steps for Codex

1. Clone the official baseline repository into `external/`.
2. Do not modify it directly unless necessary.
3. Use its conversion notebooks/scripts as reference.
4. Reproduce the official ATLOP training pipeline if environment setup permits.
5. Save results into the local project outputs folder.
6. Document all required changes in `outputs/reports/atlop_notes.md`.

Recommended approach:

```text
Use ATLOP as an optional comparison, not as the only RE implementation.
```

### Acceptance criteria

- Pair classifier is preferred as the main project implementation.
- ATLOP reproduction is a bonus experiment.

---

## Phase 8 — LLM / Ollama / GPT Experiments

LLMs should be treated as optional experiments, not the main required solution.

### Suitable uses

Use LLMs for:

- relation verification on candidate pairs;
- generating weak labels for additional training data;
- error analysis explanations;
- suggesting rules for relation patterns;
- comparing zero-shot relation extraction with trained models.

Avoid using LLMs as the only project solution, because exact character offsets for NER are difficult to guarantee.

## Option 1: Ollama local models

Recommended local models to test:

```text
llama3.1:8b-instruct
llama3.2:3b-instruct
qwen2.5:7b-instruct
mistral-nemo:12b-instruct
biomistral:7b
```

Example use case: relation verification.

Input prompt:

```text
You are a biomedical relation extraction assistant.
Given the article text and two entity mentions, decide whether one of the allowed GutBrainIE predicates holds.
Return only JSON.

Allowed predicates: located in, interact, influence, change expression, part of, impact, produced by, administered, strike, change abundance, affect, is a, target, change effect, used by, is linked to, compared to, no_relation.

Article:
{article_text}

Subject: {subject_text_span} [{subject_label}]
Object: {object_text_span} [{object_label}]

Return:
{"predicate": "...", "confidence": 0.0}
```

Implementation file:

```text
src/gutbrainie/llm/ollama_relation_verifier.py
```

Example command:

```bash
ollama pull qwen2.5:7b-instruct
python -m gutbrainie.cli predict-re-ollama \
  --articles Articles/csv_format/articles_dev.csv \
  --entities outputs/predictions/dev_t611_gliner_best.json \
  --model qwen2.5:7b-instruct \
  --output outputs/predictions/dev_t621_ollama_qwen.json
```

## Option 2: GPT models

Use GPT only if API access is available.

Suitable tasks:

- relation verification;
- error analysis;
- generating human-readable explanations for the report;
- creating additional weak labels, clearly marked as weak/LLM-generated.

Do not send restricted data to external APIs unless the dataset license and course rules permit it.

Implementation file:

```text
src/gutbrainie/llm/gpt_relation_verifier.py
```

### Recommended LLM experiment design

```text
LLM-RE-1: Ollama zero-shot relation verifier on 100 dev articles
LLM-RE-2: Ollama few-shot relation verifier on 100 dev articles
LLM-RE-3: GPT relation verifier on small dev sample, if allowed
LLM-RE-4: LLM used only for error analysis, not prediction
```

### Acceptance criteria

- LLM results are clearly separated from trained ML model results.
- Prompt, model name, date, and sampling settings are logged.
- LLM outputs are parsed and validated before evaluation.

---

## Phase 9 — End-to-End Prediction Pipeline

### Goal

Create one command that runs the full project pipeline.

### Pipeline

```text
articles
  -> NER model
  -> predicted entities
  -> RE candidate generation
  -> RE model
  -> predicted mention-level relations
  -> evaluation / submission export
```

### Command

```bash
python -m gutbrainie.cli run-pipeline \
  --data-root data/gutbrainie2026 \
  --split dev \
  --ner-model outputs/models/ner_gliner_best \
  --re-model outputs/models/re_pubmedbert_best \
  --output-dir outputs/predictions/pipeline_dev
```

### Outputs

```text
outputs/predictions/pipeline_dev/dev_t611_entities.json
outputs/predictions/pipeline_dev/dev_t621_mention_relations.json
outputs/reports/pipeline_dev_metrics.json
```

### Acceptance criteria

- The pipeline can run from one command.
- It works on dev and test articles.
- It logs all config values used for the run.

---

## Phase 10 — Submission Export

### Goal

Generate challenge-compatible JSON files and optional zip folders.

### T611 export

Implement:

```bash
python -m gutbrainie.cli export-t611 \
  --entities outputs/predictions/dev_entities_best.csv \
  --output outputs/submissions/teamID_T611_run1_gliner.json
```

JSON structure:

```json
{
  "PMID": {
    "entities": [
      {
        "start_idx": 0,
        "end_idx": 10,
        "location": "abstract",
        "text_span": "...",
        "label": "DDF"
      }
    ]
  }
}
```

### T621 export

Implement:

```bash
python -m gutbrainie.cli export-t621 \
  --relations outputs/predictions/dev_relations_best.csv \
  --output outputs/submissions/teamID_T621_run1_pubmedbert.json
```

JSON structure:

```json
{
  "PMID": {
    "mention_level_relations": [
      {
        "subject_text_span": "...",
        "subject_label": "microbiome",
        "predicate": "is linked to",
        "object_text_span": "...",
        "object_label": "DDF"
      }
    ]
  }
}
```

### Metadata files

Generate `.meta` files automatically:

```text
Team ID: teamID
Task ID: T611
Run ID: run1
System description: gliner
Training data: train_gold + train_silver
Preprocessing: offset validation, duplicate removal, title/abstract split
Model: GLiNER fine-tuned on GutBrainIE
Repository: <github-url>
```

### Zip structure

```text
teamID_GutBrainIE_2026.zip
└── teamID_T611_run1_gliner/
    ├── teamID_T611_run1_gliner.json
    └── teamID_T611_run1_gliner.meta
└── teamID_T621_run1_pubmedbert/
    ├── teamID_T621_run1_pubmedbert.json
    └── teamID_T621_run1_pubmedbert.meta
```

### Acceptance criteria

- JSON files contain only fields relevant to the task.
- T611 output does not include `uri`.
- T621 output uses `mention_level_relations`.
- Submission validation passes.

---

## Phase 11 — Model Report

### Goal

Create a final report suitable for the course assignment.

### Required report sections

```text
1. Introduction
2. Task Description
3. Dataset Description
4. Preprocessing
5. Named Entity Recognition Methods
6. Mention-Level Relation Extraction Methods
7. Experimental Setup
8. Results
9. Error Analysis
10. Discussion
11. Conclusion
12. References
```

### Required experiment table

Create `outputs/reports/model_report.csv` with columns:

```text
experiment_id
task
model
training_data
entity_source_for_re
macro_precision
macro_recall
macro_f1
micro_precision
micro_recall
micro_f1
notes
```

Example rows:

```text
NER-1,T611,Dictionary,Gold,N/A,0.30,0.20,0.24,0.42,0.31,0.36,Exact match baseline
NER-2,T611,GLiNER,Gold,N/A,...
RE-1,T621,RelationPrior,Gold,Gold entities,...
RE-2,T621,PubMedBERT Pair Classifier,Gold,Predicted entities,...
```

### Error analysis

Add scripts that sample errors:

```bash
python -m gutbrainie.cli analyze-ner-errors \
  --gold Annotations/Dev/csv_format/dev_entities.csv \
  --pred outputs/predictions/dev_t611_best.json \
  --output outputs/reports/ner_errors.md

python -m gutbrainie.cli analyze-re-errors \
  --gold Annotations/Dev/csv_format/dev_mention_level_relations.csv \
  --pred outputs/predictions/dev_t621_best.json \
  --output outputs/reports/re_errors.md
```

Analyze:

- missed abbreviations;
- overlapping entity spans;
- ambiguous labels such as `chemical` vs `drug`;
- common false positive entity types;
- relation direction errors;
- relation predicate confusion, e.g. `impact` vs `influence`;
- cascading errors from NER into RE.

### Acceptance criteria

- Report includes at least 5 NER experiments and 3 RE experiments if time permits.
- Report explains why the final selected models were chosen.
- Report includes examples of correct and incorrect predictions.

---

## 6. Development Order for Codex

Implement in this exact order:

1. Repository skeleton.
2. Data loaders.
3. Offset validation.
4. Internal NER evaluation.
5. Internal RE evaluation.
6. T611 JSON export.
7. T621 JSON export.
8. Dictionary NER baseline.
9. Relation candidate generator.
10. Relation prior baseline.
11. End-to-end baseline pipeline.
12. GLiNER NER implementation.
13. PubMedBERT/SciBERT token-classification NER, optional.
14. PubMedBERT/BioLinkBERT pair-classifier RE.
15. Official evaluation wrapper.
16. Submission zip generator.
17. Model report generation.
18. Error analysis notebooks/scripts.
19. Optional Ollama/GPT experiments.
20. Final README and reproducibility instructions.

Do not start with LLM experiments. First create deterministic baselines and evaluation.

---

## 7. Recommended Experiments

### NER experiments

| ID | Model | Training Data | Notes |
|---|---|---|---|
| NER-1 | Dictionary baseline | Gold | Exact-match lower bound |
| NER-2 | GLiNER zero-shot | None / labels only | Fast baseline |
| NER-3 | GLiNER fine-tuned | Gold | Main baseline |
| NER-4 | GLiNER fine-tuned | Gold + Silver | Stronger model |
| NER-5 | PubMedBERT token classifier | Gold | Classical transformer NER |
| NER-6 | SciBERT token classifier | Gold | General scientific encoder |
| NER-7 | PubMedBERT token classifier | Gold + Silver | Data scaling experiment |

### RE experiments

| ID | Model | Entity Source | Training Data | Notes |
|---|---|---|---|---|
| RE-1 | Relation prior baseline | Gold entities | Gold | Isolated RE lower bound |
| RE-2 | Relation prior baseline | Predicted entities | Gold | End-to-end baseline |
| RE-3 | PubMedBERT pair classifier | Gold entities | Gold | Isolated trainable RE |
| RE-4 | PubMedBERT pair classifier | Predicted entities | Gold | End-to-end trainable RE |
| RE-5 | BioLinkBERT pair classifier | Predicted entities | Gold + Silver | Stronger biomedical RE |
| RE-6 | ATLOP | Predicted entities | Gold + Silver + Silver 2025 | Optional official-baseline reproduction |
| RE-7 | Ollama verifier | Predicted entities | No training | Optional small-sample LLM experiment |

---

## 8. Suggested Final README Commands

The final repository README should contain commands like these:

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Validate data
python -m gutbrainie.cli prepare-data \
  --data-root data/gutbrainie2026 \
  --quality gold

# 3. Run NER dictionary baseline
python -m gutbrainie.cli predict-ner-dictionary \
  --data-root data/gutbrainie2026 \
  --split dev \
  --output outputs/predictions/dev_t611_dictionary.json

# 4. Evaluate NER
python -m gutbrainie.cli evaluate-ner \
  --gold data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv \
  --pred outputs/predictions/dev_t611_dictionary.json

# 5. Train stronger NER
python -m gutbrainie.cli train-ner-gliner \
  --data-root data/gutbrainie2026 \
  --train-quality gold \
  --output outputs/models/ner_gliner_gold

# 6. Predict entities
python -m gutbrainie.cli predict-ner-gliner \
  --model outputs/models/ner_gliner_gold \
  --data-root data/gutbrainie2026 \
  --split dev \
  --output outputs/predictions/dev_t611_gliner.json

# 7. Run RE baseline
python -m gutbrainie.cli predict-re-rule \
  --data-root data/gutbrainie2026 \
  --split dev \
  --entities outputs/predictions/dev_t611_gliner.json \
  --output outputs/predictions/dev_t621_rule.json

# 8. Train RE pair classifier
python -m gutbrainie.cli train-re-pair-classifier \
  --data-root data/gutbrainie2026 \
  --train-quality gold \
  --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --output outputs/models/re_pubmedbert_gold

# 9. Predict relations
python -m gutbrainie.cli predict-re-pair-classifier \
  --model outputs/models/re_pubmedbert_gold \
  --data-root data/gutbrainie2026 \
  --split dev \
  --entities outputs/predictions/dev_t611_gliner.json \
  --output outputs/predictions/dev_t621_pubmedbert.json

# 10. Generate report
python -m gutbrainie.cli build-report \
  --outputs-root outputs \
  --output outputs/reports/model_report.md
```

---

## 9. Testing Requirements

Add unit tests for the critical parts of the pipeline.

### Required tests

```text
test_data_loading.py
  - article CSV loading
  - entity CSV loading
  - relation CSV loading

test_offsets.py
  - valid entity offset check
  - invalid offset detection
  - title vs abstract offset separation

test_bio_conversion.py
  - char span to BIO labels
  - BIO labels back to char spans
  - overlapping entity handling

test_re_candidates.py
  - candidate generation from entity pairs
  - invalid type pair filtering
  - positive relation matching
  - no_relation assignment

test_submission_format.py
  - T611 JSON contains entities only
  - T611 JSON does not contain uri
  - T621 JSON contains mention_level_relations only
  - required fields are present

test_metrics.py
  - exact match true positive
  - false positive
  - false negative
  - micro/macro F1 calculation
```

Run tests:

```bash
pytest -q
```

---

## 10. Practical Notes and Risks

### Risk 1: Character offset bugs

Most NER errors in implementation will come from wrong offset handling. Always process title and abstract separately.

### Risk 2: Multiple annotators

Some CSV files contain an `annotator` column. Start with exact deduplication. If conflicts appear, use this priority:

```text
expert/gold > student/silver > distant/bronze
```

For the course project, document the decision instead of overengineering adjudication.

### Risk 3: Class imbalance

DDF, chemical, microbiome, human, and bacteria are frequent. Rare classes such as food, gene, drug, and statistical technique may have lower F1. Report macro-F1 to show this.

### Risk 4: RE depends heavily on NER

Relation extraction performance will drop when using predicted entities instead of gold entities. Report both settings.

### Risk 5: ATLOP setup complexity

ATLOP is useful as an official baseline reference, but it can be harder to set up. The pair-classifier approach is safer for the course project.

### Risk 6: LLM output validity

LLM outputs must be parsed, validated, and constrained to the official predicate labels. Never trust free-form text directly.

---

## 11. Final Deliverables

The final project should contain:

```text
README.md
IMPLEMENTATION_PLAN.md
requirements.txt
src/gutbrainie/**
tests/**
configs/**
outputs/reports/model_report.md
outputs/reports/model_report.csv
outputs/reports/ner_errors.md
outputs/reports/re_errors.md
outputs/predictions/dev_t611_best.json
outputs/predictions/dev_t621_best.json
outputs/submissions/<team>_T611_<run>_<desc>.json
outputs/submissions/<team>_T621_<run>_<desc>.json
```

For the course presentation/report, include:

- task explanation;
- dataset statistics;
- model architecture diagrams or descriptions;
- experiment table;
- best model choice;
- error examples;
- limitations;
- future work.

---

## 12. Recommended Final Architecture

The safest and most complete architecture for the course assignment is:

```text
Data Loader
  -> Offset Validator
  -> NER Dictionary Baseline
  -> GLiNER Fine-Tuned NER
  -> T611 Export + Evaluation
  -> RE Candidate Generator
  -> Relation Prior Baseline
  -> PubMedBERT Pair Classifier
  -> T621 Export + Evaluation
  -> Model Report + Error Analysis
```

Optional advanced additions:

```text
ATLOP baseline reproduction
Ollama/GPT relation verification
PubMedBERT token-classification NER
Gold vs Gold+Silver vs Gold+Silver+Silver2025 comparison
```

---

## 13. References

- GutBrainIE @ CLEF 2026 official challenge page: https://hereditary.dei.unipd.it/challenges/gutbrainie/2026/
- Official GutBrainIE 2026 baseline repository: https://github.com/MMartinelli-hub/GutBrainIE_2026_Baseline
- GLiNER: https://github.com/urchade/GLiNER
- ATLOP: https://github.com/wzhouad/ATLOP
- PubMedBERT / BiomedBERT: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
- SciBERT: https://huggingface.co/allenai/scibert_scivocab_uncased
- BioBERT: https://huggingface.co/dmis-lab/biobert-base-cased-v1.1
- BioLinkBERT: https://huggingface.co/michiyasunaga/BioLinkBERT-base
- Ollama: https://ollama.com/
