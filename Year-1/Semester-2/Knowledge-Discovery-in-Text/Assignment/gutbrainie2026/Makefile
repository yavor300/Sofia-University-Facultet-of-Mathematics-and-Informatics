PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTHONPATH ?= src
DATA_ROOT ?= data/gutbrainie2026
OUTPUTS_ROOT ?= outputs
QUALITY ?= gold
CONFIG ?= configs/paths.yaml
NER_TRANSFORMER_CONFIG ?= configs/ner_transformer.yaml
NER_PUBMEDBERT_CONFIG ?= configs/ner_pubmedbert.yaml
NER_BIOMEDBERT_CONFIG ?= configs/ner_biomedbert.yaml
NER_SCIBERT_CONFIG ?= configs/ner_scibert.yaml
NER_BIOBERT_CONFIG ?= configs/ner_biobert.yaml
GLINER_CONFIG ?= configs/ner_gliner.yaml
GLINER_CPU_CONFIG ?= configs/ner_gliner_cpu.yaml
RE_TRANSFORMER_CONFIG ?= configs/re_transformer.yaml
T611_OUTPUT ?= $(OUTPUTS_ROOT)/submissions/t611_predictions.json
T621_OUTPUT ?= $(OUTPUTS_ROOT)/submissions/t621_predictions.json
EDA_QUALITIES ?= gold dev
EVAL_TASK ?= ner
GOLD ?= data/gutbrainie2026/Annotations/Dev/csv_format/dev_entities.csv
PREDICTION ?= outputs/predictions/dev_predictions.csv
METRICS_OUTPUT ?= $(OUTPUTS_ROOT)/reports/metrics_$(EVAL_TASK).json
OFFICIAL_REPO ?= external/GutBrainIE_2026_Baseline
OFFICIAL_ENTRYPOINT ?=
ATLOP_ACTION ?= compose
ATLOP_OUTPUT ?= $(OUTPUTS_ROOT)/predictions/atlop_predicted_relations_raw.json
ATLOP_NOTES_OUTPUT ?= $(OUTPUTS_ROOT)/reports/atlop_notes.md
ATLOP_LOG ?= $(OUTPUTS_ROOT)/reports/atlop_$(ATLOP_ACTION).log
ATLOP_DRY_RUN ?=
ATLOP_PREDICTED_ENTITIES ?=
TRAIN_ENTITIES ?= $(DATA_ROOT)/Annotations/Train/gold_quality/csv_format/train_gold_entities.csv
ARTICLES ?= $(DATA_ROOT)/Articles/csv_format/articles_dev.csv
RE_ENTITIES ?= $(DATA_ROOT)/Annotations/Dev/csv_format/dev_entities.csv
TRAIN_RELATIONS ?= $(DATA_ROOT)/Annotations/Train/gold_quality/csv_format/train_gold_mention_level_relations.csv
RE_RULE_OUTPUT ?= $(OUTPUTS_ROOT)/predictions/dev_t621_rule.json
RE_RULE_THRESHOLD ?= 0.5
RE_RULE_MAX_DISTANCE ?=
RE_PAIR_EXPERIMENT ?= gold
RE_PAIR_MODEL ?=
RE_PAIR_MODEL_DIR ?= $(OUTPUTS_ROOT)/models/re_pair_classifier_$(RE_PAIR_EXPERIMENT)
RE_PAIR_OUTPUT ?= $(OUTPUTS_ROOT)/predictions/dev_t621_pair_classifier_$(RE_PAIR_EXPERIMENT).json
RE_PAIR_THRESHOLD ?= 0.5
RE_PAIR_BATCH_SIZE ?= 8
RE_PAIR_MAX_LENGTH ?= 512
OLLAMA_MODEL ?= llama3.1:8b
OLLAMA_BASE_URL ?= http://localhost:11434
OLLAMA_OUTPUT ?= $(OUTPUTS_ROOT)/predictions/dev_t621_ollama_llama31.json
OLLAMA_DECISIONS_OUTPUT ?= $(OUTPUTS_ROOT)/reports/dev_t621_ollama_llama31_decisions.jsonl
OLLAMA_THRESHOLD ?= 0.5
OLLAMA_MAX_CANDIDATES ?= 200
OLLAMA_MAX_DISTANCE ?= 0
OLLAMA_TIMEOUT ?= 120
OLLAMA_TEMPERATURE ?= 0.0
GPT_MODEL ?=
GPT_ENV_FILE ?= .env
GPT_OUTPUT ?= $(OUTPUTS_ROOT)/predictions/dev_t621_gpt.json
GPT_DECISIONS_OUTPUT ?= $(OUTPUTS_ROOT)/reports/dev_t621_gpt_decisions.jsonl
GPT_THRESHOLD ?= 0.5
GPT_MAX_CANDIDATES ?= 50
GPT_MAX_DISTANCE ?= 0
GPT_TIMEOUT ?= 120
GPT_TEMPERATURE ?= 0.0
NER_DICTIONARY_OUTPUT ?= $(OUTPUTS_ROOT)/predictions/dev_t611_dictionary.json
GLINER_EXPERIMENT ?= gold
GLINER_DATA_DIR ?= $(OUTPUTS_ROOT)/gliner
GLINER_MODEL ?=
GLINER_MODEL_DIR ?= $(OUTPUTS_ROOT)/models/gliner_$(GLINER_EXPERIMENT)
GLINER_TRAIN_DATA ?= $(GLINER_DATA_DIR)/gliner_$(GLINER_EXPERIMENT)_train.jsonl
GLINER_VALIDATION_DATA ?= $(GLINER_DATA_DIR)/gliner_$(GLINER_EXPERIMENT)_validation.jsonl
GLINER_OUTPUT ?= $(OUTPUTS_ROOT)/predictions/dev_t611_gliner_$(GLINER_EXPERIMENT).json
TOKEN_EXPERIMENT ?= gold
TOKEN_MODEL ?=
TOKEN_MODEL_DIR ?= $(OUTPUTS_ROOT)/models/token_classifier_$(TOKEN_EXPERIMENT)
TOKEN_OUTPUT ?= $(OUTPUTS_ROOT)/predictions/dev_t611_token_classifier_$(TOKEN_EXPERIMENT).json
TOKEN_BATCH_SIZE ?= 8
TOKEN_MAX_LENGTH ?= 512

CLI = PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m gutbrainie.cli
VALIDATION_OUTPUT = $(OUTPUTS_ROOT)/reports/data_validation_$(QUALITY).json
OFFICIAL_ENTRYPOINT_ARG = $(if $(OFFICIAL_ENTRYPOINT),--entrypoint $(OFFICIAL_ENTRYPOINT),)
RE_PAIR_MODEL_ARG = $(if $(RE_PAIR_MODEL),--model $(RE_PAIR_MODEL),)
ATLOP_DRY_RUN_ARG = $(if $(ATLOP_DRY_RUN),--dry-run,)
ATLOP_PREDICTED_ENTITIES_ARG = $(if $(ATLOP_PREDICTED_ENTITIES),--predicted-entities $(ATLOP_PREDICTED_ENTITIES),)
GPT_MODEL_ARG = $(if $(GPT_MODEL),--model $(GPT_MODEL),)

.PHONY: help setup install install-gliner install-atlop test cli-help prepare-data validate-all eda eda-all \
	install-gpt \
	run-ner-baseline predict-ner-dictionary run-ner-transformer run-re-baseline run-re-transformer \
	prepare-gliner-data prepare-gliner-all train-gliner train-gliner-cpu train-gliner-smoke \
	predict-gliner train-token-classifier train-pubmedbert-token-classifier \
	train-biomedbert-token-classifier train-scibert-token-classifier train-biobert-token-classifier \
	predict-token-classifier predict-re-rule train-re-pair-classifier predict-re-pair-classifier \
	predict-re-ollama predict-re-gpt atlop-notes prepare-atlop-data run-atlop evaluate evaluate-official export-t611 export-t621 clean

help:
	@echo "GutBrainIE 2026 commands"
	@echo ""
	@echo "Setup and checks:"
	@echo "  make setup                Create .venv and install requirements/package"
	@echo "  make install              Install requirements and editable package"
	@echo "  make install-gliner       Install optional GLiNER dependency"
	@echo "  make install-atlop        Install optional official ATLOP dependencies"
	@echo "  make install-gpt          Install optional OpenAI dependency"
	@echo "  make test                 Run pytest"
	@echo "  make cli-help             Show CLI help"
	@echo ""
	@echo "Data validation:"
	@echo "  make prepare-data         Validate one split; override QUALITY=gold|silver|silver_2025|bronze|dev"
	@echo "  make validate-all         Generate validation reports for all available train/dev splits"
	@echo "  make eda                  Generate data statistics CSVs and plots"
	@echo "  make eda-all              Generate EDA for gold, silver, silver_2025, bronze, and dev"
	@echo ""
	@echo "Experiment commands:"
	@echo "  make run-ner-baseline     Run dictionary NER baseline"
	@echo "  make predict-ner-dictionary  Run dictionary NER baseline"
	@echo "  make run-ner-transformer  Run transformer NER command"
	@echo "  make prepare-gliner-data  Prepare GLiNER JSONL train/validation data"
	@echo "  make prepare-gliner-all   Prepare all planned GLiNER experiment datasets"
	@echo "  make train-gliner         Fine-tune GLiNER using prepared JSONL data"
	@echo "  make train-gliner-cpu     Fine-tune GLiNER with CPU-safe config"
	@echo "  make train-gliner-smoke   One-command CPU-safe smoke training"
	@echo "  make predict-gliner       Predict T611 JSON with GLiNER"
	@echo "  make train-token-classifier  Fine-tune PubMedBERT/BioBERT/SciBERT token classifier"
	@echo "  make train-pubmedbert-token-classifier  Fine-tune PubMedBERT token classifier"
	@echo "  make train-biomedbert-token-classifier  Fine-tune BiomedBERT token classifier"
	@echo "  make train-scibert-token-classifier  Fine-tune SciBERT token classifier"
	@echo "  make train-biobert-token-classifier  Fine-tune BioBERT token classifier"
	@echo "  make predict-token-classifier  Predict T611 JSON with trained token classifier"
	@echo "  make run-re-baseline      Run relation baseline"
	@echo "  make predict-re-rule      Predict T621 JSON with prior RE baseline"
	@echo "  make train-re-pair-classifier  Fine-tune PubMedBERT RE pair classifier"
	@echo "  make predict-re-pair-classifier  Predict T621 JSON with trained RE pair classifier"
	@echo "  make predict-re-ollama    Verify T621 candidates with local Ollama"
	@echo "  make predict-re-gpt       Verify T621 candidates with OpenAI GPT"
	@echo "  make atlop-notes           Inspect optional official ATLOP setup"
	@echo "  make prepare-atlop-data    Convert local data to official ATLOP JSON files"
	@echo "  make run-atlop             Run optional official ATLOP step; override ATLOP_ACTION=compose|finetune|predict"
	@echo "  make run-re-transformer   Run transformer relation command"
	@echo "  make evaluate             Run evaluation command"
	@echo "  make evaluate-official    Run official evaluator wrapper"
	@echo "  make export-t611          Export T611 JSON"
	@echo "  make export-t621          Export T621 JSON"
	@echo ""
	@echo "Examples:"
	@echo "  make prepare-data QUALITY=dev"
	@echo "  make eda EDA_QUALITIES=\"gold dev\""
	@echo "  make eda-all"
	@echo "  make evaluate EVAL_TASK=ner GOLD=gold.csv PREDICTION=pred.csv"
	@echo "  make evaluate-official OFFICIAL_REPO=external/GutBrainIE_2026_Baseline PREDICTION=pred.json"
	@echo "  make predict-ner-dictionary TRAIN_ENTITIES=train.csv ARTICLES=articles.csv"
	@echo "  make prepare-gliner-data GLINER_EXPERIMENT=gold_silver"
	@echo "  make prepare-gliner-all"
	@echo "  make predict-gliner GLINER_MODEL=outputs/models/gliner_gold ARTICLES=articles.csv"
	@echo "  make train-gliner-cpu GLINER_EXPERIMENT=gold GLINER_MODEL=urchade/gliner_medium-v2.1"
	@echo "  make train-token-classifier TOKEN_EXPERIMENT=gold"
	@echo "  make train-pubmedbert-token-classifier TOKEN_EXPERIMENT=gold"
	@echo "  make train-biomedbert-token-classifier TOKEN_EXPERIMENT=gold"
	@echo "  make train-scibert-token-classifier TOKEN_EXPERIMENT=gold"
	@echo "  make train-biobert-token-classifier TOKEN_EXPERIMENT=gold"
	@echo "  make predict-token-classifier TOKEN_MODEL_DIR=outputs/models/token_classifier_gold"
	@echo "  make predict-re-rule RE_ENTITIES=outputs/predictions/dev_t611_best.json"
	@echo "  make train-re-pair-classifier RE_PAIR_EXPERIMENT=gold"
	@echo "  make predict-re-pair-classifier RE_PAIR_MODEL_DIR=outputs/models/re_pair_classifier_gold RE_ENTITIES=outputs/predictions/dev_t611_best.json"
	@echo "  make predict-re-ollama RE_ENTITIES=outputs/predictions/dev_t611_best.json OLLAMA_MODEL=llama3.1:8b"
	@echo "  make predict-re-gpt RE_ENTITIES=outputs/predictions/dev_t611_best.json GPT_MAX_CANDIDATES=50"
	@echo "  make atlop-notes"
	@echo "  make prepare-atlop-data"
	@echo "  make run-atlop ATLOP_ACTION=compose ATLOP_DRY_RUN=1"

setup:
	@$(PYTHON) -m venv .venv
	@.venv/bin/python -m pip install -r requirements.txt
	@.venv/bin/python -m pip install -e .

install:
	@$(PYTHON) -m pip install -r requirements.txt
	@$(PYTHON) -m pip install -e .

install-gliner:
	@$(PYTHON) -m pip install -r requirements-gliner.txt

install-atlop:
	@$(PYTHON) -m pip install -r requirements-atlop.txt

install-gpt:
	@$(PYTHON) -m pip install -r requirements-gpt.txt

test:
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

cli-help:
	@$(CLI) --help

prepare-data:
	@$(CLI) prepare-data --data-root $(DATA_ROOT) --quality $(QUALITY) --output $(VALIDATION_OUTPUT)

validate-all:
	$(MAKE) prepare-data QUALITY=gold
	$(MAKE) prepare-data QUALITY=silver
	$(MAKE) prepare-data QUALITY=silver_2025
	$(MAKE) prepare-data QUALITY=bronze
	$(MAKE) prepare-data QUALITY=dev

eda:
	@$(CLI) eda --data-root $(DATA_ROOT) --output-dir $(OUTPUTS_ROOT)/reports --qualities $(EDA_QUALITIES)

eda-all:
	@$(MAKE) eda EDA_QUALITIES="gold silver silver_2025 bronze dev"

run-ner-baseline:
	@$(MAKE) predict-ner-dictionary

predict-ner-dictionary:
	@$(CLI) predict-ner-dictionary --train-entities $(TRAIN_ENTITIES) --articles $(ARTICLES) --output $(NER_DICTIONARY_OUTPUT)

run-ner-transformer:
	@$(MAKE) predict-gliner

prepare-gliner-data:
	@$(CLI) prepare-gliner-data --data-root $(DATA_ROOT) --experiment $(GLINER_EXPERIMENT) --output-dir $(GLINER_DATA_DIR)

prepare-gliner-all:
	@$(MAKE) prepare-gliner-data GLINER_EXPERIMENT=gold
	@$(MAKE) prepare-gliner-data GLINER_EXPERIMENT=gold_silver
	@$(MAKE) prepare-gliner-data GLINER_EXPERIMENT=gold_silver_silver_2025

train-gliner:
	@$(CLI) train-gliner --config $(GLINER_CONFIG) --model "$(GLINER_MODEL)" --train-data $(GLINER_TRAIN_DATA) --validation-data $(GLINER_VALIDATION_DATA) --output-dir $(GLINER_MODEL_DIR)

train-gliner-cpu:
	@$(CLI) train-gliner --config $(GLINER_CPU_CONFIG) --model "$(GLINER_MODEL)" --train-data $(GLINER_TRAIN_DATA) --validation-data $(GLINER_VALIDATION_DATA) --output-dir $(GLINER_MODEL_DIR)

train-gliner-smoke:
	@$(MAKE) prepare-gliner-data GLINER_EXPERIMENT=$(GLINER_EXPERIMENT)
	@$(MAKE) train-gliner-cpu GLINER_EXPERIMENT=$(GLINER_EXPERIMENT) GLINER_MODEL=$(GLINER_MODEL)

predict-gliner:
	@$(CLI) predict-gliner --config $(GLINER_CONFIG) --model "$(GLINER_MODEL)" --articles $(ARTICLES) --output $(GLINER_OUTPUT)

train-token-classifier:
	@$(CLI) train-token-classifier --config $(NER_TRANSFORMER_CONFIG) --data-root $(DATA_ROOT) --experiment $(TOKEN_EXPERIMENT) --model "$(TOKEN_MODEL)" --output-dir $(TOKEN_MODEL_DIR)

train-pubmedbert-token-classifier:
	@$(CLI) train-token-classifier --config $(NER_PUBMEDBERT_CONFIG) --data-root $(DATA_ROOT) --experiment $(TOKEN_EXPERIMENT) --output-dir $(OUTPUTS_ROOT)/models/pubmedbert_$(TOKEN_EXPERIMENT)

train-biomedbert-token-classifier:
	@$(CLI) train-token-classifier --config $(NER_BIOMEDBERT_CONFIG) --data-root $(DATA_ROOT) --experiment $(TOKEN_EXPERIMENT) --output-dir $(OUTPUTS_ROOT)/models/biomedbert_$(TOKEN_EXPERIMENT)

train-scibert-token-classifier:
	@$(CLI) train-token-classifier --config $(NER_SCIBERT_CONFIG) --data-root $(DATA_ROOT) --experiment $(TOKEN_EXPERIMENT) --output-dir $(OUTPUTS_ROOT)/models/scibert_$(TOKEN_EXPERIMENT)

train-biobert-token-classifier:
	@$(CLI) train-token-classifier --config $(NER_BIOBERT_CONFIG) --data-root $(DATA_ROOT) --experiment $(TOKEN_EXPERIMENT) --output-dir $(OUTPUTS_ROOT)/models/biobert_$(TOKEN_EXPERIMENT)

predict-token-classifier:
	@$(CLI) predict-token-classifier --model $(TOKEN_MODEL_DIR) --articles $(ARTICLES) --output $(TOKEN_OUTPUT) --batch-size $(TOKEN_BATCH_SIZE) --max-length $(TOKEN_MAX_LENGTH)

run-re-baseline:
	@$(MAKE) predict-re-rule

predict-re-rule:
	@$(CLI) predict-re-rule --articles $(ARTICLES) --entities $(RE_ENTITIES) --train-relations $(TRAIN_RELATIONS) --output $(RE_RULE_OUTPUT) --threshold $(RE_RULE_THRESHOLD) $(if $(RE_RULE_MAX_DISTANCE),--max-distance $(RE_RULE_MAX_DISTANCE),)

run-re-transformer:
	@$(MAKE) predict-re-pair-classifier

train-re-pair-classifier:
	@$(CLI) train-re-pair-classifier --config $(RE_TRANSFORMER_CONFIG) --data-root $(DATA_ROOT) --experiment $(RE_PAIR_EXPERIMENT) $(RE_PAIR_MODEL_ARG) --output-dir $(RE_PAIR_MODEL_DIR)

predict-re-pair-classifier:
	@$(CLI) predict-re-pair-classifier --model $(RE_PAIR_MODEL_DIR) --articles $(ARTICLES) --entities $(RE_ENTITIES) --output $(RE_PAIR_OUTPUT) --threshold $(RE_PAIR_THRESHOLD) --batch-size $(RE_PAIR_BATCH_SIZE) --max-length $(RE_PAIR_MAX_LENGTH)

predict-re-ollama:
	@$(CLI) predict-re-ollama --articles $(ARTICLES) --entities $(RE_ENTITIES) --model $(OLLAMA_MODEL) --base-url $(OLLAMA_BASE_URL) --output $(OLLAMA_OUTPUT) --threshold $(OLLAMA_THRESHOLD) --max-candidates $(OLLAMA_MAX_CANDIDATES) --max-distance $(OLLAMA_MAX_DISTANCE) --timeout $(OLLAMA_TIMEOUT) --temperature $(OLLAMA_TEMPERATURE) --decisions-output $(OLLAMA_DECISIONS_OUTPUT)

predict-re-gpt:
	@$(CLI) predict-re-gpt --articles $(ARTICLES) --entities $(RE_ENTITIES) $(GPT_MODEL_ARG) --env-file $(GPT_ENV_FILE) --output $(GPT_OUTPUT) --threshold $(GPT_THRESHOLD) --max-candidates $(GPT_MAX_CANDIDATES) --max-distance $(GPT_MAX_DISTANCE) --timeout $(GPT_TIMEOUT) --temperature $(GPT_TEMPERATURE) --decisions-output $(GPT_DECISIONS_OUTPUT)

atlop-notes:
	@$(CLI) atlop-notes --official-repo $(OFFICIAL_REPO) --data-root $(DATA_ROOT) --output $(ATLOP_NOTES_OUTPUT)

prepare-atlop-data:
	@$(CLI) prepare-atlop-data --data-root $(DATA_ROOT) --official-repo $(OFFICIAL_REPO) $(ATLOP_PREDICTED_ENTITIES_ARG)

run-atlop:
	@$(CLI) run-atlop --official-repo $(OFFICIAL_REPO) --action $(ATLOP_ACTION) --output $(ATLOP_OUTPUT) --log $(ATLOP_LOG) $(ATLOP_DRY_RUN_ARG)

evaluate:
	@$(CLI) evaluate --task $(EVAL_TASK) --gold $(GOLD) --prediction $(PREDICTION) --output $(METRICS_OUTPUT)

evaluate-official:
	@$(CLI) evaluate-official --official-repo $(OFFICIAL_REPO) --prediction $(PREDICTION) $(OFFICIAL_ENTRYPOINT_ARG)

export-t611:
	@$(CLI) export-t611 --output $(T611_OUTPUT)

export-t621:
	@$(CLI) export-t621 --output $(T621_OUTPUT)

clean:
	@find src tests -name __pycache__ -type d -prune -exec rm -rf {} +
	@rm -rf .pytest_cache
