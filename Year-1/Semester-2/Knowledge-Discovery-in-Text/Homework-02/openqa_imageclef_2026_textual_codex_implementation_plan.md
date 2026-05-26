# ImageCLEF 2026 OpenQA Textual — Image-to-Text QA Implementation Plan

This document is intended for a Codex/code agent that will implement an end-to-end system for the **Open Question Answering (OpenQA)** task using the **ImageCLEF-MR2026 OpenQA Textual dataset**.

## Critical clarification

In this project, **Textual does not mean that the system receives ready plain-text questions**.

The system works with **images only**. Each sample contains a question rendered as an image, and the first responsibility of the system is to extract the question text from that image. The extracted text is then used for answer generation.

The full pipeline is therefore:

```text
question image -> OCR/text extraction -> text normalization -> answer generation -> JSON submission
```

Do **not** design the implementation as a direct text-input QA system. Do **not** assume that the question text is available at inference time, except for train/dev annotations that may be used only for evaluation, debugging, OCR validation, or supervised training if officially provided.

Dataset URL:

```text
https://huggingface.co/datasets/SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual
```

The implementation should support GPU execution on Runpod for OCR models, LLM inference, embedding models, rerankers, and optional LoRA/QLoRA fine-tuning.

> Do not submit this implementation plan as the final homework report. The final report must be written manually and must describe the actual experiments and results.

---

## 1. Goal

Build a reproducible OpenQA system where the only inference-time input is an **image containing a textual question**.

The final system must:

1. Load the OpenQA Textual dataset from Hugging Face.
2. Read/decode the image field for each sample.
3. Extract the question text using OCR.
4. Normalize and validate the OCR result.
5. Generate one or more free-form textual answers.
6. Evaluate experiments on the development split.
7. Generate a valid OpenQA prediction JSON.
8. Zip the prediction file correctly.
9. Keep experiment logs for the final one-page system description.

The project should implement multiple increasingly strong approaches:

- OCR + simple heuristic baseline.
- OCR + prompted local LLM.
- OCR + retrieval-augmented generation over train examples.
- OCR + hybrid retrieval + reranking + LLM generation.
- Optional OCR post-correction.
- Optional LoRA/QLoRA fine-tuning of a compact LLM.
- Optional answer ensembling/selection.

---

## 2. Task Understanding

### 2.1 Selected task

The selected competition task is **Open Question Answering (OpenQA)**.

For each sample, the input is an image of a question without predefined answer options. The expected output is one or more generated textual answers.

There are no fixed choices. The system must produce the answer text directly.

### 2.2 Meaning of the Textual dataset

Use the **Textual** dataset, where the image content is expected to be text-only. This means:

- The system receives an image or image-like field.
- The first step is OCR/text extraction.
- The answer generation step is based on extracted question text.
- No diagram, chart, medical-image, or natural-image reasoning is expected.
- However, image preprocessing still matters because OCR quality directly affects answer quality.

Allowed input usage at inference time:

- image bytes / PIL image / image path from the dataset;
- `question_id` / `id` metadata;
- `language` metadata if available.

Allowed input usage during train/dev experimentation:

- gold answers from train/dev;
- official question text fields if present, but only for OCR quality checks, training experiments, and error analysis;
- train/dev metadata that is officially released.

Do not use:

- test labels;
- manual answer lookup;
- hardcoded answers for test IDs;
- hidden ground truth;
- direct text question fields at test time unless the official task explicitly allows them. The robust implementation should assume image-only inference.

### 2.3 Expected output format

The prediction file must be a valid JSON list. Each object should contain:

```json
{
  "question_id": "sample-id",
  "answers": ["generated answer"],
  "language": "English"
}
```

Rules:

- The number of prediction objects must match the test set size.
- There must be no duplicate `question_id` values.
- `answers` must be a list of strings.
- For a single-answer question, use one string in the list.
- Do not include explanations in `answers`.
- If a question is intentionally skipped, use `[""]`.
- The final JSON must be zipped before submission.
- The ZIP must contain no more than one JSON file.

---

## 3. Repository Structure

Create the following project:

```text
openqa-imageclef-2026-textual/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── ocr.yaml
│   ├── generation.yaml
│   ├── retrieval.yaml
│   ├── finetune.yaml
│   └── runpod.yaml
├── data/
│   ├── raw/                  # optional local dataset cache
│   ├── processed/
│   ├── ocr_cache/
│   ├── submissions/
│   └── reports/
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_ocr_error_analysis.ipynb
│   └── 03_metric_analysis.ipynb
├── src/
│   ├── openqa_textual/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── image_utils.py
│   │   ├── ocr.py
│   │   ├── ocr_postprocess.py
│   │   ├── prompts.py
│   │   ├── retrieval.py
│   │   ├── reranking.py
│   │   ├── generation.py
│   │   ├── finetuning.py
│   │   ├── prediction.py
│   │   ├── evaluation.py
│   │   ├── submission.py
│   │   └── logging_utils.py
│   └── scripts/
│       ├── inspect_dataset.py
│       ├── run_ocr.py
│       ├── build_retrieval_index.py
│       ├── predict.py
│       ├── evaluate_dev.py
│       ├── finetune_lora.py
│       ├── make_submission.py
│       └── validate_submission.py
├── experiments/
│   ├── experiment_log.md
│   └── runs/
└── tests/
    ├── test_data_loading.py
    ├── test_ocr_output.py
    ├── test_submission_format.py
    └── test_prediction_pipeline.py
```

---

## 4. Environment Setup

### 4.1 Local development

Use Python 3.10+ or 3.11.

Install core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
datasets
huggingface_hub
pandas
numpy
pillow
opencv-python
scikit-image
scikit-learn
rapidfuzz
tqdm
pyyaml
jsonschema
transformers
accelerate
bitsandbytes
peft
trl
torch
sentence-transformers
faiss-cpu
rank-bm25
nltk
rouge-score
sacrebleu
bert-score
pytesseract
easyocr
paddleocr
paddlepaddle-gpu
python-dotenv
rich
pytest
```

Keep OCR dependencies modular. Some OCR engines are difficult to install together, so implement clean wrappers and enable/disable them from config.

### 4.2 Runpod GPU environment

Prepare a Runpod template for:

- GPU OCR models;
- local LLM inference;
- embedding/reranking models;
- optional LoRA/QLoRA fine-tuning.

Recommended Runpod setup:

```bash
apt-get update
apt-get install -y git git-lfs unzip tesseract-ocr libgl1 libglib2.0-0
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
huggingface-cli login
```

Recommended GPU options:

- RTX 4090 / L40S for inference and OCR experiments;
- A100 40GB/80GB if fine-tuning larger models;
- persistent volume for model and OCR cache.

Use `.env`:

```text
HF_TOKEN=...
DATASET_NAME=SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual
MODEL_CACHE_DIR=/workspace/models
DATA_CACHE_DIR=/workspace/data
OCR_CACHE_DIR=/workspace/data/ocr_cache
```

---

## 5. Implementation Phase 1 — Dataset Inspection

### 5.1 Load the dataset

Implement `src/openqa_textual/data.py`.

Required functions:

```python
def load_dataset_splits(dataset_name: str, cache_dir: str | None = None):
    """Load train/dev/test splits from Hugging Face."""


def get_sample_id(sample: dict) -> str:
    """Return question_id/id from the sample."""


def get_sample_language(sample: dict) -> str:
    """Return language value or a safe default."""


def get_sample_image(sample: dict):
    """Return a PIL image decoded from the dataset sample."""
```

The code must inspect available columns and not assume a fixed schema. Possible image formats:

- Hugging Face `Image` feature returning a PIL image;
- bytes;
- path;
- nested dict with `bytes` or `path`.

### 5.2 Inspect fields

Create `src/scripts/inspect_dataset.py`.

It should print:

- split names;
- number of rows per split;
- column names;
- feature types;
- sample IDs;
- image sizes;
- language distribution if available;
- answer field availability for train/dev;
- whether a question text field exists.

Even if a question text field exists, the prediction pipeline must still use the image as the source of the question.

### 5.3 Save debug image samples

Implement:

```bash
python -m src.scripts.inspect_dataset --save-samples data/processed/debug_images --n 30
```

Save sample images to verify:

- resolution;
- font size;
- background;
- rotation/skew;
- language/script;
- artifacts/noise;
- whether all samples are truly text-only.

---

## 6. Implementation Phase 2 — Image Preprocessing for OCR

Implement `src/openqa_textual/image_utils.py`.

The objective is to improve OCR accuracy, not to perform visual reasoning.

Required preprocessing functions:

```python
def to_rgb(image): ...
def resize_for_ocr(image, min_width=1000): ...
def deskew(image): ...
def denoise(image): ...
def increase_contrast(image): ...
def binarize(image): ...
def pad_image(image, padding=20): ...
def preprocess_for_ocr(image, config): ...
```

Recommended preprocessing variants:

1. `raw`: no preprocessing.
2. `resize_only`: upscale small images.
3. `contrast`: grayscale + contrast enhancement.
4. `binarized`: thresholding for clean text.
5. `deskewed`: skew correction if needed.
6. `ensemble`: run OCR on several variants and select the best text.

Save preprocessed debug outputs for 30 train/dev examples.

Acceptance criteria:

- The OCR module can run on a PIL image.
- The preprocessing pipeline is configurable.
- OCR failure does not crash prediction; it returns an empty or fallback string.

---

## 7. Implementation Phase 3 — OCR/Text Extraction

Implement `src/openqa_textual/ocr.py`.

### 7.1 OCR engine interface

Create a common interface:

```python
class OCRResult:
    text: str
    confidence: float | None
    engine: str
    metadata: dict

class OCREngine:
    def extract(self, image) -> OCRResult:
        raise NotImplementedError
```

### 7.2 OCR engines to implement

Implement wrappers for at least two OCR engines:

#### Engine A — EasyOCR baseline

Useful for multilingual OCR.

```python
class EasyOCREngine(OCREngine):
    def __init__(self, languages: list[str], gpu: bool = True): ...
```

#### Engine B — PaddleOCR / PP-OCR

Often strong on printed text.

```python
class PaddleOCREngine(OCREngine):
    def __init__(self, lang: str, use_gpu: bool = True): ...
```

#### Engine C — Tesseract fallback

Useful as a lightweight CPU fallback.

```python
class TesseractOCREngine(OCREngine): ...
```

Optional advanced OCR:

- TrOCR;
- Nougat-style text extraction if the images contain math-heavy text;
- Donut-style OCR if layout is unusual.

### 7.3 OCR language handling

Implement language-to-OCR mapping.

Examples:

```python
LANGUAGE_TO_EASYOCR = {
    "English": ["en"],
    "Bulgarian": ["bg", "en"],
    "German": ["de", "en"],
    "French": ["fr", "en"],
    "Spanish": ["es", "en"],
}
```

If language is missing, run multilingual OCR or default to English + likely dataset languages.

### 7.4 OCR caching

OCR is expensive. Cache all extracted text.

Implement cache format:

```json
{
  "question_id": "...",
  "language": "English",
  "ocr_engine": "easyocr",
  "preprocess_variant": "contrast",
  "ocr_text": "What is the capital of France?",
  "confidence": 0.91,
  "created_at": "..."
}
```

Cache path:

```text
data/ocr_cache/{split}/{engine}/{preprocess_variant}/{question_id}.json
```

### 7.5 OCR ensemble selection

Implement OCR selection logic:

```python
def select_best_ocr_result(results: list[OCRResult]) -> OCRResult:
    """Choose the best OCR output from multiple engines/preprocessing variants."""
```

Selection heuristics:

- prefer non-empty text;
- prefer higher confidence;
- penalize extremely short outputs;
- penalize outputs with too many replacement/unknown characters;
- prefer outputs ending with `?` when expected;
- prefer outputs with plausible word/token count;
- compare candidates by normalized edit similarity if multiple outputs agree.

---

## 8. Implementation Phase 4 — OCR Postprocessing

Implement `src/openqa_textual/ocr_postprocess.py`.

Required functions:

```python
def normalize_whitespace(text: str) -> str: ...
def fix_common_ocr_errors(text: str, language: str | None = None) -> str: ...
def normalize_quotes_and_symbols(text: str) -> str: ...
def restore_question_mark(text: str) -> str: ...
def clean_ocr_question(text: str, language: str | None = None) -> str: ...
```

Handle common OCR issues:

- duplicated spaces/newlines;
- broken hyphenation;
- confused characters: `0/O`, `1/l/I`, `rn/m` where safe;
- missing punctuation;
- math symbols where possible;
- leading/trailing artifacts;
- repeated headers/footers if present.

Do not overcorrect domain-specific terms.

### 8.1 Optional LLM-based OCR correction

For hard examples, add optional OCR correction using a small local LLM:

Input:

```text
The following text was extracted from an image of a question using OCR.
Correct OCR mistakes while preserving the original meaning.
Return only the corrected question.

OCR text:
{ocr_text}
```

Use only for experiments where it improves dev performance.

---

## 9. Implementation Phase 5 — Baseline Answering

Implement `src/openqa_textual/generation.py` and `src/openqa_textual/prediction.py`.

### 9.1 Baseline 0 — OCR-only diagnostic

This baseline does not try to answer. It only saves OCR outputs for dev analysis.

Output file:

```text
data/reports/dev_ocr_outputs.jsonl
```

Fields:

```json
{"question_id": "...", "language": "...", "ocr_text": "...", "gold_answer": "..."}
```

Use this to manually inspect OCR quality.

### 9.2 Baseline 1 — Heuristic QA

Implement simple rules for common factual/math questions:

- direct arithmetic if OCR text contains a simple expression;
- yes/no question detection;
- fallback empty answer.

This is mainly a smoke test to validate the pipeline.

### 9.3 Baseline 2 — OCR + prompted LLM

Use a local instruction model.

Recommended models:

- `Qwen/Qwen2.5-7B-Instruct`;
- `Qwen/Qwen2.5-14B-Instruct` if GPU allows;
- `meta-llama/Llama-3.1-8B-Instruct` if available;
- `mistralai/Mistral-7B-Instruct-v0.3`;
- smaller fallback: `Qwen/Qwen2.5-3B-Instruct`.

Prompt template:

```text
You are answering an exam-style open question.
The question was extracted from an image using OCR, so it may contain minor OCR mistakes.
Answer with only the final answer. Do not explain.

Language: {language}
Question: {question}

Final answer:
```

Generation settings:

```yaml
temperature: 0.0
do_sample: false
max_new_tokens: 64
num_beams: 1
```

For longer explanatory answers, test `max_new_tokens: 128`, but keep final answers concise.

---

## 10. Implementation Phase 6 — Retrieval-Augmented Generation

Even though the system input is image-only, after OCR the question becomes text. Use train examples as a retrieval memory.

Implement `src/openqa_textual/retrieval.py`.

### 10.1 Build train index

For every train example:

1. Load image.
2. Run OCR or use cached OCR.
3. Clean OCR text.
4. Store question text + answer + metadata.

Index fields:

```json
{
  "question_id": "...",
  "language": "English",
  "ocr_question": "...",
  "gold_answer": "..."
}
```

### 10.2 Retrieval methods

Implement:

- BM25 over OCR question text;
- sentence-transformer dense embeddings;
- hybrid retrieval combining BM25 and dense scores.

Recommended embedding models:

- `intfloat/multilingual-e5-base`;
- `intfloat/multilingual-e5-large`;
- `BAAI/bge-m3`;
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.

### 10.3 Reranking

Implement optional reranking:

- `BAAI/bge-reranker-base`;
- multilingual reranker if needed.

### 10.4 RAG prompt

Use retrieved examples as few-shot context:

```text
You are answering exam-style open questions extracted from images by OCR.
The OCR text may contain minor errors.
Use the examples only as guidance. Do not copy an answer unless the question is equivalent.
Return only the final answer.

Examples:
Q: {retrieved_q1}
A: {retrieved_a1}

Q: {retrieved_q2}
A: {retrieved_a2}

Current question language: {language}
Current OCR question: {question}

Final answer:
```

Experiment with `k = 1, 3, 5`.

---

## 11. Implementation Phase 7 — Optional Fine-Tuning

Only fine-tune after establishing OCR + LLM and OCR + RAG baselines.

### 11.1 Training data construction

Use train split only.

Each training example should be built from OCR output, not from directly available question text, because test-time inference starts from images.

Training record:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Answer exam-style open questions extracted from images by OCR. Return only the answer."
    },
    {
      "role": "user",
      "content": "Language: English\nQuestion: {ocr_question}"
    },
    {
      "role": "assistant",
      "content": "{gold_answer}"
    }
  ]
}
```

If the dataset provides official clean question text, optionally create two training variants:

- `ocr_question -> answer` for realistic inference;
- `clean_question -> answer` for upper-bound comparison only.

Do not report the clean-question variant as the main system unless the official test input also provides clean text.

### 11.2 QLoRA setup

Recommended base models:

- `Qwen/Qwen2.5-7B-Instruct`;
- `Qwen/Qwen2.5-14B-Instruct` if GPU allows;
- `mistralai/Mistral-7B-Instruct-v0.3`.

Suggested settings:

```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 2e-4
num_train_epochs: 2-4
per_device_train_batch_size: 1-4
gradient_accumulation_steps: 8
max_seq_length: 1024
```

### 11.3 Evaluation after fine-tuning

Compare:

- OCR + base LLM;
- OCR + RAG + base LLM;
- OCR + fine-tuned LLM;
- OCR + RAG + fine-tuned LLM.

Track whether fine-tuning overfits to train style.

---

## 12. Implementation Phase 8 — Prediction Pipeline

Implement a single end-to-end prediction script:

```bash
python -m src.scripts.predict \
  --split dev \
  --config configs/generation.yaml \
  --ocr-config configs/ocr.yaml \
  --output data/processed/dev_predictions.json
```

The script must perform:

1. Load split.
2. For each sample, read image.
3. Run OCR or load OCR cache.
4. Clean OCR question.
5. Generate answer.
6. Postprocess answer.
7. Save prediction object.

Prediction object:

```json
{
  "question_id": "...",
  "answers": ["..."],
  "language": "English",
  "debug": {
    "ocr_text": "...",
    "clean_question": "...",
    "ocr_engine": "...",
    "model": "..."
  }
}
```

The debug field is allowed only in internal files. It must be removed from the final submission.

---

## 13. Implementation Phase 9 — Answer Postprocessing

Implement:

```python
def clean_answer(answer: str, language: str | None = None) -> str:
    ...
```

Rules:

- strip whitespace;
- remove prefixes like `Final answer:`, `Answer:`, `A:`;
- remove explanations if model returns them;
- keep only first line if the task expects a short answer;
- preserve mathematical notation when needed;
- normalize empty outputs to `""`;
- ensure the final value is always a string.

Optional answer normalization:

- convert obvious number words to digits only if safe;
- normalize decimal separators depending on language;
- deduplicate repeated answer phrases.

---

## 14. Implementation Phase 10 — Evaluation

Implement `src/openqa_textual/evaluation.py`.

Use the official evaluation script if available in the baseline repository. Also implement local quick metrics:

- exact match after normalization;
- token F1;
- BLEU;
- ROUGE-L;
- METEOR if available;
- BERTScore optional;
- OCR character error rate if clean question text is available for dev.

Command:

```bash
python -m src.scripts.evaluate_dev \
  --pred data/processed/dev_predictions.json \
  --gold data/processed/dev_gold.json \
  --output data/reports/dev_metrics.json
```

The evaluation report should include:

```json
{
  "experiment_name": "ocr_easyocr_qwen7b_rag_k3",
  "ocr_engine": "easyocr",
  "preprocessing": "contrast",
  "generation_model": "Qwen2.5-7B-Instruct",
  "retrieval": "hybrid_bm25_e5",
  "metrics": {
    "exact_match": 0.0,
    "token_f1": 0.0,
    "bleu": 0.0,
    "rouge_l": 0.0
  },
  "notes": "..."
}
```

---

## 15. Implementation Phase 11 — Submission Generation

Implement `src/openqa_textual/submission.py`.

### 15.1 Create submission JSON

Command:

```bash
python -m src.scripts.make_submission \
  --pred data/processed/test_predictions.json \
  --output data/submissions/predictions.json
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

No debug fields.

### 15.2 Validate submission

Command:

```bash
python -m src.scripts.validate_submission \
  --submission data/submissions/predictions.json \
  --expected-size-from-split test
```

Validation checks:

- valid JSON;
- top-level list;
- exactly one object per test sample;
- no duplicate IDs;
- each item has `question_id`, `answers`, `language`;
- `answers` is a list;
- each answer is a string;
- no debug fields;
- no accidental nulls;
- no extra JSON files in ZIP.

### 15.3 Zip submission

Command:

```bash
cd data/submissions
zip openqa_textual_submission.zip predictions.json
```

Validate ZIP:

```bash
unzip -l openqa_textual_submission.zip
```

The ZIP must contain exactly one JSON file.

---

## 16. Experiments to Run

Record every experiment in `experiments/experiment_log.md`.

Suggested experiment table:

```markdown
| ID | OCR | Preprocess | OCR correction | Retrieval | LLM | Fine-tuned | Dev score | Notes |
|----|-----|------------|----------------|-----------|-----|------------|-----------|-------|
| E00 | easyocr | raw | no | none | none | no | - | OCR diagnostic |
| E01 | easyocr | contrast | no | none | Qwen2.5-7B | no | TBD | basic prompted LLM |
| E02 | paddleocr | contrast | no | none | Qwen2.5-7B | no | TBD | compare OCR |
| E03 | best-of-ocr | ensemble | no | BM25 k=3 | Qwen2.5-7B | no | TBD | RAG baseline |
| E04 | best-of-ocr | ensemble | no | hybrid + rerank k=3 | Qwen2.5-7B | no | TBD | stronger RAG |
| E05 | best-of-ocr | ensemble | yes | hybrid + rerank k=3 | Qwen2.5-7B | no | TBD | OCR correction |
| E06 | best-of-ocr | ensemble | no | none | Qwen2.5-7B LoRA | yes | TBD | fine-tuned |
| E07 | best-of-ocr | ensemble | no | hybrid + rerank k=3 | Qwen2.5-7B LoRA | yes | TBD | final candidate |
```

Minimum experiments for the homework:

1. OCR diagnostic.
2. OCR + prompted LLM.
3. OCR + RAG + LLM.
4. Best final model submitted to leaderboard.

---

## 17. Error Analysis

Create `notebooks/02_ocr_error_analysis.ipynb` or a script producing `data/reports/error_analysis.md`.

Analyze:

- OCR failures;
- language-specific OCR problems;
- math symbol errors;
- missing punctuation;
- malformed questions;
- LLM hallucinations caused by OCR noise;
- cases where retrieval helps;
- cases where retrieval hurts;
- too-long answers;
- empty answers;
- repeated answers.

For each failed dev sample, save:

```json
{
  "question_id": "...",
  "language": "...",
  "ocr_text": "...",
  "clean_question": "...",
  "prediction": "...",
  "gold": "...",
  "error_type": "ocr_error | reasoning_error | formatting_error | retrieval_error | unknown"
}
```

---

## 18. Final System Description Draft

The homework requires a short description of the system and results. Keep it manually written and concise.

Draft structure:

```markdown
# System Description

## Approach
I implemented an image-to-text OpenQA pipeline for the ImageCLEF MultimodalReasoning OpenQA Textual task. Since the input samples are images containing textual questions, the first stage extracts the question text with OCR. The extracted text is normalized and then passed to a local instruction-tuned language model for answer generation. I also tested a retrieval-augmented variant that retrieves similar training questions and uses them as few-shot examples.

## Processing Steps
1. Load the dataset image for each sample.
2. Apply OCR preprocessing such as resizing and contrast enhancement.
3. Extract question text using [OCR engine].
4. Normalize OCR text.
5. Generate the final answer using [model].
6. Postprocess the answer to match the required JSON format.

## Experiments
I compared [experiment A], [experiment B], and [experiment C]. The best development result was obtained with [best approach].

## Results
The submitted run achieved [score] on the official leaderboard.

## Discussion
The main source of errors was [OCR / reasoning / formatting]. Retrieval helped in cases where similar training questions existed, but it sometimes introduced incorrect answers for superficially similar questions.
```

Do not submit the full generated implementation plan as the final one-page report.

---

## 19. Codex Agent Task Checklist

Give the Codex agent these tasks in order.

### Milestone 1 — Project skeleton

- Create repository structure.
- Add `requirements.txt`, `pyproject.toml`, `.env.example`.
- Add config files.
- Add logging utilities.
- Add basic tests.

### Milestone 2 — Dataset loader

- Implement Hugging Face dataset loading.
- Implement image extraction from samples.
- Implement ID/language extraction.
- Add dataset inspection script.
- Save debug images.

### Milestone 3 — OCR pipeline

- Implement image preprocessing.
- Implement EasyOCR wrapper.
- Implement PaddleOCR or Tesseract wrapper.
- Implement OCR cache.
- Implement OCR ensemble selection.
- Implement OCR diagnostic export.

### Milestone 4 — Text cleanup

- Implement OCR text normalization.
- Add common OCR error fixes.
- Add optional LLM correction.
- Add tests for OCR postprocessing.

### Milestone 5 — Basic answer generation

- Implement prompted LLM generation.
- Add model loading with quantization.
- Add answer postprocessing.
- Add dev prediction script.

### Milestone 6 — Retrieval/RAG

- Build train OCR index.
- Implement BM25 retrieval.
- Implement dense retrieval.
- Implement hybrid retrieval.
- Add optional reranking.
- Add RAG prompt.

### Milestone 7 — Evaluation

- Implement local dev metrics.
- Integrate official evaluation script if available.
- Save experiment metrics.
- Add error analysis export.

### Milestone 8 — Optional fine-tuning

- Build instruction dataset from OCR questions and gold answers.
- Implement QLoRA training.
- Evaluate fine-tuned model on dev.
- Compare with non-fine-tuned baselines.

### Milestone 9 — Submission

- Implement test prediction.
- Strip debug fields.
- Validate JSON.
- Zip exactly one JSON file.
- Save final run metadata.

---

## 20. Definition of Done

The implementation is complete when:

- The dataset can be loaded from Hugging Face.
- Every sample image can be decoded.
- OCR can be run and cached for train/dev/test.
- The system can generate predictions from images only.
- The dev split can be evaluated.
- At least three experiments are logged.
- The final test prediction JSON validates successfully.
- The ZIP contains exactly one JSON file.
- The final approach is summarized in a short manually edited report.

