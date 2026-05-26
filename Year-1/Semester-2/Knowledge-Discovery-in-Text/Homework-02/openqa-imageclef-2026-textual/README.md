# ImageCLEF 2026 OpenQA Textual

Local project scaffold for the ImageCLEF-MR2026 OpenQA Textual task.

The inference-time input is an image containing a textual question:

```text
question image -> OCR/text extraction -> text normalization -> answer generation -> JSON submission
```

## Local Development

Use Python 3.10+ or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest
```

Optional OCR and GPU dependencies are deliberately separated because OCR stacks can conflict across platforms:

```bash
pip install -r requirements-ocr.txt
pip install -r requirements-gpu.txt
```

Tesseract also needs a system binary, for example:

```bash
sudo apt-get install tesseract-ocr
```

Copy `.env.example` to `.env` and adjust cache paths or Hugging Face tokens as needed.

## Layout

```text
configs/          YAML configuration files
data/             local data, OCR caches, reports, and submissions
experiments/      experiment logs and run metadata
notebooks/        exploratory notebooks
src/              Python package and command modules
tests/            smoke tests for the local setup
```

