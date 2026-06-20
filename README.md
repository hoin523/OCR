# OCR Foundation

Experimental OCR foundation workspace for evaluating open OCR/VLM models and
building a fine-tuning pipeline.

## Local-Only Data

Keep private screenshots, receipts, bank records, and generated run outputs out
of git:

- `data/private/`
- `results/`

Both paths are ignored because this repository is public.

## Drop Files and Run

Create the private workspace:

```bash
make setup
```

Put receipt, invoice, or transaction images into:

```text
data/private/raw/
```

Then build the file inventory:

```bash
make inventory
```

Run a quick OCR smoke test on the first 10 images with PaddleOCR text:

```bash
make smoke
```

Run baseline extraction for all uploaded files:

```bash
make baselines
```

By default `make baselines` runs `paddleocr-text`, `paddleocr-vl`, and local `qwen3-vl`.
For Qwen, make sure Ollama has the model available:

```bash
ollama pull qwen3-vl:8b
```

Useful variants:

```bash
EXTRACTORS=paddleocr-text make baselines
EXTRACTORS=paddleocr-vl make baselines
EXTRACTORS=qwen3-vl make baselines
SMOKE_LIMIT=3 make smoke
make dry-run
make test
```

Generated outputs stay local under ignored paths:

- `results/dataset_inventory.json`
- `results/baselines/<image_id>/paddleocr-text.json`
- `results/baselines/<image_id>/paddleocr-vl.json`
- `results/baselines/<image_id>/qwen3-vl.json`
- `results/document_parse/<image_id>/layout.json`
- `results/document_parse/<image_id>/viewer.html`

See `docs/data_contract.md` for the private dataset and label format.

## Local Document Parse Layout MVP

Generate an Upstage-style clickable document parse viewer from local baseline
OCR outputs:

```bash
make inventory
make smoke
make layout
```

Open the generated viewer from:

```text
results/document_parse/<image_id>/viewer.html
```

For a one-shot local smoke run, use:

```bash
SMOKE_LIMIT=3 make layout-smoke
```

To upload a PDF or image from the browser and open the generated parse viewer
automatically:

```bash
make upload-app
```

Then open `http://127.0.0.1:8770`. Uploaded files stay local under
`data/private/uploads`, and generated viewers are written under
`results/document_parse_uploads/<job_id>/index.html`.
Completed uploads are also listed in the local approval-rule registry at
`data/private/approval_rules/registry.json`, which powers the "적용된 전결규정"
panel in the upload app.

### Lightweight Parser Test Source

For a minimal upload-to-parse test app without approval-rule screens, run:

```bash
make parser-test-app
```

Then open `http://127.0.0.1:8771`. This path keeps the source intentionally
small:

- `scripts/parser_test_app.py`: parser-test-only browser UI and upload API
- `scripts/upload_app.py`: shared upload job/pipeline helpers
- `scripts/preprocess_documents.py`: PDF/image to OCR-ready WebP pages
- `scripts/run_dataset_pipeline.py`: PaddleOCR text baseline runner
- `scripts/run_document_parse_layout.py`: clickable layout JSON/viewer writer
- `scripts/document_parse_layout.py`: layout grouping, fields, overlays, HTML

The lightweight app only exposes `/api/upload`, `/api/jobs/<job_id>`,
`/api/parse-results/<job_id>`, and `/results/...`. It always stores uploads as
`parser_test`, so test runs do not enter the approval-rule registry.

### Technology Used

- Python 3.12 and `uv` for local-only scripts and repeatable test runs.
- PaddleOCR text recognition with Korean language support for open OCR
  detection/recognition boxes.
- Pillow image handling and Poppler `pdftoppm` for PDF page rendering.
- WebP preprocessing for large images and PDF pages before OCR.
- Heuristic layout grouping that converts OCR line boxes into larger clickable
  elements, serialized text, and serialized Markdown.
- Static HTML viewers for single-page, multipage, and upload-driven parser
  review without an external backend service.
- Optional baseline slots for PaddleOCR-VL and local Ollama Qwen3-VL outputs
  when richer VLM extraction is available.

### Tested Locally

- KORIE receipt image smoke parsing with clickable text boxes and grouped layout
  elements.
- PDF preprocessing and multipage parse viewer generation.
- Browser upload flow for image/PDF inputs with local WebP preprocessing.
- Parser Test JSON comparison panel for checking expected fields against
  parsed `layout.json` fields.
- Automated pytest coverage for parser layout, multipage rendering, upload API,
  parser-test-only app behavior, preprocessing, and dataset inventory helpers.

The MVP uses PaddleOCR text boxes for clickable coordinates. PaddleOCR-VL and
Qwen3-VL baseline files are recorded as available sources when present.
For large mobile photos, PaddleOCR text detection defaults to
`TEXT_DET_LIMIT_SIDE_LEN=1536` and `TEXT_DET_LIMIT_TYPE=max` to keep local CPU
runs from exhausting memory. Increase the limit when you want denser boxes and
have enough RAM.

Large images and PDFs can be preprocessed into OCR-friendly WebP files:

```bash
RAW_DIR=data/private/raw/korie PREPROCESSED_DIR=data/private/preprocessed/korie make preprocess
RAW_DIR=data/private/preprocessed/korie SMOKE_LIMIT=3 make layout-smoke
```

The preprocessor keeps originals intact, resizes only when an image exceeds
`PREPROCESS_MAX_SIDE`, writes `.webp` outputs, and records
`preprocess_manifest.json`.

PDF preprocessing renders each page to WebP using Poppler `pdftoppm`. If
`pdftoppm` is not on your `PATH`, install Poppler locally or run the command
with a PATH that includes the Poppler binary directory.

## KORIE Receipt Samples

Clone the public KORIE receipt dataset into the ignored private workspace and
copy a few receipt images into the raw drop zone:

```bash
make korie-clone
make korie-sample
RAW_DIR=data/private/raw/korie SMOKE_LIMIT=3 make layout-smoke
```

All KORIE files and generated outputs remain local under ignored `data/private`
and `results` paths.

## Smoke Test Commands

Qwen3-VL via Ollama:

```bash
ollama pull qwen3-vl:8b
uv run --with requests python scripts/run_ollama_ocr.py \
  --model qwen3-vl:8b \
  --image data/private/input/receipt.png \
  --output results/model_runs/qwen3-vl-8b_receipt.json
```

PaddleOCR text baseline:

```bash
uv run --python 3.12 --with 'paddleocr>=3.6.0' --with paddlepaddle \
  python scripts/run_paddleocr_text.py \
  --image data/private/input/receipt.png \
  --output results/model_runs/paddleocr-text_receipt.json \
  --lang korean
```

PaddleOCR-VL was tested with `paddleocr[doc-parser]` and saved generated
Markdown/JSON under ignored `results/model_runs/`.
