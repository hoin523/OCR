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

The MVP uses PaddleOCR text boxes for clickable coordinates. PaddleOCR-VL and
Qwen3-VL baseline files are recorded as available sources when present.
For large mobile photos, PaddleOCR text detection defaults to
`TEXT_DET_LIMIT_SIDE_LEN=1536` and `TEXT_DET_LIMIT_TYPE=max` to keep local CPU
runs from exhausting memory. Increase the limit when you want denser boxes and
have enough RAM.

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
