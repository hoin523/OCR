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

See `docs/data_contract.md` for the private dataset and label format.

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
