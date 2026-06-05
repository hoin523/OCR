# OCR Foundation

Experimental OCR foundation workspace for evaluating open OCR/VLM models and
building a fine-tuning pipeline.

## Local-Only Data

Keep private screenshots, receipts, bank records, and generated run outputs out
of git:

- `data/private/`
- `results/`

Both paths are ignored because this repository is public.

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
