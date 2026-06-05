# On-Prem OCR Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an on-premises receipt and invoice OCR pipeline that can be fine-tuned on approximately 1,000 private documents and deployed without external API calls.

**Architecture:** Use a staged OCR pipeline instead of one monolithic model: image/PDF ingestion, preprocessing, OCR text/bbox extraction, field-level JSON extraction, validation, review, fine-tuning, and on-prem serving. Start with PaddleOCR/PP-OCRv5 for OCR and Qwen3-VL as a local teacher/extractor, then distill or fine-tune smaller domain-specific components as the private dataset grows.

**Tech Stack:** Python 3.12, uv, PaddleOCR/PP-OCRv5, PaddleOCR-VL, Ollama Qwen3-VL, JSON/JSONL labels, pytest, Docker, local ignored `data/private/` and `results/` directories.

---

## Assumptions

- The repository is public, so all private images, labels, and generated results must stay under ignored paths.
- The user will later provide about 1,000 receipt, invoice, bank transaction, or payment screenshots.
- External API calls are not allowed for production because the target environment is on-premises.
- Local teacher inference is allowed during development if the model runs on private infrastructure.
- The first production target is field extraction, not a general OCR foundation model.

## Target Output Schema

The initial extractor should produce one JSON object per document:

```json
{
  "document_type": "receipt",
  "merchant": "Sample Store",
  "invoice_no": null,
  "order_id": "ORDER123",
  "datetime": "2026-05-14 17:40",
  "currency": "KRW",
  "subtotal": 14400,
  "tax": null,
  "delivery_fee": 0,
  "discount": 0,
  "total": 14400,
  "payment_method": "Sample Bank",
  "account_tail": "2628",
  "line_items": [
    {
      "name": "Sample Item",
      "quantity": null,
      "unit_price": null,
      "amount": 6800
    }
  ],
  "raw_text_lines": [
    "Receipt",
    "Sample Store",
    "Sample Item"
  ],
  "needs_review": false,
  "review_reasons": []
}
```

## Expected Work Sequence

### Task 1: Prepare Private Dataset Layout

**Files:**
- Create: `docs/data_contract.md`
- Private data location: `data/private/raw/`
- Private labels location: `data/private/labels/`

- [ ] **Step 1: Create private folders**

```bash
mkdir -p data/private/raw data/private/labels/raw_text data/private/labels/fields data/private/labels/boxes
```

Expected: directories exist locally and remain ignored by git.

- [ ] **Step 2: Define naming convention**

Create `docs/data_contract.md` with this structure:

```markdown
# OCR Dataset Contract

Private files are never committed.

## Image Naming

- `receipt_000001.png`
- `invoice_000001.png`
- `transaction_000001.png`

## Label Files

For each image:

- `data/private/labels/raw_text/<image_id>.txt`
- `data/private/labels/fields/<image_id>.json`
- `data/private/labels/boxes/<image_id>.json`

## Required Fields

See `docs/superpowers/plans/2026-06-06-on-prem-ocr-finetuning.md`.
```

- [ ] **Step 3: Verify ignored data**

Run:

```bash
git status --short --ignored
```

Expected: private data appears with `!! data/`, not `?? data/`.

- [ ] **Step 4: Commit docs only**

```bash
git add docs/data_contract.md
git commit -m "docs: add OCR dataset contract"
```

### Task 2: Build Dataset Inventory

**Files:**
- Create: `scripts/build_dataset_inventory.py`
- Create: `tests/test_dataset_inventory.py`
- Output: `results/dataset_inventory.json`

- [ ] **Step 1: Write tests for file discovery**

Create `tests/test_dataset_inventory.py`:

```python
from pathlib import Path

from scripts.build_dataset_inventory import build_inventory


def test_build_inventory_discovers_supported_images(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "receipt_000001.png").write_bytes(b"fake")
    (raw / "invoice_000001.jpg").write_bytes(b"fake")
    (raw / "notes.txt").write_text("skip", encoding="utf-8")

    inventory = build_inventory(raw)

    assert [item["image_id"] for item in inventory] == [
        "invoice_000001",
        "receipt_000001",
    ]
    assert inventory[0]["document_hint"] == "invoice"
    assert inventory[1]["document_hint"] == "receipt"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --with pytest pytest tests/test_dataset_inventory.py -v
```

Expected: FAIL because `scripts.build_dataset_inventory` does not exist yet.

- [ ] **Step 3: Implement inventory builder**

Create `scripts/build_dataset_inventory.py`:

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def infer_document_hint(image_id: str) -> str:
    prefix = image_id.split("_", 1)[0].lower()
    if prefix in {"receipt", "invoice", "transaction"}:
        return prefix
    return "unknown"


def build_inventory(raw_dir: Path) -> list[dict[str, str]]:
    items = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        image_id = path.stem
        items.append(
            {
                "image_id": image_id,
                "path": str(path),
                "document_hint": infer_document_hint(image_id),
            }
        )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/private/raw", type=Path)
    parser.add_argument("--output", default="results/dataset_inventory.json", type=Path)
    args = parser.parse_args()

    inventory = build_inventory(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(inventory)} items to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
uv run --with pytest pytest tests/test_dataset_inventory.py -v
```

Expected: PASS.

- [ ] **Step 5: Generate real inventory**

```bash
uv run python scripts/build_dataset_inventory.py \
  --raw-dir data/private/raw \
  --output results/dataset_inventory.json
```

Expected: `results/dataset_inventory.json` contains about 1,000 entries after the user provides files.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_dataset_inventory.py tests/test_dataset_inventory.py
git commit -m "feat: add private dataset inventory builder"
```

### Task 3: Generate Baseline OCR and Teacher Labels

**Files:**
- Modify: `scripts/run_ollama_ocr.py`
- Modify: `scripts/run_paddleocr_text.py`
- Create: `scripts/batch_extract_baselines.py`
- Output: `results/baselines/<image_id>/`

- [ ] **Step 1: Add batch extraction script**

Create `scripts/batch_extract_baselines.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="results/dataset_inventory.json", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    items = json.loads(args.inventory.read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    for item in items:
        image_id = item["image_id"]
        image_path = item["path"]
        output_dir = Path("results/baselines") / image_id
        output_dir.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                "uv",
                "run",
                "--with",
                "requests",
                "python",
                "scripts/run_ollama_ocr.py",
                "--model",
                "qwen3-vl:8b",
                "--image",
                image_path,
                "--output",
                str(output_dir / "qwen3-vl.json"),
            ]
        )

        run_command(
            [
                "uv",
                "run",
                "--python",
                "3.12",
                "--with",
                "paddleocr>=3.6.0",
                "--with",
                "paddlepaddle",
                "python",
                "scripts/run_paddleocr_text.py",
                "--image",
                image_path,
                "--output",
                str(output_dir / "paddleocr-text.json"),
                "--lang",
                "korean",
            ]
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a 10-image smoke test**

```bash
uv run python scripts/batch_extract_baselines.py --limit 10
```

Expected: `results/baselines/<image_id>/qwen3-vl.json` and `paddleocr-text.json` are created.

- [ ] **Step 3: Run full baseline extraction**

```bash
uv run python scripts/batch_extract_baselines.py
```

Expected: all private images have local OCR/teacher outputs.

- [ ] **Step 4: Commit batch script**

```bash
git add scripts/batch_extract_baselines.py
git commit -m "feat: add batch OCR baseline extraction"
```

### Task 4: Build Label Normalization and Review Queue

**Files:**
- Create: `scripts/normalize_teacher_labels.py`
- Create: `tests/test_normalize_teacher_labels.py`
- Output: `data/private/labels/fields/*.json`
- Output: `results/review_queue.json`

- [ ] **Step 1: Write normalization tests**

Create `tests/test_normalize_teacher_labels.py`:

```python
from scripts.normalize_teacher_labels import normalize_amount, mask_account_tail


def test_normalize_amount_handles_won_symbol_errors():
    assert normalize_amount("W14,400") == 14400
    assert normalize_amount("₩14,400") == 14400
    assert normalize_amount("-14,400원") == -14400
    assert normalize_amount("WO") == 0


def test_mask_account_tail_returns_last_four_digits():
    assert mask_account_tail("Sample Bank 1234567890123456") == "3456"
    assert mask_account_tail("Sample Bank **********3456") == "3456"
    assert mask_account_tail(None) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run --with pytest pytest tests/test_normalize_teacher_labels.py -v
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement normalization helpers**

Create `scripts/normalize_teacher_labels.py`:

```python
#!/usr/bin/env python3
import re
from typing import Any


def normalize_amount(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in {"WO", "W0", "₩0"}:
        return 0
    sign = -1 if text.startswith("-") else 1
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        return None
    return sign * int(digits)


def mask_account_tail(value: Any) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"[^0-9]", "", str(value))
    if len(digits) < 4:
        return None
    return digits[-4:]
```

- [ ] **Step 4: Run tests**

```bash
uv run --with pytest pytest tests/test_normalize_teacher_labels.py -v
```

Expected: PASS.

- [ ] **Step 5: Extend script to write normalized field labels**

Add a CLI to `scripts/normalize_teacher_labels.py` that reads `results/baselines/*/qwen3-vl.json`, normalizes amount fields, masks account tails, marks low-confidence issues in `needs_review`, and writes `data/private/labels/fields/<image_id>.json`.

- [ ] **Step 6: Commit**

```bash
git add scripts/normalize_teacher_labels.py tests/test_normalize_teacher_labels.py
git commit -m "feat: normalize OCR teacher labels"
```

### Task 5: Create Gold Evaluation Split

**Files:**
- Create: `scripts/create_eval_split.py`
- Output: `data/private/splits/train.jsonl`
- Output: `data/private/splits/valid.jsonl`
- Output: `data/private/splits/test.jsonl`

- [ ] **Step 1: Choose split policy**

Use this default:

```text
train: 80%
valid: 10%
test: 10%
```

The `test` split must be manually verified and never used for training.

- [ ] **Step 2: Implement split script**

Create `scripts/create_eval_split.py` that reads normalized field labels, shuffles with seed `20260606`, and writes JSONL split files.

- [ ] **Step 3: Generate splits**

```bash
uv run python scripts/create_eval_split.py \
  --labels-dir data/private/labels/fields \
  --output-dir data/private/splits \
  --seed 20260606
```

Expected: train/valid/test JSONL files exist under ignored private data.

- [ ] **Step 4: Commit**

```bash
git add scripts/create_eval_split.py
git commit -m "feat: add OCR evaluation split builder"
```

### Task 6: Evaluate Baselines Before Training

**Files:**
- Create: `scripts/evaluate_extraction.py`
- Create: `tests/test_evaluate_extraction.py`
- Output: `results/evaluation/baseline_metrics.json`

- [ ] **Step 1: Define metrics**

Use these field-level metrics:

```text
merchant_exact
datetime_exact
total_exact
currency_exact
account_tail_exact
line_item_amount_exact
json_validity
needs_review_rate
```

- [ ] **Step 2: Implement evaluator**

Create `scripts/evaluate_extraction.py` with pure functions that compare prediction JSON to gold JSON and compute exact-match scores per field.

- [ ] **Step 3: Run baseline evaluation**

```bash
uv run python scripts/evaluate_extraction.py \
  --gold data/private/splits/test.jsonl \
  --predictions results/baselines \
  --output results/evaluation/baseline_metrics.json
```

Expected: metrics file shows baseline field accuracy before fine-tuning.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate_extraction.py tests/test_evaluate_extraction.py
git commit -m "feat: add OCR extraction evaluator"
```

### Task 7: Fine-Tune OCR Recognition First

**Files:**
- Create: `training/paddleocr_recognition/README.md`
- Create: `training/paddleocr_recognition/prepare_rec_dataset.py`
- Private output: `data/private/paddleocr_rec/`

- [ ] **Step 1: Extract line crops**

Use PaddleOCR text boxes to crop text lines into:

```text
data/private/paddleocr_rec/images/
data/private/paddleocr_rec/labels.txt
```

Each label row should use:

```text
relative/path/to/crop.png<TAB>exact text
```

- [ ] **Step 2: Prioritize known errors**

Oversample lines containing:

```text
₩
원
0
O
사업자
합계
부가세
Total
```

- [ ] **Step 3: Fine-tune PP-OCR recognition model**

Run PaddleOCR recognition fine-tuning from the official PaddleOCR training workflow, using the generated line crop dataset and a Korean/multilingual pretrained recognizer.

- [ ] **Step 4: Evaluate before/after**

Run OCR line-level evaluation and confirm improvements on:

```text
₩ -> W
₩0 -> WO
0/O confusion in order IDs
```

- [ ] **Step 5: Commit training docs and dataset prep script**

```bash
git add training/paddleocr_recognition
git commit -m "docs: add PaddleOCR recognition fine-tuning plan"
```

### Task 8: Fine-Tune or Distill Field Extractor

**Files:**
- Create: `training/field_extractor/README.md`
- Create: `training/field_extractor/export_qwen_lora_dataset.py`
- Private output: `data/private/qwen_lora/train.jsonl`

- [ ] **Step 1: Export Qwen-style SFT data**

Convert each verified image and field label into a conversation example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data/private/raw/receipt_000001.png"},
        {"type": "text", "text": "Extract receipt or invoice fields as strict JSON. Return only JSON."}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"document_type\":\"receipt\",\"merchant\":\"...\",\"total\":14400}"
    }
  ]
}
```

- [ ] **Step 2: Start with LoRA**

Use LoRA/QLoRA only. Do not full fine-tune Qwen3-VL for the first pass.

- [ ] **Step 3: Train on CUDA infrastructure**

Use an on-prem NVIDIA GPU server when available. The current M4 Pro machine can prepare data and run inference, but it is not the target machine for Qwen3-VL fine-tuning.

- [ ] **Step 4: Evaluate on gold test split**

Compare trained model against the baseline Qwen3-VL zero-shot results using `scripts/evaluate_extraction.py`.

- [ ] **Step 5: Commit exporter and training notes**

```bash
git add training/field_extractor
git commit -m "docs: add field extractor fine-tuning plan"
```

### Task 9: Package On-Prem Inference Service

**Files:**
- Create: `serving/README.md`
- Create: `serving/api.py`
- Create: `serving/Dockerfile`
- Create: `tests/test_serving_contract.py`

- [ ] **Step 1: Define API contract**

Endpoint:

```text
POST /extract
```

Input:

```json
{
  "image_base64": "...",
  "document_hint": "receipt"
}
```

Output:

```json
{
  "document_type": "receipt",
  "merchant": "Sample Store",
  "total": 14400,
  "currency": "KRW",
  "needs_review": false,
  "review_reasons": []
}
```

- [ ] **Step 2: Implement local-only service**

The service must not call external APIs. It should load local OCR and field extraction models from configured paths.

- [ ] **Step 3: Add container build**

```bash
docker build -t ocr-foundation-serving -f serving/Dockerfile .
```

- [ ] **Step 4: Run local smoke test**

```bash
docker run --rm -p 8080:8080 ocr-foundation-serving
```

Expected: `/extract` responds with valid JSON for a sample private image.

- [ ] **Step 5: Commit**

```bash
git add serving tests/test_serving_contract.py
git commit -m "feat: add on-prem OCR serving skeleton"
```

## Estimated Timeline After 1,000 Files Arrive

| Phase | Expected Time |
| --- | ---: |
| Dataset inventory and sanity checks | 0.5 day |
| Baseline OCR/teacher extraction | 0.5-1 day |
| Human review of JSON labels | 1-3 days |
| Baseline evaluation | 0.5 day |
| PaddleOCR recognition fine-tuning | 0.5-1 day |
| Qwen3-VL LoRA or field extractor distillation | 1-2 days on CUDA server |
| On-prem serving package | 1-2 days |

Total expected first usable on-prem MVP: 5-10 working days after the 1,000 documents are available and reviewed.

## Hardware Guidance

| Machine | Best Use |
| --- | --- |
| Current Apple M4 Pro 48GB | Data prep, local inference smoke tests, PaddleOCR experiments |
| 1x RTX 4090 / L4 / A10G | First Qwen3-VL LoRA experiments |
| L40S / A100 | Faster and more stable VLM fine-tuning |
| CPU-only server | Final lightweight PaddleOCR-based inference if VLM is distilled away |

## Risks

- Teacher labels can teach the model the teacher's mistakes unless reviewed.
- 1,000 documents can overfit to a few layouts if the data is not diverse.
- Qwen3-VL fine-tuning on Apple MPS is not the target path.
- Full on-prem VLM serving can be expensive if the final model remains 8B+.
- Field extraction must mask sensitive account/card values by design.

## First Checkpoint When Data Arrives

Run these commands first:

```bash
git status --short --ignored
find data/private/raw -maxdepth 1 -type f | wc -l
uv run python scripts/build_dataset_inventory.py \
  --raw-dir data/private/raw \
  --output results/dataset_inventory.json
```

Expected:

```text
!! data/
1000
Wrote 1000 items to results/dataset_inventory.json
```
