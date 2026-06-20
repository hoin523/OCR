# Document Parse Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-only document parse layout MVP that normalizes OCR output, extracts simple fields, and generates a clickable HTML viewer.

**Architecture:** Add a pure Python normalization module plus a small CLI. Reuse existing inventory and baseline output conventions, and keep generated layout artifacts under ignored `results/document_parse`.

**Tech Stack:** Python 3.12, pytest, existing PaddleOCR/PaddleOCR-VL/Ollama scripts, static HTML/CSS/JavaScript.

---

## File Structure

- Create: `scripts/document_parse_layout.py`
  - Dataclasses and functions for OCR normalization, reading-order serialization, field parsing, and viewer rendering.
- Create: `scripts/run_document_parse_layout.py`
  - CLI for one image or all inventory images.
- Create: `tests/test_document_parse_layout.py`
  - TDD coverage for normalization, parser behavior, and viewer click wiring.
- Modify: `Makefile`
  - Add `layout`, `layout-smoke`, and `korie-clone` targets.
- Modify: `README.md`
  - Add local document parse layout usage.

### Task 1: Layout Normalization

**Files:**
- Create: `tests/test_document_parse_layout.py`
- Create: `scripts/document_parse_layout.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from scripts.document_parse_layout import (
    BoundingBox,
    LayoutBlock,
    normalize_paddleocr_text_payload,
    serialize_blocks,
)


def test_normalize_paddleocr_text_payload_extracts_line_boxes_in_reading_order():
    payload = {
        "model": "PaddleOCR text (korean)",
        "results": [
            {
                "res": {
                    "dt_polys": [
                        [[200, 100], [260, 100], [260, 120], [200, 120]],
                        [[20, 40], [80, 40], [80, 60], [20, 60]],
                    ],
                    "rec_texts": ["12000", "Store"],
                    "rec_scores": [0.91, 0.99],
                }
            }
        ],
    }

    blocks = normalize_paddleocr_text_payload(payload)

    assert [block.text for block in blocks] == ["Store", "12000"]
    assert blocks[0].bbox == BoundingBox(x=20, y=40, width=60, height=20)
    assert blocks[0].confidence == 0.99
    assert blocks[0].source == "paddleocr-text"


def test_serialize_blocks_joins_non_empty_text_in_reading_order():
    blocks = [
        LayoutBlock("b2", "text", "B", BoundingBox(10, 50, 20, 10), 0.9, "test"),
        LayoutBlock("b1", "text", "A", BoundingBox(10, 10, 20, 10), 0.9, "test"),
    ]

    assert serialize_blocks(blocks) == "A\nB"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest python -m pytest tests/test_document_parse_layout.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function imports.

- [ ] **Step 3: Implement minimal normalization code**

```python
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class LayoutBlock:
    id: str
    type: str
    text: str
    bbox: BoundingBox
    confidence: float | None
    source: str


def polygon_to_bbox(points: list[list[float]]) -> BoundingBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return BoundingBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def sort_blocks(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    return sorted(blocks, key=lambda block: (block.bbox.y, block.bbox.x, block.id))


def normalize_paddleocr_text_payload(payload: dict[str, Any]) -> list[LayoutBlock]:
    blocks: list[LayoutBlock] = []
    for result in payload.get("results", []):
        content = result.get("res", result) if isinstance(result, dict) else {}
        polys = content.get("dt_polys") or content.get("rec_polys") or []
        texts = content.get("rec_texts") or []
        scores = content.get("rec_scores") or []
        for index, text in enumerate(texts):
            if index >= len(polys) or not str(text).strip():
                continue
            score = scores[index] if index < len(scores) else None
            blocks.append(
                LayoutBlock(
                    id=f"block-{len(blocks) + 1:03d}",
                    type="text",
                    text=str(text),
                    bbox=polygon_to_bbox(polys[index]),
                    confidence=float(score) if score is not None else None,
                    source="paddleocr-text",
                )
            )
    return sort_blocks(blocks)


def serialize_blocks(blocks: list[LayoutBlock]) -> str:
    return "\n".join(block.text for block in sort_blocks(blocks) if block.text.strip())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest python -m pytest tests/test_document_parse_layout.py -v`
Expected: PASS.

### Task 2: Layout JSON and Field Parser

**Files:**
- Modify: `tests/test_document_parse_layout.py`
- Modify: `scripts/document_parse_layout.py`

- [ ] **Step 1: Write failing tests**

```python
from scripts.document_parse_layout import build_layout_document, parse_receipt_fields


def test_parse_receipt_fields_extracts_total_and_merchant():
    fields = parse_receipt_fields("Nice Store\n공급가 9000\n합계 10,000원")

    assert fields["merchant"] == "Nice Store"
    assert fields["total"] == "10,000원"


def test_build_layout_document_returns_serializable_shape(tmp_path: Path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fake")
    blocks = [
        LayoutBlock("block-001", "text", "Nice Store", BoundingBox(0, 0, 100, 20), 0.99, "test"),
        LayoutBlock("block-002", "text", "합계 10,000원", BoundingBox(0, 40, 100, 20), 0.98, "test"),
    ]

    layout = build_layout_document(
        image_id="receipt_abc",
        image_path=image,
        blocks=blocks,
        page_size={"width": 640, "height": 480},
        sources=["test"],
    )

    assert layout["image_id"] == "receipt_abc"
    assert layout["page"] == {"width": 640, "height": 480}
    assert layout["serialized_text"] == "Nice Store\n합계 10,000원"
    assert layout["fields"]["total"] == "10,000원"
    assert layout["blocks"][0]["bbox"]["width"] == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest python -m pytest tests/test_document_parse_layout.py -v`
Expected: FAIL with missing parser functions.

- [ ] **Step 3: Implement minimal parser code**

Add functions that build dictionaries with `asdict`, extract a merchant from the first non-total line, and extract total from Korean/English total labels.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest python -m pytest tests/test_document_parse_layout.py -v`
Expected: PASS.

### Task 3: Clickable Viewer Rendering

**Files:**
- Modify: `tests/test_document_parse_layout.py`
- Modify: `scripts/document_parse_layout.py`

- [ ] **Step 1: Write failing tests**

```python
from scripts.document_parse_layout import render_viewer_html


def test_render_viewer_html_links_blocks_and_rows():
    layout = {
        "image_id": "receipt_abc",
        "image_path": "receipt.png",
        "page": {"width": 640, "height": 480},
        "blocks": [
            {
                "id": "block-001",
                "type": "text",
                "text": "Nice Store",
                "confidence": 0.99,
                "source": "test",
                "bbox": {"x": 10, "y": 20, "width": 100, "height": 30},
            }
        ],
        "serialized_text": "Nice Store",
        "fields": {"merchant": "Nice Store"},
        "sources": ["test"],
        "warnings": [],
    }

    html = render_viewer_html(layout)

    assert 'data-block-id="block-001"' in html
    assert "selectBlock" in html
    assert "Nice Store" in html
    assert "left: 1.5625%" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest python -m pytest tests/test_document_parse_layout.py -v`
Expected: FAIL with missing viewer function.

- [ ] **Step 3: Implement static HTML renderer**

Render a self-contained document with CSS overlay boxes, block rows, field table, and JavaScript `selectBlock(blockId)` that toggles `.active` on both overlay and row elements.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest python -m pytest tests/test_document_parse_layout.py -v`
Expected: PASS.

### Task 4: CLI and Make Targets

**Files:**
- Create: `scripts/run_document_parse_layout.py`
- Modify: `tests/test_document_parse_layout.py`
- Modify: `Makefile`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI test**

```python
from scripts.run_document_parse_layout import parse_one


def test_parse_one_writes_layout_and_viewer_from_baseline(tmp_path: Path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fake")
    baseline = tmp_path / "baselines" / "receipt_abc" / "paddleocr-text.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        '{"results":[{"res":{"dt_polys":[[[0,0],[10,0],[10,10],[0,10]]],"rec_texts":["Store"],"rec_scores":[0.9]}}]}',
        encoding="utf-8",
    )

    output = tmp_path / "document_parse"
    layout_path = parse_one(
        image_id="receipt_abc",
        image_path=image,
        baselines_dir=tmp_path / "baselines",
        output_dir=output,
    )

    assert layout_path == output / "receipt_abc" / "layout.json"
    assert layout_path.exists()
    assert (output / "receipt_abc" / "viewer.html").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest python -m pytest tests/test_document_parse_layout.py -v`
Expected: FAIL with missing CLI module/function.

- [ ] **Step 3: Implement CLI and docs**

The CLI should accept `--image-id`, `--image`, `--baselines-dir`, and `--output-dir` for single-image parsing. It should also support `--inventory` for batch parsing.

- [ ] **Step 4: Run full tests**

Run: `make test`
Expected: all tests pass.

### Task 5: KORIE Local Smoke Test

**Files:**
- No tracked source changes required unless a dataset format note is discovered.

- [ ] **Step 1: Clone KORIE locally**

Run: `git clone https://github.com/MahmoudSalah/KORIE.git data/private/external/KORIE`
Expected: repository exists under ignored `data/private`.

- [ ] **Step 2: Inspect available receipt images**

Run: `find data/private/external/KORIE -type f | head -50`
Expected: identify image paths or dataset files.

- [ ] **Step 3: Copy a small image sample**

Run: create `data/private/raw/korie`, copy the first available `.jpg`, `.jpeg`, `.png`, or `.webp` file into it.

- [ ] **Step 4: Generate inventory and local layout viewer**

Run: `make inventory`, then run PaddleOCR text baseline if local dependencies work, then `make layout`.
Expected: `results/document_parse/<image_id>/viewer.html` exists for the sample.

- [ ] **Step 5: Verify viewer artifact**

Run: inspect generated `layout.json` and `viewer.html` for non-empty blocks when OCR baseline exists. If PaddleOCR dependencies fail on this machine, report the exact failure and still verify empty-layout viewer generation from the image.
