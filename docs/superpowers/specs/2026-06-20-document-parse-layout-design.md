# Document Parse Layout MVP Design

## Goal

Build a local-only MVP similar to Upstage document parsing: OCR or VLM extraction, layout-aware serialization, key-value parsing, and a clickable document viewer.

## Constraints

- Run on the user's local machine only.
- Keep private images and generated outputs out of git.
- Prefer open models and local tooling.
- Reuse the existing `data/private/raw` input and `results` output conventions.
- Test with receipt samples from `MahmoudSalah/KORIE` after cloning them locally.

## Recommended Model Stack

- Primary document parser: PaddleOCR-VL through the existing `scripts/run_paddleocr_vl.py` path.
- Coordinate fallback: PaddleOCR text through the existing `scripts/run_paddleocr_text.py` output shape.
- Optional semantic extraction: local Ollama `qwen3-vl:8b` through the existing `scripts/run_ollama_ocr.py` path.

PaddleOCR-VL is the best default for document layout parsing because it is designed for page-level document parsing and structured JSON/Markdown output. PaddleOCR text remains useful because clickable overlays need stable text-line boxes, and OCR text results often expose coordinates more directly.

## Architecture

The MVP adds a thin layout normalization layer above existing OCR outputs. It accepts image files, reads existing baseline JSON when present, normalizes OCR lines into page blocks, serializes reading order, extracts simple fields, and writes a self-contained static HTML viewer.

The viewer does not require a backend. It embeds the normalized layout JSON, renders the source image, draws proportional bounding boxes, and links document boxes with parsed block rows so either side can be clicked.

## Components

- `scripts/document_parse_layout.py`: pure normalization, parsing, and viewer rendering helpers.
- `scripts/run_document_parse_layout.py`: CLI entry point for one image or an inventory-driven batch.
- `tests/test_document_parse_layout.py`: behavior tests for normalization, reading order, parsing, and clickable viewer output.
- `Makefile`: convenience targets for layout parsing and KORIE sample testing.
- `README.md`: local usage notes.

## Data Flow

1. User places images under `data/private/raw`.
2. Existing inventory code assigns each image an `image_id`.
3. The layout parser looks for baseline OCR outputs under `results/baselines/<image_id>/`.
4. If PaddleOCR text output exists, it uses coordinates from that file.
5. If PaddleOCR-VL output exists, it keeps structured Markdown/JSON as source metadata.
6. It writes `results/document_parse/<image_id>/layout.json`.
7. It writes `results/document_parse/<image_id>/viewer.html`.

## Layout JSON Shape

```json
{
  "image_id": "receipt_abc123",
  "image_path": "data/private/raw/receipt.png",
  "page": {"width": 1000, "height": 1400},
  "blocks": [
    {
      "id": "block-001",
      "type": "text",
      "text": "합계 12000",
      "confidence": 0.98,
      "bbox": {"x": 100, "y": 240, "width": 220, "height": 32}
    }
  ],
  "serialized_text": "상호명\n합계 12000",
  "fields": {
    "merchant": "상호명",
    "total": "12000"
  },
  "sources": ["paddleocr-text"]
}
```

## Error Handling

- Missing OCR baselines are allowed. The parser emits a layout with zero blocks and a clear warning.
- Unknown OCR JSON shapes are ignored rather than crashing.
- Missing image dimensions fall back to `null` values, while the viewer still renders the image naturally.
- Viewer generation escapes embedded text and JSON to avoid invalid HTML.

## Testing

- Unit tests cover PaddleOCR output normalization, reading-order sorting, field extraction, and HTML click wiring.
- A CLI dry run is tested with synthetic OCR JSON.
- Manual/local verification uses KORIE receipt images cloned under `data/private/external/KORIE` or copied into `data/private/raw/korie`.

## KORIE Test Plan

Clone the dataset locally:

```bash
git clone https://github.com/MahmoudSalah/KORIE.git data/private/external/KORIE
```

Copy or symlink a small receipt sample set into `data/private/raw/korie`, run OCR baselines locally, then generate layout viewers. The repository keeps both dataset files and generated outputs ignored.
