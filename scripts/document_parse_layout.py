from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
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


TOTAL_LABEL_PATTERN = re.compile(
    r"(합계|총\s*금액|총액|결제\s*금액|받을\s*금액|total|amount\s*due)",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:원|krw|₩)?", re.IGNORECASE)


def parse_receipt_fields(serialized_text: str) -> dict[str, str | None]:
    lines = [line.strip() for line in serialized_text.splitlines() if line.strip()]
    total: str | None = None
    merchant: str | None = None

    for line in lines:
        if merchant is None and not TOTAL_LABEL_PATTERN.search(line):
            merchant = line
        if TOTAL_LABEL_PATTERN.search(line):
            money_matches = MONEY_PATTERN.findall(line)
            if money_matches:
                total = money_matches[-1].strip()

    return {"merchant": merchant, "total": total}


def layout_block_to_dict(block: LayoutBlock) -> dict[str, Any]:
    return {
        "id": block.id,
        "type": block.type,
        "text": block.text,
        "bbox": asdict(block.bbox),
        "confidence": block.confidence,
        "source": block.source,
    }


def build_layout_document(
    image_id: str,
    image_path: Path,
    blocks: list[LayoutBlock],
    page_size: dict[str, int | None],
    sources: list[str],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    ordered_blocks = sort_blocks(blocks)
    serialized_text = serialize_blocks(ordered_blocks)
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "page": page_size,
        "blocks": [layout_block_to_dict(block) for block in ordered_blocks],
        "serialized_text": serialized_text,
        "fields": parse_receipt_fields(serialized_text),
        "sources": sources,
        "warnings": warnings or [],
    }
