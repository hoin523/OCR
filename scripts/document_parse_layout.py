from __future__ import annotations

from dataclasses import dataclass
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
