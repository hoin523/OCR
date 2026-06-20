from __future__ import annotations

import html
import json
import re
import struct
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


@dataclass(frozen=True)
class LayoutElement:
    id: str
    type: str
    text: str
    bbox: BoundingBox
    child_block_ids: list[str]
    markdown: str
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


def _median(values: list[float], default: float) -> float:
    if not values:
        return default
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _union_bbox(blocks: list[LayoutBlock]) -> BoundingBox:
    min_x = min(block.bbox.x for block in blocks)
    min_y = min(block.bbox.y for block in blocks)
    max_x = max(block.bbox.x + block.bbox.width for block in blocks)
    max_y = max(block.bbox.y + block.bbox.height for block in blocks)
    return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)


def _vertical_gap(previous: LayoutBlock, current: LayoutBlock) -> float:
    return current.bbox.y - (previous.bbox.y + previous.bbox.height)


def _horizontal_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    overlap = min(left.x + left.width, right.x + right.width) - max(left.x, right.x)
    if overlap <= 0:
        return 0.0
    return overlap / max(min(left.width, right.width), 1.0)


def _looks_like_same_text_region(previous: LayoutBlock, current: LayoutBlock, gap_threshold: float) -> bool:
    gap = _vertical_gap(previous, current)
    left_delta = abs(previous.bbox.x - current.bbox.x)
    same_column = left_delta <= max(previous.bbox.height, current.bbox.height, 18) * 1.4
    overlaps = _horizontal_overlap_ratio(previous.bbox, current.bbox) >= 0.18
    same_row = gap < 0 and abs(previous.bbox.y - current.bbox.y) <= max(previous.bbox.height, current.bbox.height) * 0.55
    return (0 <= gap <= gap_threshold and (same_column or overlaps)) or same_row


def _element_type_for_blocks(blocks: list[LayoutBlock]) -> str:
    text = "\n".join(block.text.strip() for block in blocks if block.text.strip())
    lines = [line for line in text.splitlines() if line]
    if len(lines) >= 3 and sum(1 for line in lines if re.search(r"\s{2,}|[:：]\s*[^:：]+", line)) >= 2:
        return "table_candidate"
    if lines and all(re.match(r"^(\d+[.)]|[-*•])\s+", line) for line in lines):
        return "list"
    return "paragraph"


def _markdown_for_element(element_type: str, text: str) -> str:
    if element_type == "list":
        return "\n".join(
            re.sub(r"^(\d+[.)]|[-*•])\s+", "- ", line)
            for line in text.splitlines()
            if line.strip()
        )
    return text


def build_layout_elements(blocks: list[LayoutBlock], page_size: dict[str, int | None] | None = None) -> list[LayoutElement]:
    ordered_blocks = sort_blocks(blocks)
    if not ordered_blocks:
        return []

    line_heights = [block.bbox.height for block in ordered_blocks if block.bbox.height > 0]
    median_line_height = _median(line_heights, 18)
    gap_threshold = max(10.0, median_line_height * 0.8)

    groups: list[list[LayoutBlock]] = []
    current_group: list[LayoutBlock] = []
    for block in ordered_blocks:
        if not current_group:
            current_group = [block]
            continue

        previous = current_group[-1]
        if _looks_like_same_text_region(previous, block, gap_threshold):
            current_group.append(block)
        else:
            groups.append(current_group)
            current_group = [block]

    if current_group:
        groups.append(current_group)

    elements: list[LayoutElement] = []
    for index, group in enumerate(groups, start=1):
        text = "\n".join(block.text.strip() for block in group if block.text.strip())
        element_type = _element_type_for_blocks(group)
        elements.append(
            LayoutElement(
                id=f"element-{index:03d}",
                type=element_type,
                text=text,
                bbox=_union_bbox(group),
                child_block_ids=[block.id for block in group],
                markdown=_markdown_for_element(element_type, text),
                source="heuristic-layout",
            )
        )
    return elements


def serialize_elements(elements: list[LayoutElement]) -> str:
    return "\n\n".join(element.markdown for element in elements if element.markdown.strip())


TOTAL_LABEL_PATTERN = re.compile(
    r"(합계|총\s*금액|총액|결제\s*금액|받을\s*금액|total|amount\s*due)",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:원|krw|₩)?", re.IGNORECASE)


def _money_value(token: str) -> float:
    normalized = re.sub(r"[^\d.+-]", "", token.replace(",", ""))
    try:
        return float(normalized)
    except ValueError:
        return 0.0


def _positive_money_tokens(lines: list[str]) -> list[str]:
    tokens: list[str] = []
    for line in lines:
        for token in MONEY_PATTERN.findall(line):
            stripped = token.strip()
            if stripped.startswith("-"):
                continue
            if _money_value(stripped) <= 0:
                continue
            tokens.append(stripped)
    return tokens


def parse_receipt_fields(serialized_text: str) -> dict[str, str | None]:
    lines = [line.strip() for line in serialized_text.splitlines() if line.strip()]
    total: str | None = None
    merchant: str | None = None

    for index, line in enumerate(lines):
        if merchant is None and not TOTAL_LABEL_PATTERN.search(line):
            merchant = line
        if TOTAL_LABEL_PATTERN.search(line):
            nearby_lines = lines[index : index + 7]
            money_matches = _positive_money_tokens(nearby_lines)
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


def layout_element_to_dict(element: LayoutElement) -> dict[str, Any]:
    return {
        "id": element.id,
        "type": element.type,
        "text": element.text,
        "bbox": asdict(element.bbox),
        "child_block_ids": element.child_block_ids,
        "markdown": element.markdown,
        "source": element.source,
    }


def build_layout_document(
    image_id: str,
    image_path: Path,
    blocks: list[LayoutBlock],
    page_size: dict[str, int | None],
    sources: list[str],
    processing: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    ordered_blocks = sort_blocks(blocks)
    elements = build_layout_elements(ordered_blocks, page_size)
    serialized_text = serialize_blocks(ordered_blocks)
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "page": page_size,
        "blocks": [layout_block_to_dict(block) for block in ordered_blocks],
        "elements": [layout_element_to_dict(element) for element in elements],
        "serialized_text": serialized_text,
        "serialized_markdown": serialize_elements(elements),
        "fields": parse_receipt_fields(serialized_text),
        "sources": sources,
        "processing": processing or {},
        "warnings": warnings or [],
    }


def read_image_size(path: Path) -> dict[str, int | None]:
    try:
        with path.open("rb") as handle:
            header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width, height = struct.unpack(">II", header[16:24])
                return {"width": int(width), "height": int(height)}

            if header[:2] == b"\xff\xd8":
                handle.seek(2)
                while True:
                    marker_start = handle.read(1)
                    if not marker_start:
                        break
                    if marker_start != b"\xff":
                        continue
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7"}:
                        segment = handle.read(7)
                        height, width = struct.unpack(">HH", segment[3:7])
                        return {"width": int(width), "height": int(height)}
                    length_bytes = handle.read(2)
                    if len(length_bytes) != 2:
                        break
                    length = struct.unpack(">H", length_bytes)[0]
                    handle.seek(max(length - 2, 0), 1)

            if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                handle.seek(12)
                while True:
                    chunk_header = handle.read(8)
                    if len(chunk_header) != 8:
                        break
                    chunk_type = chunk_header[:4]
                    chunk_size = struct.unpack("<I", chunk_header[4:])[0]
                    chunk_data = handle.read(chunk_size)
                    if chunk_type == b"VP8 " and len(chunk_data) >= 10 and chunk_data[3:6] == b"\x9d\x01\x2a":
                        width = struct.unpack("<H", chunk_data[6:8])[0] & 0x3FFF
                        height = struct.unpack("<H", chunk_data[8:10])[0] & 0x3FFF
                        return {"width": int(width), "height": int(height)}
                    if chunk_type == b"VP8X" and len(chunk_data) >= 10:
                        width = 1 + int.from_bytes(chunk_data[4:7], "little")
                        height = 1 + int.from_bytes(chunk_data[7:10], "little")
                        return {"width": int(width), "height": int(height)}
                    if chunk_type == b"VP8L" and len(chunk_data) >= 5 and chunk_data[0] == 0x2F:
                        bits = int.from_bytes(chunk_data[1:5], "little")
                        width = 1 + (bits & 0x3FFF)
                        height = 1 + ((bits >> 14) & 0x3FFF)
                        return {"width": int(width), "height": int(height)}
                    if chunk_size % 2 == 1:
                        handle.seek(1, 1)
    except OSError:
        pass

    return {"width": None, "height": None}


def _format_percent(value: float, total: float | int | None) -> str:
    if not total:
        return "0%"
    formatted = f"{(float(value) / float(total)) * 100:.4f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def _image_src_for_html(image_path: str) -> str:
    path = Path(image_path)
    if path.exists():
        return path.resolve().as_uri()
    return image_path


def _safe_json_for_script(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")


def _fallback_elements_from_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    elements = []
    for index, block in enumerate(blocks, start=1):
        block_id = str(block.get("id", f"block-{index:03d}"))
        text = str(block.get("text", ""))
        elements.append(
            {
                "id": f"element-{index:03d}",
                "type": "paragraph",
                "text": text,
                "bbox": block.get("bbox", {}),
                "child_block_ids": [block_id],
                "markdown": text,
                "source": "viewer-fallback",
            }
        )
    return elements


def _layout_elements_for_render(layout: dict[str, Any]) -> list[dict[str, Any]]:
    elements = layout.get("elements")
    if isinstance(elements, list) and elements:
        return [element for element in elements if isinstance(element, dict)]
    blocks = layout.get("blocks")
    if isinstance(blocks, list):
        return _fallback_elements_from_blocks([block for block in blocks if isinstance(block, dict)])
    return []


def render_multipage_viewer_html(layouts: list[dict[str, Any]], title: str = "Document Parse") -> str:
    pages = []
    for index, layout in enumerate(layouts):
        blocks = layout.get("blocks", [])
        elements = _layout_elements_for_render(layout)
        fields = layout.get("fields", {})
        processing = layout.get("processing", {})
        image_id = str(layout.get("image_id") or f"page_{index + 1:03d}")
        page_title = str(layout.get("page_title") or image_id)
        pages.append(
            {
                "index": index,
                "label": f"{index + 1}",
                "title": page_title,
                "imageId": image_id,
                "imagePath": _image_src_for_html(str(layout.get("image_path", ""))),
                "viewerPath": str(layout.get("viewer_path", "")),
                "page": layout.get("page", {}),
                "blocks": blocks,
                "elements": elements,
                "serializedText": str(layout.get("serialized_text", "")),
                "serializedMarkdown": str(layout.get("serialized_markdown", "")),
                "fields": fields,
                "sources": layout.get("sources", []),
                "processing": processing,
                "warnings": layout.get("warnings", []),
                "blockCount": len(blocks),
                "elementCount": len(elements),
                "fieldCount": sum(1 for value in fields.values() if value is not None) if isinstance(fields, dict) else 0,
            }
        )

    pages_json = _safe_json_for_script(pages)
    document_title = html.escape(title)

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{document_title} multipage parse layout</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #edf2f7;
      color: #18202f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #f8fafc 0%, #edf2f7 48%, #e7edf5 100%);
    }}
    button {{
      font: inherit;
    }}
    .multi-shell {{
      display: grid;
      grid-template-columns: 248px minmax(380px, 1fr) 400px;
      grid-template-rows: auto 1fr;
      gap: 14px;
      min-height: 100vh;
      padding: 16px;
    }}
    .pipeline {{
      grid-column: 1 / -1;
      background: #fff;
      border: 1px solid #d8e0ec;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 12px 32px rgba(31, 44, 71, 0.08);
    }}
    .pipeline-header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 14px 16px 10px;
      border-bottom: 1px solid #e6ebf2;
    }}
    .pipeline-title h1 {{
      margin: 0 0 2px;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 750;
    }}
    .pipeline-title span, .pipeline-meta {{
      color: #607087;
      font-size: 12px;
    }}
    .pipeline-rail {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      padding: 14px 16px 16px;
    }}
    .pipeline-step {{
      position: relative;
      display: grid;
      gap: 8px;
      min-height: 92px;
      padding: 14px;
      border: 1px solid #dbe3ef;
      border-right: 0;
      background: #fbfcfe;
    }}
    .pipeline-step:first-child {{ border-radius: 8px 0 0 8px; }}
    .pipeline-step:last-child {{
      border-right: 1px solid #dbe3ef;
      border-radius: 0 8px 8px 0;
    }}
    .pipeline-step:not(:last-child)::after {{
      content: "";
      position: absolute;
      top: 50%;
      right: -8px;
      width: 16px;
      height: 16px;
      transform: translateY(-50%) rotate(45deg);
      border-top: 1px solid #dbe3ef;
      border-right: 1px solid #dbe3ef;
      background: #fbfcfe;
      z-index: 2;
    }}
    .step-kicker {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: #607087;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .step-dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #2f6fed;
    }}
    .pipeline-step:nth-child(2) .step-dot {{ background: #00a18a; }}
    .pipeline-step:nth-child(3) .step-dot {{ background: #b86b00; }}
    .pipeline-step:nth-child(4) .step-dot {{ background: #cf2f64; }}
    .step-name {{
      margin: 0;
      font-size: 15px;
      font-weight: 750;
    }}
    .step-value {{
      color: #334155;
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .page-nav, .document-pane, .parsed-pane {{
      min-width: 0;
      background: #fff;
      border: 1px solid #d8e0ec;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 12px 32px rgba(31, 44, 71, 0.08);
    }}
    .page-nav {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      max-height: calc(100vh - 148px);
    }}
    .nav-header, .document-header, .parsed-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 48px;
      padding: 0 16px;
      border-bottom: 1px solid #e6ebf2;
      font-size: 14px;
      font-weight: 650;
    }}
    .nav-header span:last-child, .document-header span:last-child, .parsed-header span:last-child {{
      color: #607087;
      font-size: 12px;
      font-weight: 650;
    }}
    .page-tabs {{
      overflow: auto;
      padding: 10px;
    }}
    .page-tab {{
      display: grid;
      grid-template-columns: 34px 1fr;
      gap: 9px;
      width: 100%;
      min-height: 54px;
      margin: 0 0 8px;
      padding: 8px;
      border: 1px solid #dfe5ee;
      border-radius: 7px;
      background: #fbfcfe;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }}
    .page-tab:hover, .page-tab.active {{
      border-color: #2f6fed;
      background: #f4f7ff;
    }}
    .page-number {{
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 7px;
      background: #edf3ff;
      color: #1e4fb8;
      font-size: 13px;
      font-weight: 750;
    }}
    .page-title {{
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.35;
    }}
    .page-meta {{
      display: block;
      margin-top: 3px;
      color: #697588;
      font-size: 11px;
    }}
    .document-pane {{
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      max-height: calc(100vh - 148px);
    }}
    .page-controls {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid #eef1f5;
    }}
    .nav-button {{
      min-width: 36px;
      height: 32px;
      border: 1px solid #d6deea;
      border-radius: 7px;
      background: #fff;
      color: #28364a;
      cursor: pointer;
    }}
    .nav-button:hover {{
      border-color: #2f6fed;
      color: #1e4fb8;
    }}
    .current-label {{
      text-align: center;
      color: #334155;
      font-size: 13px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    .stage-wrap {{
      overflow: auto;
      padding: 16px;
    }}
    .stage {{
      position: relative;
      width: min(100%, 980px);
      margin: 0 auto;
      background: #f1f5f9;
      border: 1px solid #e1e7f0;
      border-radius: 8px;
      overflow: hidden;
    }}
    .stage img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .element-box, .ocr-box {{
      position: absolute;
      appearance: none;
      padding: 0;
      cursor: pointer;
    }}
    .element-box {{
      border: 1.5px solid rgba(0, 161, 138, 0.58);
      background: rgba(0, 161, 138, 0.025);
      border-radius: 5px;
      z-index: 3;
    }}
    .element-box:hover, .element-box.active {{
      border-color: #b86b00;
      background: rgba(184, 107, 0, 0.08);
      box-shadow: 0 0 0 2px rgba(184, 107, 0, 0.1);
    }}
    .ocr-box {{
      border: 1px solid rgba(47, 111, 237, 0.55);
      background: transparent;
      border-radius: 3px;
      display: none;
      opacity: 0.7;
      z-index: 2;
    }}
    .stage.show-ocr-lines .ocr-box {{ display: block; }}
    .stage.elements-only .ocr-box {{ display: none; }}
    .ocr-box:hover, .ocr-box.active {{
      border-color: #d91f4c;
      background: rgba(217, 31, 76, 0.08);
      box-shadow: 0 0 0 2px rgba(217, 31, 76, 0.1);
    }}
    .overlay-controls {{
      display: inline-flex;
      gap: 4px;
      padding: 3px;
      border: 1px solid #d8e0ec;
      border-radius: 7px;
      background: #f8fafc;
    }}
    .overlay-toggle {{
      min-height: 30px;
      padding: 0 10px;
      border: 0;
      border-radius: 5px;
      background: transparent;
      color: #607087;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .overlay-toggle.active {{
      background: #ffffff;
      color: #18202f;
      box-shadow: 0 1px 3px rgba(31, 44, 71, 0.12);
    }}
    .parsed-pane {{
      display: grid;
      grid-template-rows: auto auto auto auto minmax(120px, 1fr) minmax(120px, 1fr);
      max-height: calc(100vh - 148px);
    }}
    .section-title {{
      margin: 0;
      padding: 12px 12px 8px;
      color: #607087;
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }}
    .field-section, .timing {{
      border-bottom: 1px solid #eef2f7;
    }}
    .fields {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .fields th, .fields td {{
      padding: 9px 12px;
      border-bottom: 1px solid #eef1f5;
      text-align: left;
      vertical-align: top;
    }}
    .fields th {{
      width: 116px;
      color: #607087;
      font-weight: 650;
    }}
    .timing {{
      padding: 0 12px 10px;
    }}
    .timing table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .timing th, .timing td {{
      padding: 4px 0;
      text-align: left;
      vertical-align: top;
    }}
    .timing th {{
      color: #697588;
      font-weight: 600;
    }}
    .timing td {{
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    .element-section, .block-section {{
      min-height: 0;
      overflow: hidden;
    }}
    .element-list, .block-list {{
      overflow: auto;
      padding: 8px;
      height: 100%;
    }}
    .element-row, .block-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      width: 100%;
      min-height: 40px;
      margin: 0 0 6px;
      padding: 9px 10px;
      border: 1px solid #dfe5ee;
      border-radius: 6px;
      background: #fbfcfe;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }}
    .element-row:hover, .element-row.active {{
      border-color: #00a18a;
      background: #f0fffb;
    }}
    .block-row:hover, .block-row.active {{
      border-color: #d91f4c;
      background: #fff5f7;
    }}
    .block-text {{
      overflow-wrap: anywhere;
      line-height: 1.35;
      font-size: 13px;
    }}
    .block-meta {{
      color: #697588;
      font-size: 12px;
      white-space: nowrap;
    }}
    .warnings {{
      margin: 0;
      padding: 10px 16px 10px 28px;
      border-bottom: 1px solid #eef1f5;
      color: #9b3a13;
      background: #fff7ed;
      font-size: 13px;
    }}
    @media (max-width: 1080px) {{
      .multi-shell {{
        grid-template-columns: 1fr;
        padding: 10px;
      }}
      .pipeline-rail {{
        grid-template-columns: 1fr;
        gap: 8px;
      }}
      .pipeline-step, .pipeline-step:first-child, .pipeline-step:last-child {{
        border: 1px solid #dbe3ef;
        border-radius: 8px;
      }}
      .pipeline-step:not(:last-child)::after {{ display: none; }}
      .page-nav, .document-pane, .parsed-pane {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <main class="multi-shell">
    <section class="pipeline">
      <header class="pipeline-header">
        <div class="pipeline-title">
          <h1>Document Parse Pipeline</h1>
          <span>{document_title}</span>
        </div>
        <div class="pipeline-meta"><span id="source-label">local parser</span></div>
      </header>
      <div class="pipeline-rail">
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>AI Model</span></div>
          <p class="step-name">Detector</p>
          <div class="step-value"><span id="detector-count">0</span> layout regions</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>OCR</span></div>
          <p class="step-name">Recognizer</p>
          <div class="step-value"><span id="recognizer-count">0</span> text lines</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>2D to 1D</span></div>
          <p class="step-name">Serializer</p>
          <div class="step-value"><span id="line-count">0</span> ordered lines</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>JSON</span></div>
          <p class="step-name">Parser</p>
          <div class="step-value"><span id="field-count">0</span> extracted fields</div>
        </article>
      </div>
    </section>
    <nav class="page-nav" aria-label="Parsed pages">
      <header class="nav-header"><span>Pages</span><span id="page-count">0 pages</span></header>
      <div class="page-tabs" id="page-tabs"></div>
    </nav>
    <section class="document-pane">
      <header class="document-header">
        <span>Detected Layout</span>
        <div class="overlay-controls" aria-label="Overlay mode">
          <button class="overlay-toggle active" type="button" data-overlay-mode="elements" onclick="setOverlayMode('elements')">Elements</button>
          <button class="overlay-toggle" type="button" data-overlay-mode="lines" onclick="setOverlayMode('lines')">OCR Lines</button>
        </div>
        <span id="block-count">0 blocks</span>
      </header>
      <div class="page-controls">
        <button class="nav-button" type="button" onclick="goPage(-1)" aria-label="Previous page">‹</button>
        <div class="current-label" id="current-label">Page</div>
        <button class="nav-button" type="button" onclick="goPage(1)" aria-label="Next page">›</button>
      </div>
      <div class="stage-wrap">
        <div id="stage" class="stage elements-only"></div>
      </div>
    </section>
    <aside class="parsed-pane">
      <header class="parsed-header"><span>Parser Output</span><span id="parser-source">local parser</span></header>
      <ul class="warnings" id="warnings" hidden></ul>
      <section class="field-section">
        <h2 class="section-title">Key-Value Output</h2>
        <table class="fields"><tbody id="field-body"></tbody></table>
      </section>
      <section class="timing">
        <h2 class="section-title">Processing Time</h2>
        <table><tbody id="timing-body"></tbody></table>
      </section>
      <section class="element-section">
        <h2 class="section-title">Layout Elements</h2>
        <div class="element-list" id="element-list"></div>
      </section>
      <section class="block-section">
        <h2 class="section-title">OCR Lines</h2>
        <div class="block-list" id="block-list"></div>
      </section>
    </aside>
  </main>
  <script>
    const PAGES = {pages_json};
    let currentPageIndex = 0;
    let overlayMode = "elements";

    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function percent(value, total) {{
      if (!total) return "0%";
      const raw = (Number(value || 0) / Number(total)) * 100;
      return `${{Number(raw.toFixed(4))}}%`;
    }}

    function sourceLabel(page) {{
      return (page.sources && page.sources.length) ? page.sources.join(", ") : "local parser";
    }}

    function timingRows(page) {{
      const rows = [];
      const elapsed = page.processing && page.processing.source_elapsed_seconds;
      if (elapsed) {{
        Object.entries(elapsed).forEach(([key, value]) => {{
          if (value !== null && value !== undefined) rows.push([key, `${{Number(value).toFixed(2)}}s`]);
        }});
      }}
      const layoutElapsed = page.processing && page.processing.layout_elapsed_seconds;
      if (layoutElapsed !== null && layoutElapsed !== undefined) rows.push(["layout", `${{Number(layoutElapsed).toFixed(2)}}s`]);
      return rows;
    }}

    function selectBlock(blockId) {{
      document.querySelectorAll("[data-block-id]").forEach(function (node) {{
        node.classList.toggle("active", node.dataset.blockId === blockId);
      }});
      setOverlayMode("lines");
      const row = document.querySelector('.block-row[data-block-id="' + blockId + '"]');
      if (row) row.scrollIntoView({{ block: "nearest" }});
    }}

    function selectElement(elementId) {{
      document.querySelectorAll("[data-element-id]").forEach(function (node) {{
        node.classList.toggle("active", node.dataset.elementId === elementId);
      }});
      const row = document.querySelector('.element-row[data-element-id="' + elementId + '"]');
      if (row) row.scrollIntoView({{ block: "nearest" }});
    }}

    function applyOverlayMode() {{
      const stage = document.getElementById("stage");
      if (!stage) return;
      stage.classList.toggle("elements-only", overlayMode === "elements");
      stage.classList.toggle("show-ocr-lines", overlayMode === "lines");
      document.querySelectorAll("[data-overlay-mode]").forEach(function (node) {{
        node.classList.toggle("active", node.dataset.overlayMode === overlayMode);
      }});
    }}

    function setOverlayMode(mode) {{
      overlayMode = mode === "lines" ? "lines" : "elements";
      applyOverlayMode();
    }}

    function goPage(offset) {{
      if (!PAGES.length) return;
      const next = Math.max(0, Math.min(PAGES.length - 1, currentPageIndex + offset));
      renderPage(next);
    }}

    function renderPage(index) {{
      if (!PAGES.length) return;
      currentPageIndex = Math.max(0, Math.min(PAGES.length - 1, index));
      const page = PAGES[currentPageIndex];
      const pageWidth = page.page && page.page.width;
      const pageHeight = page.page && page.page.height;
      const blocks = page.blocks || [];
      const elements = page.elements || [];
      const nonEmptyLines = String(page.serializedText || "").split("\\n").filter(Boolean);
      const fields = page.fields || {{}};
      const fieldEntries = Object.entries(fields).filter(([, value]) => value !== null && value !== undefined);
      const source = sourceLabel(page);

      document.getElementById("source-label").textContent = source;
      document.getElementById("parser-source").textContent = source;
      document.getElementById("detector-count").textContent = elements.length;
      document.getElementById("recognizer-count").textContent = blocks.filter((block) => String(block.text || "").trim()).length;
      document.getElementById("line-count").textContent = nonEmptyLines.length;
      document.getElementById("field-count").textContent = fieldEntries.length;
      document.getElementById("block-count").textContent = `${{elements.length}} elements · ${{blocks.length}} lines`;
      document.getElementById("page-count").textContent = `${{PAGES.length}} pages`;
      document.getElementById("current-label").textContent = `Page ${{currentPageIndex + 1}} / ${{PAGES.length}} · ${{page.title}}`;

      document.getElementById("page-tabs").innerHTML = PAGES.map((item, itemIndex) => `
        <button class="page-tab ${{itemIndex === currentPageIndex ? "active" : ""}}" type="button" onclick="renderPage(${{itemIndex}})">
          <span class="page-number">${{escapeHtml(item.label)}}</span>
          <span class="page-title">${{escapeHtml(item.title)}}<span class="page-meta">${{item.elementCount}} elements · ${{item.blockCount}} lines</span></span>
        </button>
      `).join("");

      const elementMarkup = elements.map((element) => {{
        const bbox = element.bbox || {{}};
        const style = [
          `left: ${{percent(bbox.x, pageWidth)}}`,
          `top: ${{percent(bbox.y, pageHeight)}}`,
          `width: ${{percent(bbox.width, pageWidth)}}`,
          `height: ${{percent(bbox.height, pageHeight)}}`
        ].join("; ");
        return `<button class="element-box" style="${{style}}" data-element-id="${{escapeHtml(element.id)}}" aria-label="${{escapeHtml(element.type + ': ' + element.text)}}" onclick="selectElement('${{escapeHtml(element.id)}}')"></button>`;
      }}).join("");
      const boxMarkup = blocks.map((block) => {{
        const bbox = block.bbox || {{}};
        const style = [
          `left: ${{percent(bbox.x, pageWidth)}}`,
          `top: ${{percent(bbox.y, pageHeight)}}`,
          `width: ${{percent(bbox.width, pageWidth)}}`,
          `height: ${{percent(bbox.height, pageHeight)}}`
        ].join("; ");
        return `<button class="ocr-box" style="${{style}}" data-block-id="${{escapeHtml(block.id)}}" aria-label="${{escapeHtml(block.text)}}" onclick="selectBlock('${{escapeHtml(block.id)}}')"></button>`;
      }}).join("");
      document.getElementById("stage").innerHTML = `<img src="${{escapeHtml(page.imagePath)}}" alt="${{escapeHtml(page.title)}}">${{elementMarkup}}${{boxMarkup}}`;
      applyOverlayMode();

      document.getElementById("field-body").innerHTML = fieldEntries.length
        ? fieldEntries.map(([key, value]) => `<tr><th>${{escapeHtml(key)}}</th><td>${{escapeHtml(value)}}</td></tr>`).join("")
        : `<tr><td colspan="2">No extracted fields</td></tr>`;

      const timing = timingRows(page);
      document.getElementById("timing-body").innerHTML = timing.length
        ? timing.map(([key, value]) => `<tr><th>${{escapeHtml(key)}}</th><td>${{escapeHtml(value)}}</td></tr>`).join("")
        : `<tr><td colspan="2">No timing metadata</td></tr>`;

      const warnings = page.warnings || [];
      const warningNode = document.getElementById("warnings");
      warningNode.hidden = warnings.length === 0;
      warningNode.innerHTML = warnings.map((warning) => `<li>${{escapeHtml(warning)}}</li>`).join("");

      document.getElementById("element-list").innerHTML = elements.map((element) => {{
        const blockCount = (element.child_block_ids || []).length;
        return `
          <button class="element-row" type="button" data-element-id="${{escapeHtml(element.id)}}" onclick="selectElement('${{escapeHtml(element.id)}}')">
            <span class="block-text">${{escapeHtml(element.text)}}</span>
            <span class="block-meta">${{escapeHtml(element.type)}} · ${{blockCount}}</span>
          </button>
        `;
      }}).join("");

      document.getElementById("block-list").innerHTML = blocks.map((block) => {{
        const confidence = block.confidence === null || block.confidence === undefined ? "" : Number(block.confidence).toFixed(2);
        return `
          <button class="block-row" type="button" data-block-id="${{escapeHtml(block.id)}}" onclick="selectBlock('${{escapeHtml(block.id)}}')">
            <span class="block-text">${{escapeHtml(block.text)}}</span>
            <span class="block-meta">${{escapeHtml(confidence)}}</span>
          </button>
        `;
      }}).join("");
    }}

    renderPage(0);
  </script>
</body>
</html>
"""


def render_viewer_html(layout: dict[str, Any]) -> str:
    page = layout.get("page", {})
    page_width = page.get("width")
    page_height = page.get("height")
    blocks = layout.get("blocks", [])
    elements = _layout_elements_for_render(layout)
    fields = layout.get("fields", {})
    processing = layout.get("processing", {})
    warnings = layout.get("warnings", [])
    image_src = html.escape(_image_src_for_html(str(layout.get("image_path", ""))), quote=True)
    title = html.escape(str(layout.get("image_id", "document")))

    element_box_markup = []
    element_row_markup = []
    for element in elements:
        element_id = html.escape(str(element.get("id", "")), quote=True)
        element_type = html.escape(str(element.get("type", "element")))
        text = html.escape(str(element.get("text", "")))
        bbox = element.get("bbox", {})
        style = "; ".join(
            [
                f"left: {_format_percent(bbox.get('x', 0), page_width)}",
                f"top: {_format_percent(bbox.get('y', 0), page_height)}",
                f"width: {_format_percent(bbox.get('width', 0), page_width)}",
                f"height: {_format_percent(bbox.get('height', 0), page_height)}",
            ]
        )
        child_count = len(element.get("child_block_ids", []) or [])
        element_box_markup.append(
            f'<button class="element-box" style="{style}" data-element-id="{element_id}" '
            f'aria-label="{element_type}: {text}" onclick="selectElement(\'{element_id}\')"></button>'
        )
        element_row_markup.append(
            f'<button class="element-row" data-element-id="{element_id}" '
            f'onclick="selectElement(\'{element_id}\')">'
            f'<span class="block-text">{text}</span>'
            f'<span class="block-meta">{element_type} · {child_count}</span>'
            "</button>"
        )

    box_markup = []
    row_markup = []
    for block in blocks:
        block_id = html.escape(str(block.get("id", "")), quote=True)
        text = html.escape(str(block.get("text", "")))
        bbox = block.get("bbox", {})
        style = "; ".join(
            [
                f"left: {_format_percent(bbox.get('x', 0), page_width)}",
                f"top: {_format_percent(bbox.get('y', 0), page_height)}",
                f"width: {_format_percent(bbox.get('width', 0), page_width)}",
                f"height: {_format_percent(bbox.get('height', 0), page_height)}",
            ]
        )
        confidence = block.get("confidence")
        confidence_text = "" if confidence is None else f"{float(confidence):.2f}"
        box_markup.append(
            f'<button class="ocr-box" style="{style}" data-block-id="{block_id}" '
            f'aria-label="{text}" onclick="selectBlock(\'{block_id}\')"></button>'
        )
        row_markup.append(
            f'<button class="block-row" data-block-id="{block_id}" '
            f'onclick="selectBlock(\'{block_id}\')">'
            f'<span class="block-text">{text}</span>'
            f'<span class="block-meta">{html.escape(confidence_text)}</span>'
            "</button>"
        )

    field_markup = []
    for key, value in fields.items():
        if value is None:
            continue
        field_markup.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
        )

    timing_markup = []
    source_elapsed = processing.get("source_elapsed_seconds", {})
    if isinstance(source_elapsed, dict):
        for key, value in source_elapsed.items():
            if value is None:
                continue
            timing_markup.append(
                "<tr>"
                f"<th>{html.escape(str(key))}</th>"
                f"<td>{float(value):.2f}s</td>"
                "</tr>"
            )
    if processing.get("layout_elapsed_seconds") is not None:
        timing_markup.append(
            "<tr>"
            "<th>layout</th>"
            f"<td>{float(processing['layout_elapsed_seconds']):.2f}s</td>"
            "</tr>"
        )
    timing_section = ""
    if timing_markup:
        timing_section = (
            '<section class="timing">'
            "<h2>Processing Time</h2>"
            f"<table><tbody>{''.join(timing_markup)}</tbody></table>"
            "</section>"
        )

    warning_markup = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings)
    warning_section = f'<ul class="warnings">{warning_markup}</ul>' if warning_markup else ""
    sources_label = html.escape(", ".join(layout.get("sources", [])) or "local parser")
    extracted_field_count = sum(1 for value in fields.values() if value is not None)
    serialized_line_count = len(str(layout.get("serialized_text", "")).splitlines())

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} parse layout</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef2f7;
      color: #18202f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, #f8fafc 0%, #eef2f7 44%, #e8edf5 100%);
    }}
    .parser-shell {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) 400px;
      grid-template-rows: auto 1fr;
      gap: 14px;
      min-height: 100vh;
      padding: 16px;
    }}
    .pipeline {{
      grid-column: 1 / -1;
      background: #ffffff;
      border: 1px solid #d8e0ec;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 12px 32px rgba(31, 44, 71, 0.08);
    }}
    .pipeline-header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 14px 16px 10px;
      border-bottom: 1px solid #e6ebf2;
    }}
    .pipeline-title {{
      display: grid;
      gap: 2px;
    }}
    .pipeline-title h1 {{
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 750;
    }}
    .pipeline-title span, .pipeline-meta {{
      color: #607087;
      font-size: 12px;
    }}
    .pipeline-rail {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0;
      padding: 14px 16px 16px;
    }}
    .pipeline-step {{
      position: relative;
      display: grid;
      gap: 8px;
      min-height: 96px;
      padding: 14px;
      border: 1px solid #dbe3ef;
      border-right: 0;
      background: #fbfcfe;
    }}
    .pipeline-step:first-child {{ border-radius: 8px 0 0 8px; }}
    .pipeline-step:last-child {{
      border-right: 1px solid #dbe3ef;
      border-radius: 0 8px 8px 0;
    }}
    .pipeline-step:not(:last-child)::after {{
      content: "";
      position: absolute;
      top: 50%;
      right: -8px;
      width: 16px;
      height: 16px;
      transform: translateY(-50%) rotate(45deg);
      border-top: 1px solid #dbe3ef;
      border-right: 1px solid #dbe3ef;
      background: #fbfcfe;
      z-index: 2;
    }}
    .step-kicker {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: #607087;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .step-dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #2f6fed;
    }}
    .pipeline-step:nth-child(2) .step-dot {{ background: #00a18a; }}
    .pipeline-step:nth-child(3) .step-dot {{ background: #b86b00; }}
    .pipeline-step:nth-child(4) .step-dot {{ background: #cf2f64; }}
    .step-name {{
      margin: 0;
      font-size: 15px;
      font-weight: 750;
    }}
    .step-value {{
      color: #334155;
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .document-pane, .parsed-pane {{
      min-width: 0;
      background: #ffffff;
      border: 1px solid #d8e0ec;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 12px 32px rgba(31, 44, 71, 0.08);
    }}
    .document-header, .parsed-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-height: 48px;
      padding: 0 16px;
      border-bottom: 1px solid #e6ebf2;
      font-size: 14px;
      font-weight: 650;
    }}
    .document-header span:last-child, .parsed-header span:last-child {{
      color: #607087;
      font-size: 12px;
      font-weight: 650;
    }}
    .stage {{
      position: relative;
      width: min(100%, 980px);
      margin: 16px auto;
      background: #f1f5f9;
      border: 1px solid #e1e7f0;
      border-radius: 8px;
      overflow: hidden;
    }}
    .stage img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .element-box, .ocr-box {{
      position: absolute;
      appearance: none;
      padding: 0;
      cursor: pointer;
    }}
    .element-box {{
      border: 1.5px solid rgba(0, 161, 138, 0.58);
      background: rgba(0, 161, 138, 0.025);
      border-radius: 5px;
      z-index: 3;
    }}
    .element-box:hover, .element-box.active {{
      border-color: #b86b00;
      background: rgba(184, 107, 0, 0.08);
      box-shadow: 0 0 0 2px rgba(184, 107, 0, 0.1);
    }}
    .ocr-box {{
      border: 1px solid rgba(47, 111, 237, 0.55);
      background: transparent;
      border-radius: 3px;
      display: none;
      opacity: 0.7;
      z-index: 2;
    }}
    .stage.show-ocr-lines .ocr-box {{ display: block; }}
    .stage.elements-only .ocr-box {{ display: none; }}
    .ocr-box:hover, .ocr-box.active {{
      border-color: #d91f4c;
      background: rgba(217, 31, 76, 0.08);
      box-shadow: 0 0 0 2px rgba(217, 31, 76, 0.1);
    }}
    .overlay-controls {{
      display: inline-flex;
      gap: 4px;
      padding: 3px;
      border: 1px solid #d8e0ec;
      border-radius: 7px;
      background: #f8fafc;
    }}
    .overlay-toggle {{
      min-height: 30px;
      padding: 0 10px;
      border: 0;
      border-radius: 5px;
      background: transparent;
      color: #607087;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .overlay-toggle.active {{
      background: #ffffff;
      color: #18202f;
      box-shadow: 0 1px 3px rgba(31, 44, 71, 0.12);
    }}
    .parsed-pane {{
      display: grid;
      grid-template-rows: auto auto auto auto minmax(120px, 1fr) minmax(120px, 1fr);
      max-height: calc(100vh - 148px);
    }}
    .section-title {{
      margin: 0;
      padding: 12px 12px 8px;
      color: #607087;
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }}
    .field-section, .timing {{
      border-bottom: 1px solid #eef2f7;
    }}
    .fields {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .fields th, .fields td {{
      padding: 9px 12px;
      border-bottom: 1px solid #eef1f5;
      text-align: left;
      vertical-align: top;
    }}
    .fields th {{
      width: 116px;
      color: #607087;
      font-weight: 650;
    }}
    .timing {{
      padding: 0 12px 10px;
      border-bottom: 1px solid #eef1f5;
    }}
    .timing h2 {{
      margin: 0;
      padding: 12px 0 6px;
      font-size: 12px;
      font-weight: 700;
      color: #607087;
      text-transform: uppercase;
    }}
    .timing table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .timing th, .timing td {{
      padding: 4px 0;
      text-align: left;
      vertical-align: top;
    }}
    .timing th {{
      color: #697588;
      font-weight: 600;
    }}
    .timing td {{
      font-variant-numeric: tabular-nums;
      text-align: right;
    }}
    .element-list, .block-list {{
      overflow: auto;
      padding: 8px;
    }}
    .element-section, .block-section {{
      min-height: 0;
      overflow: hidden;
    }}
    .element-row, .block-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      width: 100%;
      min-height: 40px;
      margin: 0 0 6px;
      padding: 9px 10px;
      border: 1px solid #dfe5ee;
      border-radius: 6px;
      background: #fbfcfe;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }}
    .element-row:hover, .element-row.active {{
      border-color: #00a18a;
      background: #f0fffb;
    }}
    .block-row:hover, .block-row.active {{
      border-color: #d91f4c;
      background: #fff5f7;
    }}
    .block-text {{
      overflow-wrap: anywhere;
      line-height: 1.35;
      font-size: 13px;
    }}
    .block-meta {{
      color: #697588;
      font-size: 12px;
      white-space: nowrap;
    }}
    .warnings {{
      margin: 0;
      padding: 10px 16px 10px 28px;
      border-bottom: 1px solid #eef1f5;
      color: #9b3a13;
      background: #fff7ed;
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      .parser-shell {{ grid-template-columns: 1fr; padding: 10px; }}
      .pipeline-rail {{ grid-template-columns: 1fr; }}
      .pipeline-step, .pipeline-step:first-child, .pipeline-step:last-child {{
        border: 1px solid #dbe3ef;
        border-radius: 8px;
      }}
      .pipeline-step:not(:last-child)::after {{ display: none; }}
      .parsed-pane {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <main class="parser-shell">
    <section class="pipeline">
      <header class="pipeline-header">
        <div class="pipeline-title">
          <h1>Document Parse Pipeline</h1>
          <span>{title}</span>
        </div>
        <div class="pipeline-meta">{sources_label}</div>
      </header>
      <div class="pipeline-rail">
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>AI Model</span></div>
          <p class="step-name">Detector</p>
          <div class="step-value">{len(elements)} layout regions</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>OCR</span></div>
          <p class="step-name">Recognizer</p>
          <div class="step-value">{len([block for block in blocks if str(block.get('text', '')).strip()])} text lines</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>2D to 1D</span></div>
          <p class="step-name">Serializer</p>
          <div class="step-value">{serialized_line_count} ordered lines</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>JSON</span></div>
          <p class="step-name">Parser</p>
          <div class="step-value">{extracted_field_count} extracted fields</div>
        </article>
      </div>
    </section>
    <section class="document-pane">
      <header class="document-header">
        <span>Detected Layout</span>
        <div class="overlay-controls" aria-label="Overlay mode">
          <button class="overlay-toggle active" type="button" data-overlay-mode="elements" onclick="setOverlayMode('elements')">Elements</button>
          <button class="overlay-toggle" type="button" data-overlay-mode="lines" onclick="setOverlayMode('lines')">OCR Lines</button>
        </div>
        <span>{len(elements)} elements · {len(blocks)} lines</span>
      </header>
      <div class="stage elements-only">
        <img src="{image_src}" alt="{title}">
        {''.join(element_box_markup)}
        {''.join(box_markup)}
      </div>
    </section>
    <aside class="parsed-pane">
      <header class="parsed-header"><span>Parser Output</span><span>{sources_label}</span></header>
      {warning_section}
      <section class="field-section">
        <h2 class="section-title">Key-Value Output</h2>
        <table class="fields"><tbody>{''.join(field_markup)}</tbody></table>
      </section>
      {timing_section}
      <section class="element-section">
        <h2 class="section-title">Layout Elements</h2>
        <div class="element-list">{''.join(element_row_markup)}</div>
      </section>
      <section class="block-section">
        <h2 class="section-title">OCR Lines</h2>
        <div class="block-list">{''.join(row_markup)}</div>
      </section>
    </aside>
  </main>
  <script>
    var overlayMode = 'elements';
    function applyOverlayMode() {{
      var stage = document.querySelector('.stage');
      if (!stage) return;
      stage.classList.toggle('elements-only', overlayMode === 'elements');
      stage.classList.toggle('show-ocr-lines', overlayMode === 'lines');
      document.querySelectorAll('[data-overlay-mode]').forEach(function (node) {{
        node.classList.toggle('active', node.dataset.overlayMode === overlayMode);
      }});
    }}
    function setOverlayMode(mode) {{
      overlayMode = mode === 'lines' ? 'lines' : 'elements';
      applyOverlayMode();
    }}
    function selectBlock(blockId) {{
      document.querySelectorAll('[data-block-id]').forEach(function (node) {{
        node.classList.toggle('active', node.dataset.blockId === blockId);
      }});
      setOverlayMode('lines');
      var row = document.querySelector('.block-row[data-block-id="' + blockId + '"]');
      if (row) row.scrollIntoView({{ block: 'nearest' }});
    }}
    function selectElement(elementId) {{
      document.querySelectorAll('[data-element-id]').forEach(function (node) {{
        node.classList.toggle('active', node.dataset.elementId === elementId);
      }});
      var row = document.querySelector('.element-row[data-element-id="' + elementId + '"]');
      if (row) row.scrollIntoView({{ block: 'nearest' }});
    }}
    applyOverlayMode();
  </script>
</body>
</html>
"""
