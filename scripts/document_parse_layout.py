from __future__ import annotations

import html
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
    serialized_text = serialize_blocks(ordered_blocks)
    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "page": page_size,
        "blocks": [layout_block_to_dict(block) for block in ordered_blocks],
        "serialized_text": serialized_text,
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


def render_viewer_html(layout: dict[str, Any]) -> str:
    page = layout.get("page", {})
    page_width = page.get("width")
    page_height = page.get("height")
    blocks = layout.get("blocks", [])
    fields = layout.get("fields", {})
    processing = layout.get("processing", {})
    warnings = layout.get("warnings", [])
    image_src = html.escape(_image_src_for_html(str(layout.get("image_path", ""))), quote=True)
    title = html.escape(str(layout.get("image_id", "document")))

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
    .ocr-box {{
      position: absolute;
      appearance: none;
      border: 2px solid rgba(47, 111, 237, 0.78);
      background: rgba(47, 111, 237, 0.1);
      border-radius: 3px;
      padding: 0;
      cursor: pointer;
    }}
    .ocr-box:hover, .ocr-box.active {{
      border-color: #d91f4c;
      background: rgba(217, 31, 76, 0.18);
      box-shadow: 0 0 0 2px rgba(217, 31, 76, 0.16);
    }}
    .parsed-pane {{
      display: grid;
      grid-template-rows: auto auto auto auto minmax(0, 1fr);
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
    .block-list {{
      overflow: auto;
      padding: 8px;
    }}
    .block-section {{
      min-height: 0;
      overflow: hidden;
    }}
    .block-row {{
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
          <div class="step-value">{len(blocks)} layout regions</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>OCR</span></div>
          <p class="step-name">Recognizer</p>
          <div class="step-value">{len([block for block in blocks if str(block.get('text', '')).strip()])} text lines</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>2D to 1D</span></div>
          <p class="step-name">Serializer</p>
          <div class="step-value">{len(str(layout.get('serialized_text', '')).splitlines())} ordered lines</div>
        </article>
        <article class="pipeline-step">
          <div class="step-kicker"><span class="step-dot"></span><span>JSON</span></div>
          <p class="step-name">Parser</p>
          <div class="step-value">{extracted_field_count} extracted fields</div>
        </article>
      </div>
    </section>
    <section class="document-pane">
      <header class="document-header"><span>Detected Layout</span><span>{len(blocks)} blocks</span></header>
      <div class="stage">
        <img src="{image_src}" alt="{title}">
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
      <section class="block-section">
        <h2 class="section-title">Layout Blocks</h2>
        <div class="block-list">{''.join(row_markup)}</div>
      </section>
    </aside>
  </main>
  <script>
    function selectBlock(blockId) {{
      document.querySelectorAll('[data-block-id]').forEach(function (node) {{
        node.classList.toggle('active', node.dataset.blockId === blockId);
      }});
      var row = document.querySelector('.block-row[data-block-id="' + blockId + '"]');
      if (row) row.scrollIntoView({{ block: 'nearest' }});
    }}
  </script>
</body>
</html>
"""
