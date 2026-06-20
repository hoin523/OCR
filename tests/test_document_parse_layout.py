from pathlib import Path

from scripts.document_parse_layout import (
    BoundingBox,
    LayoutBlock,
    build_layout_elements,
    build_layout_document,
    normalize_paddleocr_text_payload,
    parse_receipt_fields,
    read_image_size,
    render_multipage_viewer_html,
    render_viewer_html,
    serialize_blocks,
)
from scripts.run_document_parse_layout import parse_inventory, parse_one


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


def test_build_layout_elements_groups_nearby_lines_into_paragraphs():
    blocks = [
        LayoutBlock("block-001", "text", "출장비 정산 기준", BoundingBox(40, 40, 170, 20), 0.99, "test"),
        LayoutBlock("block-002", "text", "국내 출장은 실비 기준으로 정산한다.", BoundingBox(40, 70, 260, 20), 0.98, "test"),
        LayoutBlock("block-003", "text", "영수증 원본을 첨부해야 한다.", BoundingBox(40, 98, 230, 20), 0.97, "test"),
        LayoutBlock("block-004", "text", "승인권자", BoundingBox(410, 250, 80, 20), 0.96, "test"),
    ]

    elements = build_layout_elements(blocks, {"width": 640, "height": 480})

    assert [element.type for element in elements] == ["paragraph", "paragraph"]
    assert elements[0].child_block_ids == ["block-001", "block-002", "block-003"]
    assert elements[0].text == "출장비 정산 기준\n국내 출장은 실비 기준으로 정산한다.\n영수증 원본을 첨부해야 한다."
    assert elements[0].bbox == BoundingBox(x=40, y=40, width=260, height=78)
    assert elements[0].markdown == "출장비 정산 기준\n국내 출장은 실비 기준으로 정산한다.\n영수증 원본을 첨부해야 한다."


def test_parse_receipt_fields_extracts_total_and_merchant():
    fields = parse_receipt_fields("Nice Store\n공급가 9000\n합계 10,000원")

    assert fields["merchant"] == "Nice Store"
    assert fields["total"] == "10,000원"


def test_parse_receipt_fields_extracts_total_from_nearby_amount_lines():
    fields = parse_receipt_fields(
        "<<<영 수증>>>\n"
        "면세합:\n"
        "0\n"
        "합계액:\n"
        "3,560\n"
        "카드\n"
        "3,560"
    )

    assert fields["merchant"] == "<<<영 수증>>>"
    assert fields["total"] == "3,560"


def test_parse_receipt_fields_ignores_negative_discounts_near_payment_total():
    fields = parse_receipt_fields(
        "롯데쇼핑\n"
        "결제금액\n"
        "할인 상세내역\n"
        "-3,980\n"
        "에누리\n"
        "현금IC\n"
        "3,980"
    )

    assert fields["total"] == "3,980"


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
        processing={"source_elapsed_seconds": {"test": 1.234}},
    )

    assert layout["image_id"] == "receipt_abc"
    assert layout["page"] == {"width": 640, "height": 480}
    assert layout["serialized_text"] == "Nice Store\n합계 10,000원"
    assert layout["serialized_markdown"] == "Nice Store\n\n합계 10,000원"
    assert layout["elements"][0]["child_block_ids"] == ["block-001"]
    assert layout["fields"]["total"] == "10,000원"
    assert layout["blocks"][0]["bbox"]["width"] == 100
    assert layout["processing"]["source_elapsed_seconds"]["test"] == 1.234


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
        "processing": {
            "source_elapsed_seconds": {"paddleocr-text": 12.346},
            "total_source_elapsed_seconds": 12.346,
            "layout_elapsed_seconds": 0.012,
        },
        "warnings": [],
    }

    html = render_viewer_html(layout)

    assert 'data-element-id=' in html
    assert 'data-block-id="block-001"' in html
    assert 'class="stage elements-only"' in html
    assert ".stage.elements-only .ocr-box" in html
    assert "setOverlayMode" in html
    assert "show-ocr-lines" in html
    assert "selectElement" in html
    assert "selectBlock" in html
    assert "Nice Store" in html
    assert "left: 1.5625%" in html
    assert "Processing Time" in html
    assert "12.35s" in html


def test_render_viewer_html_has_parser_dashboard_structure():
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
        "fields": {"merchant": "Nice Store", "total": "10,000"},
        "sources": ["paddleocr-text"],
        "processing": {
            "source_elapsed_seconds": {"paddleocr-text": 12.346},
            "layout_elapsed_seconds": 0.012,
        },
        "warnings": [],
    }

    html = render_viewer_html(layout)

    assert 'class="parser-shell"' in html
    assert "Document Parse Pipeline" in html
    assert "Detector" in html
    assert "Recognizer" in html
    assert "Serializer" in html
    assert "Parser" in html
    assert "Key-Value Output" in html
    assert "Layout Elements" in html
    assert "OCR Lines" in html


def test_render_multipage_viewer_html_embeds_pages_and_navigation():
    layouts = [
        {
            "image_id": "policy_page_001",
            "image_path": "policy_page_001/policy_page_001.webp",
            "page": {"width": 640, "height": 480},
            "blocks": [
                {
                    "id": "block-001",
                    "type": "text",
                    "text": "Page One",
                    "confidence": 0.99,
                    "source": "test",
                    "bbox": {"x": 10, "y": 20, "width": 100, "height": 30},
                }
            ],
            "serialized_text": "Page One",
            "fields": {"merchant": "Page One"},
            "sources": ["paddleocr-text"],
            "processing": {"source_elapsed_seconds": {"paddleocr-text": 1.25}},
            "warnings": [],
        },
        {
            "image_id": "policy_page_002",
            "image_path": "policy_page_002/policy_page_002.webp",
            "page": {"width": 640, "height": 480},
            "blocks": [],
            "serialized_text": "",
            "fields": {},
            "sources": ["paddleocr-text"],
            "processing": {},
            "warnings": [],
        },
    ]

    html = render_multipage_viewer_html(layouts, title="Policy Document")

    assert "Policy Document" in html
    assert "policy_page_001" in html
    assert "policy_page_002" in html
    assert "page-tabs" in html
    assert "element-box" in html
    assert 'id="stage" class="stage elements-only"' in html
    assert "setOverlayMode" in html
    assert "selectElement" in html
    assert "goPage(-1)" in html
    assert "goPage(1)" in html
    assert "renderPage(0)" in html
    assert "Page One" in html
    assert "policy_page_001/policy_page_001.webp" in html


def test_read_image_size_supports_lossy_webp(tmp_path: Path):
    webp = tmp_path / "sample.webp"
    vp8_payload = b"\x00\x00\x00\x9d\x01\x2a\x80\x04\x00\x06"
    webp.write_bytes(b"RIFF" + (18).to_bytes(4, "little") + b"WEBP" + b"VP8 " + (10).to_bytes(4, "little") + vp8_payload)

    assert read_image_size(webp) == {"width": 1152, "height": 1536}


def test_parse_one_writes_layout_and_viewer_from_baseline(tmp_path: Path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fake")
    baseline = tmp_path / "baselines" / "receipt_abc" / "paddleocr-text.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        '{"elapsed_seconds":7.891,"results":[{"res":{"dt_polys":[[[0,0],[10,0],[10,10],[0,10]]],"rec_texts":["Store"],"rec_scores":[0.9]}}]}',
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
    assert '"paddleocr-text": 7.891' in layout_path.read_text(encoding="utf-8")
    assert (output / "receipt_abc" / "receipt.png").exists()
    viewer = (output / "receipt_abc" / "viewer.html")
    assert viewer.exists()
    assert 'src="receipt.png"' in viewer.read_text(encoding="utf-8")


def test_parse_inventory_writes_multipage_index(tmp_path: Path):
    inventory = []
    baselines = tmp_path / "baselines"
    for page in (1, 2):
        image_id = f"doc_page_{page:03d}"
        image = tmp_path / f"{image_id}.png"
        image.write_bytes(b"fake")
        baseline = baselines / image_id / "paddleocr-text.json"
        baseline.parent.mkdir(parents=True)
        baseline.write_text(
            '{"elapsed_seconds":1.5,"results":[{"res":{"dt_polys":[[[0,0],[10,0],[10,10],[0,10]]],"rec_texts":["Page"],"rec_scores":[0.9]}}]}',
            encoding="utf-8",
        )
        inventory.append({"image_id": image_id, "path": str(image)})

    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(__import__("json").dumps(inventory), encoding="utf-8")
    output = tmp_path / "document_parse"

    parse_inventory(inventory_path, baselines, output)

    index = output / "index.html"
    assert index.exists()
    html = index.read_text(encoding="utf-8")
    assert "doc_page_001" in html
    assert "doc_page_002" in html
    assert "doc_page_001/doc_page_001.png" in html
    assert "doc_page_002/doc_page_002.png" in html
