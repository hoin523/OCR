from pathlib import Path

from scripts.document_parse_layout import (
    BoundingBox,
    LayoutBlock,
    build_layout_document,
    normalize_paddleocr_text_payload,
    parse_receipt_fields,
    render_viewer_html,
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
