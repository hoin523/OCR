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
