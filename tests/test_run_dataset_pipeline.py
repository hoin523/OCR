from pathlib import Path

import pytest

from scripts.run_dataset_pipeline import build_extraction_plan, parse_extractors


def test_parse_extractors_accepts_none_and_comma_separated_values():
    assert parse_extractors("none") == []
    assert parse_extractors("paddleocr-text,qwen3-vl") == ["paddleocr-text", "qwen3-vl"]
    assert parse_extractors("paddleocr-text,paddleocr-vl,qwen3-vl") == [
        "paddleocr-text",
        "paddleocr-vl",
        "qwen3-vl",
    ]
    assert parse_extractors(" paddleocr-text , qwen3-vl ") == ["paddleocr-text", "qwen3-vl"]


def test_parse_extractors_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unsupported extractor"):
        parse_extractors("unknown-model")


def test_build_extraction_plan_creates_model_commands(tmp_path: Path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fake-image")
    inventory = [{"image_id": "receipt_abc123", "path": str(image)}]

    plan = build_extraction_plan(
        inventory=inventory,
        extractors=["paddleocr-text", "paddleocr-vl", "qwen3-vl"],
        results_dir=tmp_path / "results",
    )

    assert [item["extractor"] for item in plan] == [
        "paddleocr-text",
        "paddleocr-vl",
        "qwen3-vl",
    ]
    assert plan[0]["output_path"].endswith("results/receipt_abc123/paddleocr-text.json")
    assert "scripts/run_paddleocr_text.py" in plan[0]["command"]
    assert "--lang" in plan[0]["command"]
    assert "korean" in plan[0]["command"]
    assert "--text-det-limit-side-len" in plan[0]["command"]
    assert "1536" in plan[0]["command"]
    assert "--text-det-limit-type" in plan[0]["command"]
    assert "max" in plan[0]["command"]
    assert plan[1]["output_path"].endswith("results/receipt_abc123/paddleocr-vl.json")
    assert "scripts/run_paddleocr_vl.py" in plan[1]["command"]
    assert "paddleocr[doc-parser]>=3.6.0" in plan[1]["command"]
    assert plan[2]["output_path"].endswith("results/receipt_abc123/qwen3-vl.json")
    assert "scripts/run_ollama_ocr.py" in plan[2]["command"]
    assert "qwen3-vl:8b" in plan[2]["command"]


def test_build_extraction_plan_skips_existing_outputs(tmp_path: Path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"fake-image")
    existing = tmp_path / "results" / "receipt_abc123" / "paddleocr-text.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="utf-8")
    inventory = [{"image_id": "receipt_abc123", "path": str(image)}]

    plan = build_extraction_plan(
        inventory=inventory,
        extractors=["paddleocr-text", "qwen3-vl"],
        results_dir=tmp_path / "results",
        skip_existing=True,
    )

    assert [item["extractor"] for item in plan] == ["qwen3-vl"]
