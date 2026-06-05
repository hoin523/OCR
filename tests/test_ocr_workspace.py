from pathlib import Path

from scripts.ocr_workspace import (
    build_inventory,
    ensure_private_workspace,
    infer_document_hint,
    make_image_id,
)


def test_ensure_private_workspace_creates_expected_directories(tmp_path: Path):
    ensure_private_workspace(tmp_path)

    expected = [
        "data/private/raw",
        "data/private/labels/raw_text",
        "data/private/labels/fields",
        "data/private/labels/boxes",
        "data/private/splits",
        "results/baselines",
    ]
    for relative in expected:
        assert (tmp_path / relative).is_dir()


def test_make_image_id_is_stable_and_filesystem_safe():
    first = make_image_id("KakaoTalk Photo 001.png", b"same-bytes")
    second = make_image_id("KakaoTalk Photo 001.png", b"same-bytes")

    assert first == second
    assert first.startswith("kakaotalk_photo_001_")
    assert " " not in first


def test_make_image_id_falls_back_for_non_ascii_names():
    image_id = make_image_id("영수증.png", b"same-bytes")

    assert image_id.startswith("document_")
    assert image_id.endswith("_7ad5509f")


def test_infer_document_hint_from_filename():
    assert infer_document_hint("receipt_000001") == "receipt"
    assert infer_document_hint("invoice_000001") == "invoice"
    assert infer_document_hint("transaction_000001") == "transaction"
    assert infer_document_hint("random_file") == "unknown"


def test_build_inventory_discovers_supported_images_with_metadata(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    first = raw / "receipt 001.png"
    second = raw / "invoice_001.JPG"
    ignored = raw / "notes.txt"
    first.write_bytes(b"receipt")
    second.write_bytes(b"invoice")
    ignored.write_text("skip", encoding="utf-8")

    inventory = build_inventory(raw)

    assert len(inventory) == 2
    assert inventory[0]["document_hint"] == "invoice"
    assert inventory[0]["original_name"] == "invoice_001.JPG"
    assert inventory[0]["path"] == str(second)
    assert inventory[0]["size_bytes"] == len(b"invoice")
    assert inventory[1]["document_hint"] == "receipt"
    assert inventory[1]["original_name"] == "receipt 001.png"
