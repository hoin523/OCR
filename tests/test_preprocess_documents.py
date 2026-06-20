from pathlib import Path

from scripts.preprocess_documents import (
    collect_document_inputs,
    output_image_path,
    write_manifest,
    resized_dimensions,
)


def test_resized_dimensions_preserves_aspect_when_larger_than_max_side():
    assert resized_dimensions(width=3024, height=4032, max_side=1536) == (1152, 1536)


def test_resized_dimensions_keeps_smaller_images_unchanged():
    assert resized_dimensions(width=800, height=600, max_side=1536) == (800, 600)


def test_output_image_path_converts_images_and_pdf_pages_to_webp(tmp_path: Path):
    output_dir = tmp_path / "out"

    assert output_image_path(Path("IMG00764.jpeg"), output_dir) == output_dir / "IMG00764.webp"
    assert output_image_path(Path("receipt.pdf"), output_dir, page_index=2) == output_dir / "receipt_page_002.webp"


def test_collect_document_inputs_finds_supported_images_and_pdfs(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "b.pdf").write_bytes(b"pdf")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.webp").write_bytes(b"webp")

    assert [path.name for path in collect_document_inputs(tmp_path)] == ["a.jpg", "b.pdf", "c.webp"]


def test_write_manifest_records_pdf_page_outputs(tmp_path: Path):
    manifest_path = write_manifest(
        tmp_path,
        [
            {
                "source_path": "receipt.pdf",
                "output_path": "receipt_page_001.webp",
                "source_type": "pdf_page",
                "page_index": 1,
                "elapsed_seconds": 0.1,
            }
        ],
    )

    assert manifest_path == tmp_path / "preprocess_manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    assert '"source_type": "pdf_page"' in text
    assert '"page_index": 1' in text
