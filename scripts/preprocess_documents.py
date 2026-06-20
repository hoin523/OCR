#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_DOCUMENT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


def collect_document_inputs(input_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
        ),
        key=lambda path: str(path).lower(),
    )


def resized_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max(width, height) <= max_side:
        return width, height
    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def output_image_path(source_path: Path, output_dir: Path, page_index: int | None = None) -> Path:
    if page_index is None:
        name = f"{source_path.stem}.webp"
    else:
        name = f"{source_path.stem}_page_{page_index:03d}.webp"
    return output_dir / name


def _load_pillow():
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required. Run with: uv run --with pillow python scripts/preprocess_documents.py") from exc
    return Image, ImageOps


def convert_image_to_webp(
    source_path: Path,
    output_path: Path,
    max_side: int = 1536,
    quality: int = 88,
) -> dict[str, Any]:
    Image, ImageOps = _load_pillow()
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        original_width, original_height = image.size
        target_width, target_height = resized_dimensions(original_width, original_height, max_side)
        if (target_width, target_height) != image.size:
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        image.save(output_path, "WEBP", quality=quality, method=6)

    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "source_type": "image",
        "original_size": {"width": original_width, "height": original_height},
        "output_size": {"width": target_width, "height": target_height},
        "quality": quality,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _poppler_env(poppler_bin: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if poppler_bin:
        env["PATH"] = f"{poppler_bin}:{env.get('PATH', '')}"
    return env


def render_pdf_pages_to_webp(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    max_side: int = 1536,
    quality: int = 88,
    poppler_bin: Path | None = None,
) -> list[dict[str, Any]]:
    if not shutil.which("pdftoppm", path=_poppler_env(poppler_bin).get("PATH")):
        raise RuntimeError("pdftoppm is required for PDF preprocessing. Install Poppler or pass --poppler-bin.")

    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / pdf_path.stem
        command = ["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(prefix)]
        subprocess.run(command, check=True, env=_poppler_env(poppler_bin))
        page_images = sorted(Path(tmp).glob(f"{pdf_path.stem}-*.png"))

        results = []
        for index, page_image in enumerate(page_images, start=1):
            output_path = output_image_path(pdf_path, output_dir, page_index=index)
            item = convert_image_to_webp(page_image, output_path, max_side=max_side, quality=quality)
            item["source_path"] = str(pdf_path)
            item["source_type"] = "pdf_page"
            item["page_index"] = index
            item["dpi"] = dpi
            results.append(item)
        return results


def preprocess_documents(
    input_dir: Path,
    output_dir: Path,
    max_side: int = 1536,
    quality: int = 88,
    dpi: int = 200,
    poppler_bin: Path | None = None,
) -> list[dict[str, Any]]:
    manifest = []
    for source_path in collect_document_inputs(input_dir):
        if source_path.suffix.lower() == ".pdf":
            manifest.extend(
                render_pdf_pages_to_webp(
                    source_path,
                    output_dir,
                    dpi=dpi,
                    max_side=max_side,
                    quality=quality,
                    poppler_bin=poppler_bin,
                )
            )
        else:
            manifest.append(
                convert_image_to_webp(
                    source_path,
                    output_image_path(source_path, output_dir),
                    max_side=max_side,
                    quality=quality,
                )
            )
    return manifest


def write_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "preprocess_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess large images and PDFs into OCR-friendly WebP pages.")
    parser.add_argument("--input-dir", default="data/private/raw", type=Path)
    parser.add_argument("--output-dir", default="data/private/preprocessed", type=Path)
    parser.add_argument("--max-side", default=1536, type=int)
    parser.add_argument("--quality", default=88, type=int)
    parser.add_argument("--pdf-dpi", default=200, type=int)
    parser.add_argument("--poppler-bin", type=Path)
    args = parser.parse_args()

    manifest = preprocess_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_side=args.max_side,
        quality=args.quality,
        dpi=args.pdf_dpi,
        poppler_bin=args.poppler_bin,
    )
    manifest_path = write_manifest(args.output_dir, manifest)
    print(json.dumps({"manifest": str(manifest_path), "items": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
