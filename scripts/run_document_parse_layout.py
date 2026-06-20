#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from scripts.document_parse_layout import (
        build_layout_document,
        normalize_paddleocr_text_payload,
        read_image_size,
        render_viewer_html,
    )
    from scripts.ocr_workspace import build_inventory
except ModuleNotFoundError:
    from document_parse_layout import (
        build_layout_document,
        normalize_paddleocr_text_payload,
        read_image_size,
        render_viewer_html,
    )
    from ocr_workspace import build_inventory


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_blocks_from_baselines(image_id: str, baselines_dir: Path) -> tuple[list[Any], list[str], list[str]]:
    baseline_dir = baselines_dir / image_id
    blocks = []
    sources = []
    warnings = []

    text_path = baseline_dir / "paddleocr-text.json"
    if text_path.exists():
        payload = read_json(text_path)
        blocks.extend(normalize_paddleocr_text_payload(payload))
        sources.append("paddleocr-text")
    else:
        warnings.append(f"Missing PaddleOCR text baseline: {text_path}")

    vl_path = baseline_dir / "paddleocr-vl.json"
    if vl_path.exists():
        sources.append("paddleocr-vl")

    qwen_path = baseline_dir / "qwen3-vl.json"
    if qwen_path.exists():
        sources.append("qwen3-vl")

    if not blocks:
        warnings.append("No coordinate blocks were extracted. Run the paddleocr-text baseline for clickable boxes.")

    return blocks, sources, warnings


def parse_one(
    image_id: str,
    image_path: Path,
    baselines_dir: Path,
    output_dir: Path,
) -> Path:
    blocks, sources, warnings = load_blocks_from_baselines(image_id, baselines_dir)
    target_dir = output_dir / image_id
    target_dir.mkdir(parents=True, exist_ok=True)
    viewer_image_path = target_dir / image_path.name
    if image_path.exists() and image_path.resolve() != viewer_image_path.resolve():
        shutil.copy2(image_path, viewer_image_path)

    layout = build_layout_document(
        image_id=image_id,
        image_path=Path(viewer_image_path.name),
        blocks=blocks,
        page_size=read_image_size(image_path),
        sources=sources,
        warnings=warnings,
    )
    layout["source_image_path"] = str(image_path)

    layout_path = target_dir / "layout.json"
    viewer_path = target_dir / "viewer.html"
    write_json(layout_path, layout)
    viewer_path.write_text(render_viewer_html(layout), encoding="utf-8")
    return layout_path


def parse_inventory(inventory_path: Path, baselines_dir: Path, output_dir: Path) -> list[Path]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    outputs = []
    for item in inventory:
        outputs.append(
            parse_one(
                image_id=str(item["image_id"]),
                image_path=Path(str(item["path"])),
                baselines_dir=baselines_dir,
                output_dir=output_dir,
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local clickable document parse layouts.")
    parser.add_argument("--image-id")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--raw-dir", default="data/private/raw", type=Path)
    parser.add_argument("--baselines-dir", default="results/baselines", type=Path)
    parser.add_argument("--output-dir", default="results/document_parse", type=Path)
    args = parser.parse_args()

    if args.inventory:
        outputs = parse_inventory(args.inventory, args.baselines_dir, args.output_dir)
    elif args.image_id and args.image:
        outputs = [parse_one(args.image_id, args.image, args.baselines_dir, args.output_dir)]
    else:
        inventory = build_inventory(args.raw_dir)
        outputs = []
        for item in inventory:
            outputs.append(
                parse_one(
                    image_id=str(item["image_id"]),
                    image_path=Path(str(item["path"])),
                    baselines_dir=args.baselines_dir,
                    output_dir=args.output_dir,
                )
            )

    print(json.dumps([str(path) for path in outputs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
