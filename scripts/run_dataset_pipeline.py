#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

try:
    from scripts.ocr_workspace import build_inventory, ensure_private_workspace, write_inventory
except ModuleNotFoundError:
    from ocr_workspace import build_inventory, ensure_private_workspace, write_inventory


SUPPORTED_EXTRACTORS = {"paddleocr-text", "paddleocr-vl", "qwen3-vl"}


def parse_extractors(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized in {"", "none"}:
        return []

    extractors = [item.strip().lower() for item in value.split(",") if item.strip()]
    for extractor in extractors:
        if extractor not in SUPPORTED_EXTRACTORS:
            supported = ", ".join(sorted(SUPPORTED_EXTRACTORS))
            raise ValueError(f"Unsupported extractor: {extractor}. Supported: {supported}, none")
    return extractors


def baseline_output_path(results_dir: Path, image_id: str, extractor: str) -> Path:
    return results_dir / image_id / f"{extractor}.json"


def build_extractor_command(
    extractor: str,
    image_path: str,
    output_path: Path,
    qwen_model: str = "qwen3-vl:8b",
    text_det_limit_side_len: int = 1536,
    text_det_limit_type: str = "max",
) -> list[str]:
    if extractor == "paddleocr-text":
        return [
            "uv",
            "run",
            "--python",
            "3.12",
            "--with",
            "paddleocr>=3.6.0",
            "--with",
            "paddlepaddle",
            "python",
            "scripts/run_paddleocr_text.py",
            "--image",
            image_path,
            "--output",
            str(output_path),
            "--lang",
            "korean",
            "--text-det-limit-side-len",
            str(text_det_limit_side_len),
            "--text-det-limit-type",
            text_det_limit_type,
        ]
    if extractor == "qwen3-vl":
        return [
            "uv",
            "run",
            "--with",
            "requests",
            "python",
            "scripts/run_ollama_ocr.py",
            "--model",
            qwen_model,
            "--image",
            image_path,
            "--output",
            str(output_path),
        ]
    if extractor == "paddleocr-vl":
        return [
            "uv",
            "run",
            "--python",
            "3.12",
            "--with",
            "paddleocr[doc-parser]>=3.6.0",
            "--with",
            "paddlepaddle",
            "python",
            "scripts/run_paddleocr_vl.py",
            "--image",
            image_path,
            "--output",
            str(output_path),
        ]
    raise ValueError(f"Unsupported extractor: {extractor}")


def build_extraction_plan(
    inventory: list[dict[str, object]],
    extractors: list[str],
    results_dir: Path,
    skip_existing: bool = True,
    qwen_model: str = "qwen3-vl:8b",
    text_det_limit_side_len: int = 1536,
    text_det_limit_type: str = "max",
) -> list[dict[str, str | list[str]]]:
    plan: list[dict[str, str | list[str]]] = []
    for item in inventory:
        image_id = str(item["image_id"])
        image_path = str(item["path"])
        for extractor in extractors:
            output_path = baseline_output_path(results_dir, image_id, extractor)
            if skip_existing and output_path.exists():
                continue
            plan.append(
                {
                    "image_id": image_id,
                    "extractor": extractor,
                    "output_path": str(output_path),
                    "command": build_extractor_command(
                        extractor=extractor,
                        image_path=image_path,
                        output_path=output_path,
                        qwen_model=qwen_model,
                        text_det_limit_side_len=text_det_limit_side_len,
                        text_det_limit_type=text_det_limit_type,
                    ),
                }
            )
    return plan


def run_extraction_plan(plan: list[dict[str, str | list[str]]]) -> None:
    for index, item in enumerate(plan, start=1):
        command = item["command"]
        if not isinstance(command, list):
            raise TypeError("Pipeline command must be a list of strings")
        print(f"[{index}/{len(plan)}] {item['extractor']} -> {item['output_path']}")
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inventory and run local OCR baselines.")
    parser.add_argument("--raw-dir", default="data/private/raw", type=Path)
    parser.add_argument("--inventory", default="results/dataset_inventory.json", type=Path)
    parser.add_argument("--results-dir", default="results/baselines", type=Path)
    parser.add_argument("--extractors", default="none")
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--qwen-model", default="qwen3-vl:8b")
    parser.add_argument("--text-det-limit-side-len", default=1536, type=int)
    parser.add_argument("--text-det-limit-type", default="max", choices=["min", "max"])
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_private_workspace(Path("."))
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(args.raw_dir)
    write_inventory(inventory, args.inventory)
    print(f"Wrote {len(inventory)} inventory items to {args.inventory}")

    if args.limit > 0:
        inventory = inventory[: args.limit]
        print(f"Limited run to {len(inventory)} item(s)")

    extractors = parse_extractors(args.extractors)
    if not extractors:
        print("No extractors selected. Inventory/workspace setup complete.")
        return

    plan = build_extraction_plan(
        inventory=inventory,
        extractors=extractors,
        results_dir=args.results_dir,
        skip_existing=not args.no_skip_existing,
        qwen_model=args.qwen_model,
        text_det_limit_side_len=args.text_det_limit_side_len,
        text_det_limit_type=args.text_det_limit_type,
    )

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if not plan:
        print("No extraction work to run. Existing outputs were skipped.")
        return

    run_extraction_plan(plan)


if __name__ == "__main__":
    main()
