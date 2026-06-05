#!/usr/bin/env python3
import argparse
from pathlib import Path

try:
    from scripts.ocr_workspace import build_inventory, ensure_private_workspace, write_inventory
except ModuleNotFoundError:
    from ocr_workspace import build_inventory, ensure_private_workspace, write_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private OCR dataset inventory.")
    parser.add_argument("--raw-dir", default="data/private/raw", type=Path)
    parser.add_argument("--output", default="results/dataset_inventory.json", type=Path)
    args = parser.parse_args()

    ensure_private_workspace(Path("."))
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(args.raw_dir)
    write_inventory(inventory, args.output)
    print(f"Wrote {len(inventory)} inventory items to {args.output}")


if __name__ == "__main__":
    main()
