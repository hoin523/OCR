#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

PRIVATE_DIRECTORIES = [
    Path("data/private/raw"),
    Path("data/private/labels/raw_text"),
    Path("data/private/labels/fields"),
    Path("data/private/labels/boxes"),
    Path("data/private/splits"),
    Path("results/baselines"),
]


def ensure_private_workspace(root: Path = Path(".")) -> None:
    for relative in PRIVATE_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_image_id(filename: str, file_bytes: bytes) -> str:
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not slug:
        slug = "document"
    digest = hashlib.sha256(file_bytes).hexdigest()[:8]
    return f"{slug}_{digest}"


def infer_document_hint(image_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", image_id.lower()).strip("_")
    prefix = normalized.split("_", 1)[0]
    if prefix in {"receipt", "invoice", "transaction"}:
        return prefix
    return "unknown"


def build_inventory(raw_dir: Path) -> list[dict[str, object]]:
    inventory = []
    for path in sorted(raw_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        content = path.read_bytes()
        image_id = make_image_id(path.name, content)
        inventory.append(
            {
                "image_id": image_id,
                "path": str(path),
                "original_name": path.name,
                "document_hint": infer_document_hint(path.stem),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory


def write_inventory(inventory: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
