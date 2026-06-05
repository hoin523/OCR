#!/usr/bin/env python3
import argparse
import base64
import json
import time
from pathlib import Path

import requests


PROMPT = """Extract the visible OCR information from this screenshot.

Return strict JSON only with these keys:
- document_type: "receipt" or "transaction_detail" or "unknown"
- merchant_or_store
- datetime
- total_amount
- currency
- items: array of objects with name and amount
- payment_method
- account_tail
- category
- raw_text_lines: array of visible text lines in reading order

Preserve original Korean/English text. Do not translate. Use null when unknown."""


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run(model: str, image_path: Path, output_path: Path) -> None:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [image_to_base64(image_path)],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 8192,
        },
    }
    started = time.time()
    response = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=900)
    elapsed = time.time() - started
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        output = {
            "model": model,
            "image": str(image_path),
            "elapsed_seconds": round(elapsed, 3),
            "error": str(exc),
            "response_text": response.text,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        raise
    data = response.json()
    parsed_response = data.get("response") or data.get("thinking", "")
    try:
        parsed_response = json.loads(parsed_response)
    except json.JSONDecodeError:
        pass
    output = {
        "model": model,
        "image": str(image_path),
        "elapsed_seconds": round(elapsed, 3),
        "response": parsed_response,
        "ollama": {k: v for k, v in data.items() if k != "response"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:8b")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.model, args.image, args.output)


if __name__ == "__main__":
    main()
