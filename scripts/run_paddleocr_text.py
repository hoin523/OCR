#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

from paddleocr import PaddleOCR


def result_to_dict(result):
    if hasattr(result, "json"):
        return result.json
    if hasattr(result, "to_json"):
        return result.to_json()
    if isinstance(result, dict):
        return result
    return {"repr": repr(result)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lang", default="korean")
    args = parser.parse_args()

    started = time.time()
    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    results = ocr.predict(str(args.image))
    elapsed = time.time() - started

    payload = {
        "model": f"PaddleOCR text ({args.lang})",
        "image": str(args.image),
        "elapsed_seconds": round(elapsed, 3),
        "results": [result_to_dict(result) for result in results],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
