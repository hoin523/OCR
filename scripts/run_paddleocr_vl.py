#!/usr/bin/env python3
import argparse
import json
import tempfile
import time
from pathlib import Path

from paddleocr import PaddleOCRVL


def read_generated_files(directory: Path, suffix: str) -> list[dict[str, object]]:
    files = []
    for path in sorted(directory.glob(f"*{suffix}")):
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            try:
                content: object = json.loads(text)
            except json.JSONDecodeError:
                content = text
        else:
            content = text
        files.append({"name": path.name, "content": content})
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    started = time.time()
    pipeline = PaddleOCRVL()
    results = pipeline.predict(str(args.image))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        result_reprs = []
        for result in results:
            result_reprs.append(repr(result))
            if hasattr(result, "save_to_json"):
                result.save_to_json(save_path=str(tmp_dir))
            if hasattr(result, "save_to_markdown"):
                result.save_to_markdown(save_path=str(tmp_dir))

        payload = {
            "model": "PaddleOCR-VL",
            "image": str(args.image),
            "elapsed_seconds": round(time.time() - started, 3),
            "json_files": read_generated_files(tmp_dir, ".json"),
            "markdown_files": read_generated_files(tmp_dir, ".md"),
            "result_reprs": result_reprs,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
