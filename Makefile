RAW_DIR ?= data/private/raw
INVENTORY ?= results/dataset_inventory.json
RESULTS_DIR ?= results/baselines
EXTRACTORS ?= paddleocr-text,paddleocr-vl,qwen3-vl
SMOKE_LIMIT ?= 10
QWEN_MODEL ?= qwen3-vl:8b

.PHONY: setup inventory dry-run smoke baselines test

setup:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors none

inventory:
	uv run python scripts/build_dataset_inventory.py --raw-dir $(RAW_DIR) --output $(INVENTORY)

dry-run:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors $(EXTRACTORS) --qwen-model $(QWEN_MODEL) --dry-run

smoke:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors paddleocr-text --limit $(SMOKE_LIMIT)

baselines:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors $(EXTRACTORS) --qwen-model $(QWEN_MODEL)

test:
	uv run --with pytest python -m pytest -v
