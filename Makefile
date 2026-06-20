RAW_DIR ?= data/private/raw
INVENTORY ?= results/dataset_inventory.json
RESULTS_DIR ?= results/baselines
DOCUMENT_PARSE_DIR ?= results/document_parse
PREPROCESSED_DIR ?= data/private/preprocessed
EXTRACTORS ?= paddleocr-text,paddleocr-vl,qwen3-vl
SMOKE_LIMIT ?= 10
QWEN_MODEL ?= qwen3-vl:8b
TEXT_DET_LIMIT_SIDE_LEN ?= 1536
TEXT_DET_LIMIT_TYPE ?= max
PREPROCESS_MAX_SIDE ?= 1536
WEBP_QUALITY ?= 88
PDF_DPI ?= 200
UPLOAD_PORT ?= 8770
PARSER_TEST_PORT ?= 8771
KORIE_DIR ?= data/private/external/KORIE
KORIE_SAMPLE_DIR ?= data/private/raw/korie

.PHONY: setup inventory preprocess dry-run smoke baselines layout layout-smoke upload-app parser-test-app korie-clone korie-sample test

setup:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors none

inventory:
	uv run python scripts/build_dataset_inventory.py --raw-dir $(RAW_DIR) --output $(INVENTORY)

preprocess:
	uv run --with pillow python scripts/preprocess_documents.py --input-dir $(RAW_DIR) --output-dir $(PREPROCESSED_DIR) --max-side $(PREPROCESS_MAX_SIDE) --quality $(WEBP_QUALITY) --pdf-dpi $(PDF_DPI)

dry-run:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors $(EXTRACTORS) --qwen-model $(QWEN_MODEL) --text-det-limit-side-len $(TEXT_DET_LIMIT_SIDE_LEN) --text-det-limit-type $(TEXT_DET_LIMIT_TYPE) --dry-run

smoke:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors paddleocr-text --limit $(SMOKE_LIMIT) --text-det-limit-side-len $(TEXT_DET_LIMIT_SIDE_LEN) --text-det-limit-type $(TEXT_DET_LIMIT_TYPE)

baselines:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors $(EXTRACTORS) --qwen-model $(QWEN_MODEL) --text-det-limit-side-len $(TEXT_DET_LIMIT_SIDE_LEN) --text-det-limit-type $(TEXT_DET_LIMIT_TYPE)

layout:
	uv run python scripts/run_document_parse_layout.py --raw-dir $(RAW_DIR) --baselines-dir $(RESULTS_DIR) --output-dir $(DOCUMENT_PARSE_DIR)

layout-smoke:
	uv run python scripts/run_dataset_pipeline.py --raw-dir $(RAW_DIR) --inventory $(INVENTORY) --results-dir $(RESULTS_DIR) --extractors paddleocr-text --limit $(SMOKE_LIMIT) --text-det-limit-side-len $(TEXT_DET_LIMIT_SIDE_LEN) --text-det-limit-type $(TEXT_DET_LIMIT_TYPE)
	uv run python scripts/run_document_parse_layout.py --inventory $(INVENTORY) --baselines-dir $(RESULTS_DIR) --output-dir $(DOCUMENT_PARSE_DIR)

upload-app:
	uv run --with pillow python scripts/upload_app.py --host 127.0.0.1 --port $(UPLOAD_PORT)

parser-test-app:
	uv run --with pillow python scripts/parser_test_app.py --host 127.0.0.1 --port $(PARSER_TEST_PORT)

korie-clone:
	mkdir -p data/private/external
	test -d $(KORIE_DIR)/.git || git clone https://github.com/MahmoudSalah/KORIE.git $(KORIE_DIR)

korie-sample:
	mkdir -p $(KORIE_SAMPLE_DIR)
	find $(KORIE_DIR) -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | head -5 | while read image; do cp "$$image" $(KORIE_SAMPLE_DIR)/; done

test:
	uv run --with pytest python -m pytest -v
