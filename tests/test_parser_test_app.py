import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.parser_test_app import PARSER_TEST_HTML, make_handler
from scripts.upload_app import JobStore


def test_parser_test_html_is_parser_only():
    assert "Parser Test" in PARSER_TEST_HTML
    assert "expectedJsonInput" in PARSER_TEST_HTML
    assert "upload-dialog" in PARSER_TEST_HTML
    assert "전결규정" not in PARSER_TEST_HTML
    assert "VectorDB" not in PARSER_TEST_HTML
    assert "approval_rule" not in PARSER_TEST_HTML


def test_parser_test_upload_creates_parser_test_job(tmp_path: Path):
    store = JobStore()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/upload",
            data=b"fake-webp",
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "X-File-Name": "receipt.webp",
            },
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 202
        assert payload["filename"] == "receipt.webp"
        assert payload["document_type"] == "parser_test"
        assert "data/private/uploads" in str(tmp_path / "data/private/uploads" / payload["job_id"] / "raw")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_parser_test_parse_results_api_returns_fields(tmp_path: Path):
    layout = tmp_path / "results/document_parse_uploads/job123/page_001/layout.json"
    layout.parent.mkdir(parents=True)
    layout.write_text(json.dumps({"image_id": "page_001", "fields": {"total": "10,000"}}), encoding="utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, JobStore()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/parse-results/job123", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert payload["merged_fields"] == {"total": "10,000"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
