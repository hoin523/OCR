import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from scripts.upload_app import (
    ApprovalRuleRegistry,
    DOCUMENT_TYPE_APPROVAL_RULE,
    DOCUMENT_TYPE_PARSER_TEST,
    JobStore,
    UPLOAD_HTML,
    UploadJob,
    build_result_url,
    build_parse_result_summary,
    default_approval_registry_path,
    make_handler,
    normalize_document_type,
    sanitize_upload_filename,
    should_record_approval_rule,
    validate_upload_filename,
)


def test_sanitize_upload_filename_removes_paths_and_keeps_extension():
    assert sanitize_upload_filename("../../법인카드 영수증.pdf") == "법인카드_영수증.pdf"
    assert sanitize_upload_filename("receipt 001.PNG") == "receipt_001.PNG"


def test_validate_upload_filename_accepts_supported_documents():
    validate_upload_filename("receipt.pdf")
    validate_upload_filename("receipt.png")
    validate_upload_filename("receipt.jpeg")
    validate_upload_filename("receipt.webp")


def test_validate_upload_filename_rejects_unsupported_documents():
    try:
        validate_upload_filename("receipt.txt")
    except ValueError as exc:
        assert "Unsupported upload type" in str(exc)
    else:
        raise AssertionError("Expected unsupported extension to raise")


def test_normalize_document_type_accepts_parser_test_and_approval_rules():
    assert normalize_document_type("parser_test") == DOCUMENT_TYPE_PARSER_TEST
    assert normalize_document_type("approval_rule") == DOCUMENT_TYPE_APPROVAL_RULE
    assert normalize_document_type("") == DOCUMENT_TYPE_APPROVAL_RULE


def test_normalize_document_type_rejects_unknown_values():
    try:
        normalize_document_type("invoice")
    except ValueError as exc:
        assert "Unsupported document type" in str(exc)
    else:
        raise AssertionError("Expected unsupported document type to raise")


def test_job_store_creates_jobs_and_records_completion(tmp_path: Path):
    store = JobStore()

    job = store.create("receipt.pdf", document_type=DOCUMENT_TYPE_PARSER_TEST)
    store.update(job.job_id, status="running", message="OCR running")
    store.complete(job.job_id, result_url=build_result_url("abc123"))

    snapshot = store.to_dict(job.job_id)
    assert snapshot["filename"] == "receipt.pdf"
    assert snapshot["document_type"] == DOCUMENT_TYPE_PARSER_TEST
    assert snapshot["status"] == "done"
    assert snapshot["message"] == "Done"
    assert snapshot["result_url"] == "/results/document_parse_uploads/abc123/index.html"


def test_job_store_records_failure():
    store = JobStore()
    job = store.create("receipt.pdf")

    store.fail(job.job_id, RuntimeError("boom"))

    snapshot = store.to_dict(job.job_id)
    assert snapshot["status"] == "error"
    assert snapshot["error"] == "boom"


def test_upload_ui_uses_dialog_and_minimizes_after_parse_completion():
    assert "Parser Test" in UPLOAD_HTML
    assert "전결규정 업로드" in UPLOAD_HTML
    assert "전결규정 관리" in UPLOAD_HTML
    assert 'data-view="parser-test"' in UPLOAD_HTML
    assert 'data-view="approval-upload"' in UPLOAD_HTML
    assert 'data-view="approval-manage"' in UPLOAD_HTML
    assert 'data-document-type="parser_test"' in UPLOAD_HTML
    assert 'data-document-type="approval_rule"' in UPLOAD_HTML
    assert "expectedJsonInput" in UPLOAD_HTML
    assert "runParserComparison" in UPLOAD_HTML
    assert "JSON 검증" in UPLOAD_HTML
    assert "VectorDB" in UPLOAD_HTML
    assert "criteria_case_chunks" in UPLOAD_HTML
    assert "showView" in UPLOAD_HTML
    assert "approval-list" in UPLOAD_HTML
    assert "loadApprovalRules" in UPLOAD_HTML
    assert '<body class="dialog-hidden">' in UPLOAD_HTML
    assert "dialog-backdrop" in UPLOAD_HTML
    assert "upload-dialog" in UPLOAD_HTML
    assert "openUploadDialog" in UPLOAD_HTML
    assert "dialog-minimized" in UPLOAD_HTML
    assert "result-mode" in UPLOAD_HTML
    assert "viewer-toolbar" in UPLOAD_HTML
    assert 'document.body.classList.add("result-mode")' in UPLOAD_HTML
    assert 'document.body.classList.add("dialog-minimized")' in UPLOAD_HTML
    assert 'min-height: calc(100vh - 53px)' in UPLOAD_HTML


def test_static_results_decode_percent_encoded_file_paths(tmp_path: Path):
    result_dir = tmp_path / "results/document_parse_uploads/job/page_001"
    result_dir.mkdir(parents=True)
    image = result_dir / "영수증.webp"
    image.write_bytes(b"webp-bytes")

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, JobStore()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        encoded_name = quote(image.name)
        url = f"http://127.0.0.1:{server.server_port}/results/document_parse_uploads/job/page_001/{encoded_name}"
        with urlopen(url, timeout=5) as response:
            assert response.status == 200
            assert response.read() == b"webp-bytes"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_approval_rule_registry_records_completed_documents(tmp_path: Path):
    registry = ApprovalRuleRegistry(tmp_path / "registry.json")
    job = UploadJob(job_id="abc123", filename="전결규정.pdf", created_at=10.0, updated_at=17.5)
    job.status = "done"
    job.result_url = build_result_url(job.job_id)

    registry.record_completed(job, raw_file=tmp_path / "raw" / "전결규정.pdf", page_count=3)

    rules = registry.list_rules()
    assert rules == [
        {
            "rule_id": "abc123",
            "filename": "전결규정.pdf",
            "title": "전결규정",
            "status": "applied",
            "document_type": "approval_rule",
            "result_url": "/results/document_parse_uploads/abc123/index.html",
            "page_count": 3,
            "uploaded_at": 10.0,
            "applied_at": 17.5,
            "processing_seconds": 7.5,
            "source_path": str(tmp_path / "raw" / "전결규정.pdf"),
            "vector_db": {
                "status": "pending",
                "namespace": "approval_rule:abc123",
                "embedding_strategy": "criteria_case_chunks",
                "criteria_count": 0,
                "case_count": 0,
            },
        }
    ]


def test_approval_rule_registry_hydrates_vector_db_plan_for_existing_rules(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"rules": [{"rule_id": "old123", "filename": "old.pdf", "applied_at": 1.0}]}),
        encoding="utf-8",
    )
    registry = ApprovalRuleRegistry(path)

    rules = registry.list_rules()

    assert rules[0]["vector_db"]["namespace"] == "approval_rule:old123"
    assert rules[0]["vector_db"]["embedding_strategy"] == "criteria_case_chunks"


def test_parser_test_jobs_are_not_recorded_as_approval_rules():
    parser_job = UploadJob(job_id="parser123", filename="parser.webp", document_type=DOCUMENT_TYPE_PARSER_TEST)
    approval_job = UploadJob(job_id="rule123", filename="rule.pdf", document_type=DOCUMENT_TYPE_APPROVAL_RULE)

    assert not should_record_approval_rule(parser_job)
    assert should_record_approval_rule(approval_job)


def test_approval_rules_api_returns_registry_entries(tmp_path: Path):
    registry = ApprovalRuleRegistry(default_approval_registry_path(tmp_path))
    registry.upsert(
        {
            "rule_id": "abc123",
            "filename": "전결규정.pdf",
            "title": "전결규정",
            "status": "applied",
            "document_type": "approval_rule",
            "result_url": "/results/document_parse_uploads/abc123/index.html",
            "page_count": 3,
            "uploaded_at": 10.0,
            "applied_at": 17.5,
            "processing_seconds": 7.5,
            "source_path": str(tmp_path / "raw" / "전결규정.pdf"),
            "vector_db": {
                "status": "pending",
                "namespace": "approval_rule:abc123",
                "embedding_strategy": "criteria_case_chunks",
                "criteria_count": 0,
                "case_count": 0,
            },
        }
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, JobStore(), registry))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/approval-rules"
        with urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["rules"][0]["rule_id"] == "abc123"
        assert payload["rules"][0]["status"] == "applied"
        assert payload["rules"][0]["vector_db"]["embedding_strategy"] == "criteria_case_chunks"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_build_parse_result_summary_merges_page_fields(tmp_path: Path):
    page_1 = tmp_path / "results/document_parse_uploads/job123/page_001/layout.json"
    page_2 = tmp_path / "results/document_parse_uploads/job123/page_002/layout.json"
    page_1.parent.mkdir(parents=True)
    page_2.parent.mkdir(parents=True)
    page_1.write_text(
        json.dumps({"image_id": "page_001", "fields": {"merchant": "A Mart"}, "serialized_text": "A"}),
        encoding="utf-8",
    )
    page_2.write_text(
        json.dumps({"image_id": "page_002", "fields": {"total": "10,000"}, "serialized_text": "B"}),
        encoding="utf-8",
    )

    summary = build_parse_result_summary(tmp_path, "job123")

    assert summary["job_id"] == "job123"
    assert summary["page_count"] == 2
    assert summary["merged_fields"] == {"merchant": "A Mart", "total": "10,000"}
    assert summary["pages"][0]["fields"] == {"merchant": "A Mart"}


def test_parse_results_api_returns_merged_fields(tmp_path: Path):
    layout = tmp_path / "results/document_parse_uploads/job123/page_001/layout.json"
    layout.parent.mkdir(parents=True)
    layout.write_text(json.dumps({"image_id": "page_001", "fields": {"total": "10,000"}}), encoding="utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, JobStore()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/parse-results/job123"
        with urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["merged_fields"] == {"total": "10,000"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
