#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    from scripts.ocr_workspace import build_inventory, write_inventory
    from scripts.preprocess_documents import preprocess_documents, write_manifest
    from scripts.run_dataset_pipeline import build_extraction_plan, run_extraction_plan
    from scripts.run_document_parse_layout import parse_inventory
except ModuleNotFoundError:
    from ocr_workspace import build_inventory, write_inventory
    from preprocess_documents import preprocess_documents, write_manifest
    from run_dataset_pipeline import build_extraction_plan, run_extraction_plan
    from run_document_parse_layout import parse_inventory


SUPPORTED_UPLOAD_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DOCUMENT_TYPE_PARSER_TEST = "parser_test"
DOCUMENT_TYPE_APPROVAL_RULE = "approval_rule"
SUPPORTED_DOCUMENT_TYPES = {DOCUMENT_TYPE_PARSER_TEST, DOCUMENT_TYPE_APPROVAL_RULE}


def default_approval_registry_path(workspace: Path) -> Path:
    return workspace / "data/private/approval_rules/registry.json"


def sanitize_upload_filename(filename: str) -> str:
    name = Path(filename or "upload").name.strip() or "upload"
    stem = Path(name).stem
    suffix = Path(name).suffix
    safe_stem = re.sub(r"[^\w가-힣.-]+", "_", stem, flags=re.UNICODE).strip("._")
    if not safe_stem:
        safe_stem = "upload"
    return f"{safe_stem}{suffix}"


def validate_upload_filename(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_SUFFIXES))
        raise ValueError(f"Unsupported upload type: {suffix or '(none)'}. Supported: {supported}")


def normalize_document_type(value: str | None) -> str:
    document_type = (value or DOCUMENT_TYPE_APPROVAL_RULE).strip() or DOCUMENT_TYPE_APPROVAL_RULE
    if document_type not in SUPPORTED_DOCUMENT_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_TYPES))
        raise ValueError(f"Unsupported document type: {document_type}. Supported: {supported}")
    return document_type


def build_result_url(job_id: str) -> str:
    return f"/results/document_parse_uploads/{job_id}/index.html"


def build_vector_db_plan(job_id: str) -> dict[str, Any]:
    return {
        "status": "pending",
        "namespace": f"approval_rule:{job_id}",
        "embedding_strategy": "criteria_case_chunks",
        "criteria_count": 0,
        "case_count": 0,
    }


def build_parse_result_summary(workspace: Path, job_id: str) -> dict[str, Any]:
    output_dir = workspace / "results/document_parse_uploads" / job_id
    layout_paths = sorted(output_dir.glob("*/layout.json"))
    if not layout_paths:
        raise FileNotFoundError(f"No parse layouts found for job: {job_id}")

    pages = []
    merged_fields: dict[str, Any] = {}
    for page_number, layout_path in enumerate(layout_paths, start=1):
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        fields = layout.get("fields") if isinstance(layout.get("fields"), dict) else {}
        for key, value in fields.items():
            if key not in merged_fields and value not in (None, ""):
                merged_fields[key] = value
        pages.append(
            {
                "page": page_number,
                "image_id": layout.get("image_id", layout_path.parent.name),
                "fields": fields,
                "serialized_text": layout.get("serialized_text", ""),
                "layout_path": str(layout_path.relative_to(workspace)),
            }
        )

    return {
        "job_id": job_id,
        "page_count": len(pages),
        "merged_fields": merged_fields,
        "pages": pages,
    }


def _default_poppler_bin() -> Path | None:
    configured = os.environ.get("POPPLER_BIN")
    if configured:
        return Path(configured)
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/bin"
    if (bundled / "pdftoppm").exists():
        return bundled
    return None


@dataclass
class UploadJob:
    job_id: str
    filename: str
    document_type: str = DOCUMENT_TYPE_APPROVAL_RULE
    status: str = "pending"
    message: str = "Queued"
    result_url: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, UploadJob] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, document_type: str = DOCUMENT_TYPE_APPROVAL_RULE) -> UploadJob:
        job = UploadJob(job_id=uuid.uuid4().hex[:12], filename=filename, document_type=document_type)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update(self, job_id: str, *, status: str | None = None, message: str | None = None) -> UploadJob:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if message is not None:
                job.message = message
            job.updated_at = time.time()
            return job

    def complete(self, job_id: str, *, result_url: str) -> UploadJob:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "done"
            job.message = "Done"
            job.result_url = result_url
            job.error = None
            job.updated_at = time.time()
            return job

    def fail(self, job_id: str, error: BaseException) -> UploadJob:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "error"
            job.message = "Failed"
            job.error = str(error)
            job.updated_at = time.time()
            return job

    def to_dict(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            return asdict(job)


class ApprovalRuleRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rules = payload.get("rules", [])
        else:
            rules = payload
        if not isinstance(rules, list):
            return []
        return [rule for rule in rules if isinstance(rule, dict)]

    def _write_unlocked(self, rules: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({"rules": rules}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def _hydrate_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(rule)
        rule_id = str(hydrated.get("rule_id") or "")
        if rule_id and not isinstance(hydrated.get("vector_db"), dict):
            hydrated["vector_db"] = build_vector_db_plan(rule_id)
        return hydrated

    def list_rules(self) -> list[dict[str, Any]]:
        with self._lock:
            rules = [self._hydrate_rule(rule) for rule in self._read_unlocked()]
        return sorted(rules, key=lambda rule: float(rule.get("applied_at") or 0), reverse=True)

    def upsert(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rules = [rule for rule in self._read_unlocked() if rule.get("rule_id") != entry.get("rule_id")]
            rules.append(entry)
            rules = sorted(rules, key=lambda rule: float(rule.get("applied_at") or 0), reverse=True)
            self._write_unlocked(rules)
        return entry

    def record_completed(self, job: UploadJob, *, raw_file: Path, page_count: int) -> dict[str, Any]:
        applied_at = round(job.updated_at, 3)
        uploaded_at = round(job.created_at, 3)
        entry = {
            "rule_id": job.job_id,
            "filename": job.filename,
            "title": Path(job.filename).stem or job.filename,
            "status": "applied",
            "document_type": "approval_rule",
            "result_url": job.result_url,
            "page_count": page_count,
            "uploaded_at": uploaded_at,
            "applied_at": applied_at,
            "processing_seconds": round(max(0.0, applied_at - uploaded_at), 3),
            "source_path": str(raw_file),
            "vector_db": build_vector_db_plan(job.job_id),
        }
        return self.upsert(entry)


def should_record_approval_rule(job: UploadJob) -> bool:
    return job.document_type == DOCUMENT_TYPE_APPROVAL_RULE


def run_uploaded_document_pipeline(
    workspace: Path,
    job_id: str,
    raw_file: Path,
    store: JobStore,
    approval_registry: ApprovalRuleRegistry | None = None,
) -> None:
    upload_root = workspace / "data/private/uploads" / job_id
    preprocessed_dir = upload_root / "preprocessed"
    inventory_path = workspace / "results/upload_inventories" / f"{job_id}.json"
    baselines_dir = workspace / "results/baselines_uploads" / job_id
    output_dir = workspace / "results/document_parse_uploads" / job_id

    store.update(job_id, status="running", message="Preprocessing")
    manifest = preprocess_documents(
        input_dir=raw_file.parent,
        output_dir=preprocessed_dir,
        poppler_bin=_default_poppler_bin(),
    )
    write_manifest(preprocessed_dir, manifest)

    store.update(job_id, status="running", message="Building inventory")
    inventory = build_inventory(preprocessed_dir)
    if not inventory:
        raise RuntimeError("No OCR-ready pages were created from the upload.")
    write_inventory(inventory, inventory_path)

    store.update(job_id, status="running", message=f"Running OCR on {len(inventory)} page(s)")
    plan = build_extraction_plan(
        inventory=inventory,
        extractors=["paddleocr-text"],
        results_dir=baselines_dir,
        skip_existing=False,
    )
    run_extraction_plan(plan)

    store.update(job_id, status="running", message="Rendering parse viewer")
    parse_inventory(inventory_path, baselines_dir, output_dir)
    completed_job = store.complete(job_id, result_url=build_result_url(job_id))
    if approval_registry is not None and should_record_approval_rule(completed_job):
        approval_registry.record_completed(completed_job, raw_file=raw_file, page_count=len(inventory))


def _json_bytes(payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> tuple[int, bytes, str]:
    return status.value, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


def _html_bytes() -> tuple[int, bytes, str]:
    return HTTPStatus.OK.value, UPLOAD_HTML.encode("utf-8"), "text/html; charset=utf-8"


def _not_found() -> tuple[int, bytes, str]:
    return _json_bytes({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def make_handler(workspace: Path, store: JobStore, approval_registry: ApprovalRuleRegistry | None = None):
    approval_registry = approval_registry or ApprovalRuleRegistry(default_approval_registry_path(workspace))

    class UploadHandler(BaseHTTPRequestHandler):
        server_version = "OCRUpload/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(*_html_bytes())
                return
            if parsed.path == "/api/approval-rules":
                self._send(*_json_bytes({"rules": approval_registry.list_rules()}))
                return
            if parsed.path.startswith("/api/parse-results/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                try:
                    self._send(*_json_bytes(build_parse_result_summary(workspace, job_id)))
                except FileNotFoundError:
                    self._send(*_not_found())
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                try:
                    self._send(*_json_bytes(store.to_dict(job_id)))
                except KeyError:
                    self._send(*_not_found())
                return
            if parsed.path.startswith("/results/"):
                self._serve_static(parsed.path)
                return
            self._send(*_not_found())

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/upload":
                self._send(*_not_found())
                return

            raw_filename = self.headers.get("X-File-Name", "upload")
            raw_document_type = self.headers.get("X-Document-Type")
            filename = sanitize_upload_filename(unquote(raw_filename))
            try:
                validate_upload_filename(filename)
                document_type = normalize_document_type(raw_document_type)
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    raise ValueError("Upload body is empty.")
                if content_length > MAX_UPLOAD_BYTES:
                    raise ValueError("Upload is too large.")
                content = self.rfile.read(content_length)
            except ValueError as exc:
                self._send(*_json_bytes({"error": str(exc)}, HTTPStatus.BAD_REQUEST))
                return

            job = store.create(filename, document_type=document_type)
            raw_dir = workspace / "data/private/uploads" / job.job_id / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / filename
            raw_file.write_bytes(content)

            thread = threading.Thread(
                target=self._run_job,
                args=(job.job_id, raw_file),
                daemon=True,
            )
            thread.start()

            self._send(*_json_bytes(store.to_dict(job.job_id), HTTPStatus.ACCEPTED))

        def _run_job(self, job_id: str, raw_file: Path) -> None:
            try:
                run_uploaded_document_pipeline(workspace, job_id, raw_file, store, approval_registry)
            except BaseException as exc:
                store.fail(job_id, exc)

        def _serve_static(self, request_path: str) -> None:
            relative = unquote(request_path).lstrip("/")
            target = (workspace / relative).resolve()
            workspace_root = workspace.resolve()
            try:
                target.relative_to(workspace_root)
            except ValueError:
                self._send(*_not_found())
                return
            if not target.is_file():
                self._send(*_not_found())
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            body = target.read_bytes()
            self._send(HTTPStatus.OK.value, body, content_type)

    return UploadHandler


UPLOAD_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Approval Rule Document Parser</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #eef2f7;
      color: #172033;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      overflow: hidden;
      background: #f5f7fb;
    }
    .viewer-shell {
      display: grid;
      grid-template-rows: 53px minmax(0, 1fr);
      min-height: 100vh;
    }
    .viewer-toolbar {
      position: relative;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      min-height: 53px;
      padding: 0 16px;
      background: #fff;
      border-bottom: 1px solid #dce4ef;
      box-shadow: 0 8px 24px rgba(31, 44, 71, 0.06);
    }
    .toolbar-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 0 0 auto;
    }
    .workspace-tabs {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px;
      border: 1px solid #d6deea;
      border-radius: 8px;
      background: #f8fafc;
    }
    .tab-button {
      min-height: 32px;
      padding: 0 11px;
      border: 0;
      border-radius: 6px;
      background: transparent;
      color: #607087;
      font-size: 13px;
      font-weight: 750;
      cursor: pointer;
      white-space: nowrap;
    }
    .tab-button.active {
      background: #172033;
      color: #fff;
    }
    .viewer-heading {
      min-width: 0;
      display: grid;
      gap: 2px;
    }
    .viewer-status {
      color: #607087;
      font-size: 12px;
      font-weight: 650;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .viewer-panel {
      min-height: calc(100vh - 53px);
      background: #f8fafc;
      overflow: hidden;
    }
    .workspace {
      min-height: calc(100vh - 53px);
      padding: 18px;
      overflow: auto;
    }
    .app-view {
      display: none;
      max-width: 1120px;
      margin: 0 auto;
    }
    .app-view.active {
      display: grid;
      gap: 14px;
    }
    .view-header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 14px;
      padding: 16px 0 2px;
    }
    .view-copy {
      display: grid;
      gap: 5px;
      min-width: 0;
    }
    .view-copy h2 {
      font-size: 21px;
    }
    .view-copy p {
      margin: 0;
      color: #607087;
      font-size: 13px;
      line-height: 1.5;
    }
    .view-grid {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(280px, 420px);
      gap: 14px;
      align-items: start;
    }
    .view-card {
      display: grid;
      gap: 13px;
      padding: 16px;
      border: 1px solid #dce4ef;
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 10px 26px rgba(31, 44, 71, 0.05);
    }
    .view-card p {
      margin: 0;
      color: #607087;
      font-size: 13px;
      line-height: 1.5;
    }
    .json-input {
      width: 100%;
      min-height: 210px;
      resize: vertical;
      padding: 12px;
      border: 1px solid #cfd8e6;
      border-radius: 8px;
      background: #fbfdff;
      color: #172033;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.55;
    }
    .json-status {
      min-height: 20px;
      color: #607087;
      font-size: 12px;
      font-weight: 650;
    }
    .json-status.error {
      color: #b42318;
    }
    .result-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 400px);
      height: calc(100vh - 53px);
      background: #fff;
    }
    .result-layout .result-frame {
      height: calc(100vh - 53px);
      min-height: calc(100vh - 53px);
    }
    .parser-compare-panel {
      min-height: 0;
      overflow: auto;
      padding: 14px;
      border-left: 1px solid #dce4ef;
      background: #f8fafc;
    }
    .compare-card {
      display: grid;
      gap: 10px;
      padding: 12px;
      border: 1px solid #dce4ef;
      border-radius: 8px;
      background: #fff;
    }
    .compare-row {
      display: grid;
      gap: 4px;
      padding: 9px;
      border: 1px solid #edf1f6;
      border-radius: 7px;
      background: #fbfdff;
      font-size: 12px;
    }
    .compare-row.match {
      border-color: #b8e6cf;
      background: #f2fbf6;
    }
    .compare-row.mismatch,
    .compare-row.missing {
      border-color: #ffd6d1;
      background: #fff7f5;
    }
    .compare-key {
      color: #172033;
      font-weight: 750;
    }
    .compare-values {
      color: #607087;
      overflow-wrap: anywhere;
    }
    .viewer-host[hidden],
    .workspace[hidden] {
      display: none;
    }
    .rules-drawer {
      position: fixed;
      top: 53px;
      right: 0;
      z-index: 30;
      display: grid;
      grid-template-rows: 57px minmax(0, 1fr);
      width: min(420px, 100vw);
      height: calc(100vh - 53px);
      background: #fff;
      border-left: 1px solid #dce4ef;
      box-shadow: -14px 0 32px rgba(31, 44, 71, 0.08);
      transform: translateX(100%);
      transition: transform 180ms ease;
    }
    body.rules-open .rules-drawer {
      transform: translateX(0);
    }
    .rules-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 57px;
      padding: 0 14px 0 16px;
      border-bottom: 1px solid #e6ebf2;
    }
    .rules-body {
      min-height: 0;
      overflow: auto;
      padding: 12px;
      background: #f8fafc;
    }
    .approval-list {
      display: grid;
      gap: 10px;
    }
    .rule-item {
      display: grid;
      gap: 9px;
      padding: 12px;
      border: 1px solid #dce4ef;
      border-radius: 8px;
      background: #fff;
    }
    .rule-main {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }
    .rule-title {
      min-width: 0;
      color: #172033;
      font-size: 14px;
      font-weight: 750;
      overflow-wrap: anywhere;
    }
    .badge {
      display: inline-grid;
      place-items: center;
      min-height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      background: #eaf7f1;
      color: #14734d;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }
    .rule-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: #607087;
      font-size: 12px;
    }
    .rule-actions {
      display: flex;
      justify-content: flex-end;
    }
    .button.compact {
      min-height: 32px;
      padding: 0 10px;
      font-size: 12px;
    }
    .dialog-backdrop {
      position: fixed;
      inset: 0;
      z-index: 40;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(15, 23, 42, 0.36);
      transition: background 180ms ease, opacity 180ms ease;
    }
    .upload-dialog {
      width: min(560px, calc(100vw - 32px));
      max-height: calc(100vh - 48px);
      background: #fff;
      border: 1px solid #d8e0ec;
      border-radius: 8px;
      box-shadow: 0 24px 70px rgba(20, 31, 52, 0.24);
      overflow: hidden;
      transform: translateY(0);
      opacity: 1;
      transition: transform 240ms ease, opacity 180ms ease, max-height 220ms ease, width 220ms ease, box-shadow 220ms ease;
    }
    .dialog-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 56px;
      padding: 0 16px;
      border-bottom: 1px solid #e6ebf2;
    }
    .dialog-title {
      min-width: 0;
      display: grid;
      gap: 2px;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 750;
    }
    h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
      font-weight: 750;
    }
    .meta {
      color: #607087;
      font-size: 12px;
      font-weight: 650;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .dialog-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 0 0 auto;
    }
    .dropzone {
      display: grid;
      align-content: center;
      justify-items: center;
      gap: 12px;
      min-height: 320px;
      margin: 16px;
      padding: 24px;
      border: 2px dashed #b8c4d6;
      border-radius: 8px;
      background: #f8fafc;
      text-align: center;
    }
    .dropzone.dragover {
      border-color: #2f6fed;
      background: #f3f7ff;
    }
    .file-input {
      display: none;
    }
    .button {
      display: inline-grid;
      place-items: center;
      min-height: 38px;
      padding: 0 14px;
      border: 1px solid #2f6fed;
      border-radius: 7px;
      background: #2f6fed;
      color: #fff;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
    }
    button.button:disabled {
      cursor: default;
      opacity: 0.6;
    }
    .button.secondary {
      border-color: #d6deea;
      background: #fff;
      color: #26364d;
    }
    .file-name {
      color: #334155;
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    .status {
      display: grid;
      gap: 10px;
      padding: 16px;
    }
    .status-row {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 12px;
      min-height: 36px;
      align-items: center;
      border-bottom: 1px solid #edf1f6;
      font-size: 13px;
    }
    .status-row strong {
      color: #607087;
      font-weight: 650;
    }
    .result-frame {
      width: 100%;
      height: calc(100vh - 53px);
      min-height: calc(100vh - 53px);
      border: 0;
      background: #fff;
      display: block;
    }
    .empty {
      display: grid;
      place-items: center;
      min-height: calc(100vh - 53px);
      padding: 24px;
      color: #607087;
      font-size: 14px;
      text-align: center;
    }
    .empty strong {
      display: block;
      margin-bottom: 6px;
      color: #26364d;
      font-size: 16px;
    }
    .dashboard {
      display: grid;
      grid-template-columns: minmax(280px, 520px) minmax(280px, 420px);
      gap: 14px;
      align-items: stretch;
      width: min(1000px, 100%);
    }
    .dashboard-panel {
      display: grid;
      align-content: start;
      gap: 14px;
      min-height: 230px;
      padding: 18px;
      border: 1px solid #dce4ef;
      border-radius: 8px;
      background: #fff;
      text-align: left;
      box-shadow: 0 12px 28px rgba(31, 44, 71, 0.06);
    }
    .dashboard-panel h2 {
      font-size: 17px;
    }
    .dashboard-panel p {
      margin: 0;
      color: #607087;
      font-size: 13px;
      line-height: 1.5;
    }
    body.dialog-hidden .dialog-backdrop {
      opacity: 0;
      pointer-events: none;
      background: transparent;
    }
    body.dialog-hidden .upload-dialog {
      transform: translateY(18px);
      opacity: 0;
    }
    body.dialog-minimized .dialog-backdrop {
      align-items: end;
      justify-items: center;
      padding: 0 16px 14px;
      pointer-events: none;
      background: transparent;
    }
    body.dialog-minimized .upload-dialog {
      width: min(520px, calc(100vw - 32px));
      max-height: 56px;
      pointer-events: auto;
      cursor: pointer;
      box-shadow: 0 12px 34px rgba(20, 31, 52, 0.18);
    }
    body.dialog-minimized .dropzone,
    body.dialog-minimized .status {
      display: none;
    }
    body.dialog-minimized .dialog-header {
      border-bottom: 0;
    }
    body.dialog-minimized .dialog-actions {
      display: none;
    }
    body.dialog-minimized .dialog-title .meta {
      max-width: 420px;
    }
    @media (max-width: 980px) {
      body { overflow: auto; }
      .viewer-shell { min-height: 100vh; }
      .result-frame, .empty { min-height: calc(100vh - 53px); }
      .viewer-toolbar { padding: 0 12px; }
      .toolbar-actions { gap: 6px; }
      .workspace-tabs { overflow-x: auto; max-width: 100%; }
      .tab-button { padding: 0 9px; }
      .view-header { align-items: start; flex-direction: column; }
      .view-grid { grid-template-columns: 1fr; }
      .result-layout { grid-template-columns: 1fr; height: auto; }
      .parser-compare-panel { border-left: 0; border-top: 1px solid #dce4ef; }
      .dashboard { grid-template-columns: 1fr; }
      .rules-drawer { width: 100vw; }
      .dialog-backdrop { padding: 12px; }
      body.dialog-minimized .dialog-backdrop { padding: 0 10px 10px; }
    }
  </style>
</head>
<body class="dialog-hidden">
  <main class="viewer-shell">
    <header class="viewer-toolbar">
      <div class="viewer-heading">
        <h1>Document Parser Console</h1>
        <span class="viewer-status" id="viewerStatus">Local OCR ready</span>
      </div>
      <nav class="workspace-tabs" aria-label="작업 화면">
        <button class="tab-button active" type="button" data-view-target="parser-test">Parser Test</button>
        <button class="tab-button" type="button" data-view-target="approval-upload">전결규정 업로드</button>
        <button class="tab-button" type="button" data-view-target="approval-manage">전결규정 관리</button>
      </nav>
    </header>
    <section class="viewer-panel" id="resultPanel">
      <div class="workspace" id="workspaceRoot">
        <section class="app-view active" data-view="parser-test">
          <div class="view-header">
            <div class="view-copy">
              <h2>Parser Test</h2>
              <p>문서 파싱 품질을 빠르게 확인하는 테스트 영역입니다. 결과는 레지스트리에 저장하지 않습니다.</p>
            </div>
            <button class="button upload-trigger" type="button" data-document-type="parser_test">테스트 문서 업로드</button>
          </div>
          <div class="view-grid">
            <section class="view-card">
              <h2>OCR + Layout Parser 테스트</h2>
              <p>PDF, PNG, JPG, WebP 파일을 올리면 로컬 PaddleOCR 파이프라인을 거쳐 클릭 가능한 레이아웃 뷰어를 생성합니다.</p>
            </section>
            <section class="view-card">
              <h2>JSON 검증</h2>
              <p>기대하는 JSON을 입력하면 파싱 후 실제 추출 fields와 key/value 단위로 비교합니다.</p>
              <textarea class="json-input" id="expectedJsonInput" spellcheck="false">{
  "merchant": "MART RECEIPT",
  "total": "13,200 KRW"
}</textarea>
              <div class="json-status" id="expectedJsonStatus">Parser Test 결과는 전결규정 관리 목록에 저장하지 않습니다.</div>
            </section>
          </div>
        </section>
        <section class="app-view" data-view="approval-upload">
          <div class="view-header">
            <div class="view-copy">
              <h2>전결규정 업로드</h2>
              <p>실제 적용할 전결규정 문서를 업로드합니다. 완료된 문서는 관리 목록에 자동 등록됩니다.</p>
            </div>
            <button class="button upload-trigger" type="button" data-document-type="approval_rule">전결규정 업로드</button>
          </div>
          <div class="view-grid">
            <section class="view-card">
              <h2>적용 대상 문서</h2>
              <p>전결규정, 위임전결표, 출장/비용/법인카드 관련 승인 규정 PDF나 이미지를 업로드하는 화면입니다.</p>
            </section>
            <section class="view-card">
              <h2>처리 결과</h2>
              <p>파싱 완료 후 뷰어가 열리고, 파일명/페이지 수/처리시간/결과 링크가 전결규정 관리 화면에 저장됩니다.</p>
              <p>VectorDB 저장은 기준별/케이스별 chunk 임베딩 전략(`criteria_case_chunks`)으로 이어질 수 있게 메타데이터를 준비합니다.</p>
            </section>
          </div>
        </section>
        <section class="app-view" data-view="approval-manage">
          <div class="view-header">
            <div class="view-copy">
              <h2>전결규정 관리</h2>
              <p>업로드되어 적용된 전결규정 문서를 확인하고 이전 파싱 결과를 다시 엽니다.</p>
            </div>
            <button class="button secondary" id="refreshRulesButton" type="button">새로고침</button>
          </div>
          <div class="approval-list" id="approvalList">
            <div class="rule-item"><div class="rule-title">불러오는 중</div></div>
          </div>
        </section>
      </div>
      <div class="viewer-host" id="viewerHost" hidden></div>
    </section>
  </main>
  <div class="dialog-backdrop" id="uploadBackdrop">
    <section class="upload-dialog" id="uploadDialog" role="dialog" aria-modal="true" aria-labelledby="uploadTitle">
      <header class="dialog-header">
        <div class="dialog-title">
          <h2 id="uploadTitle">전결규정 업로드</h2>
          <span class="meta" id="dialogMeta">PDF / PNG / JPG / WebP</span>
        </div>
        <div class="dialog-actions">
          <button class="button secondary" id="closeDialogButton" type="button">닫기</button>
        </div>
      </header>
      <div id="dropzone" class="dropzone">
        <div class="file-name" id="fileName">PDF / PNG / JPG / WebP</div>
        <label class="button" for="fileInput">파일 선택</label>
        <input class="file-input" id="fileInput" type="file" accept=".pdf,.png,.jpg,.jpeg,.webp,application/pdf,image/png,image/jpeg,image/webp">
      </div>
      <section class="status">
        <div class="status-row"><strong>상태</strong><span id="status">Idle</span></div>
        <div class="status-row"><strong>메시지</strong><span id="message">Ready</span></div>
        <div class="status-row"><strong>Job</strong><span id="jobId">-</span></div>
        <div class="status-row"><strong>결과</strong><span id="result"><button class="button secondary" type="button" disabled>대기</button></span></div>
      </section>
    </section>
  </div>
  <script>
    const uploadBackdrop = document.getElementById("uploadBackdrop");
    const uploadDialog = document.getElementById("uploadDialog");
    const workspaceRoot = document.getElementById("workspaceRoot");
    const viewerHost = document.getElementById("viewerHost");
    const tabButtons = Array.from(document.querySelectorAll("[data-view-target]"));
    const appViews = Array.from(document.querySelectorAll("[data-view]"));
    const uploadTriggers = Array.from(document.querySelectorAll(".upload-trigger"));
    const refreshRulesButton = document.getElementById("refreshRulesButton");
    const closeDialogButton = document.getElementById("closeDialogButton");
    const approvalList = document.getElementById("approvalList");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const fileName = document.getElementById("fileName");
    const uploadTitle = document.getElementById("uploadTitle");
    const expectedJsonInput = document.getElementById("expectedJsonInput");
    const expectedJsonStatus = document.getElementById("expectedJsonStatus");
    const statusNode = document.getElementById("status");
    const messageNode = document.getElementById("message");
    const jobNode = document.getElementById("jobId");
    const resultNode = document.getElementById("result");
    const resultPanel = document.getElementById("resultPanel");
    const viewerStatus = document.getElementById("viewerStatus");
    const dialogMeta = document.getElementById("dialogMeta");
    let currentDocumentType = "parser_test";

    function setStatus(status, message) {
      statusNode.textContent = status;
      messageNode.textContent = message || "";
      viewerStatus.textContent = message ? `${status} - ${message}` : status;
      dialogMeta.textContent = message ? `${status} - ${message}` : "PDF / PNG / JPG / WebP";
    }

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      })[char]);
    }

    function formatDate(epochSeconds) {
      if (!epochSeconds) return "-";
      return new Date(epochSeconds * 1000).toLocaleString("ko-KR", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit"
      });
    }

    function formatSeconds(seconds) {
      if (seconds === undefined || seconds === null) return "-";
      const value = Number(seconds);
      if (!Number.isFinite(value)) return "-";
      return `${value.toFixed(1)}s`;
    }

    function getExpectedJson() {
      const raw = expectedJsonInput.value.trim();
      expectedJsonStatus.classList.remove("error");
      if (!raw) {
        expectedJsonStatus.textContent = "기대 JSON 없이 파싱 결과만 확인합니다.";
        return null;
      }
      try {
        const parsed = JSON.parse(raw);
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
          throw new Error("top-level JSON object required");
        }
        expectedJsonStatus.textContent = `${Object.keys(parsed).length}개 필드 기준으로 검증합니다.`;
        return parsed;
      } catch (error) {
        expectedJsonStatus.textContent = `JSON 형식 오류: ${error.message}`;
        expectedJsonStatus.classList.add("error");
        return undefined;
      }
    }

    function normalizeCompareValue(value) {
      if (value === undefined || value === null) return "";
      return String(value).trim();
    }

    function compareExpectedToActual(expected, actual) {
      if (!expected) return [];
      return Object.entries(expected).map(([key, expectedValue]) => {
        const actualValue = actual ? actual[key] : undefined;
        const expectedText = normalizeCompareValue(expectedValue);
        const actualText = normalizeCompareValue(actualValue);
        let status = "match";
        if (!actualText) status = "missing";
        else if (expectedText !== actualText) status = "mismatch";
        return { key, expected: expectedText, actual: actualText, status };
      });
    }

    function renderCompareRows(rows) {
      if (!rows.length) {
        return `<div class="compare-row"><div class="compare-key">기대 JSON 없음</div><div class="compare-values">파싱된 fields만 확인하세요.</div></div>`;
      }
      return rows.map((row) => `
        <div class="compare-row ${row.status}">
          <div class="compare-key">${escapeHtml(row.key)} · ${row.status}</div>
          <div class="compare-values">expected: ${escapeHtml(row.expected || "-")}</div>
          <div class="compare-values">actual: ${escapeHtml(row.actual || "-")}</div>
        </div>
      `).join("");
    }

    function renderApprovalRules(rules) {
      if (document.querySelector('[data-view="approval-manage"]').classList.contains("active")) {
        viewerStatus.textContent = `${rules.length} applied approval rule(s)`;
      }
      if (!rules.length) {
        approvalList.innerHTML = `<div class="rule-item"><div class="rule-title">적용된 전결규정 없음</div><div class="rule-meta">업로드 완료 후 여기에 표시됩니다.</div></div>`;
        return;
      }
      approvalList.innerHTML = rules.map((rule) => `
        <article class="rule-item">
          <div class="rule-main">
            <div class="rule-title">${escapeHtml(rule.title || rule.filename)}</div>
            <span class="badge">적용</span>
          </div>
          <div class="rule-meta">
            <span>${escapeHtml(rule.filename)}</span>
            <span>${Number(rule.page_count || 0)} page</span>
            <span>${formatSeconds(rule.processing_seconds)}</span>
            <span>${formatDate(rule.applied_at)}</span>
            <span>VectorDB ${escapeHtml(rule.vector_db?.status || "pending")}</span>
            <span>${escapeHtml(rule.vector_db?.namespace || "")}</span>
          </div>
          <div class="rule-actions">
            <button class="button secondary compact" type="button" data-result-url="${escapeHtml(rule.result_url)}">열기</button>
          </div>
        </article>
      `).join("");
    }

    async function loadApprovalRules() {
      try {
        const response = await fetch("/api/approval-rules");
        const payload = await response.json();
        renderApprovalRules(payload.rules || []);
      } catch (error) {
        viewerStatus.textContent = "approval rules load failed";
        approvalList.innerHTML = `<div class="rule-item"><div class="rule-title">목록을 불러오지 못했습니다</div></div>`;
      }
    }

    function showView(viewName) {
      workspaceRoot.hidden = false;
      viewerHost.hidden = true;
      viewerHost.innerHTML = "";
      document.body.classList.remove("result-mode", "dialog-minimized");
      document.body.classList.add("dialog-hidden");
      tabButtons.forEach((button) => button.classList.toggle("active", button.dataset.viewTarget === viewName));
      appViews.forEach((view) => view.classList.toggle("active", view.dataset.view === viewName));
      if (viewName === "parser-test") {
        viewerStatus.textContent = "Parser Test ready";
      } else if (viewName === "approval-upload") {
        viewerStatus.textContent = "Approval rule upload ready";
      } else {
        loadApprovalRules();
      }
    }

    function openUploadDialog(documentType) {
      currentDocumentType = documentType || currentDocumentType;
      const isParserTest = currentDocumentType === "parser_test";
      uploadTitle.textContent = isParserTest ? "Parser Test 업로드" : "전결규정 업로드";
      dialogMeta.textContent = isParserTest ? "테스트 결과만 생성" : "완료 후 전결규정 관리에 저장";
      fileName.textContent = "PDF / PNG / JPG / WebP";
      document.body.classList.remove("dialog-hidden", "dialog-minimized");
    }

    function hideUploadDialog() {
      if (document.body.classList.contains("result-mode")) {
        document.body.classList.add("dialog-minimized");
        document.body.classList.remove("dialog-hidden");
        return;
      }
      document.body.classList.add("dialog-hidden");
    }

    async function runParserComparison(jobId, expectedJson) {
      const panel = document.getElementById("parserComparePanel");
      if (!panel) return;
      panel.innerHTML = `<div class="compare-card"><h2>JSON 검증</h2><div class="json-status">파싱 결과를 불러오는 중입니다.</div></div>`;
      try {
        const response = await fetch(`/api/parse-results/${jobId}`);
        const summary = await response.json();
        if (!response.ok) throw new Error(summary.error || "parse result load failed");
        const rows = compareExpectedToActual(expectedJson, summary.merged_fields || {});
        const matchCount = rows.filter((row) => row.status === "match").length;
        panel.innerHTML = `
          <div class="compare-card">
            <h2>JSON 검증</h2>
            <div class="json-status">${rows.length ? `${matchCount}/${rows.length} 필드 일치` : "기대 JSON 없음"}</div>
            ${renderCompareRows(rows)}
          </div>
          <div class="compare-card">
            <h2>추출 fields</h2>
            <pre>${escapeHtml(JSON.stringify(summary.merged_fields || {}, null, 2))}</pre>
          </div>
        `;
      } catch (error) {
        panel.innerHTML = `<div class="compare-card"><h2>JSON 검증</h2><div class="json-status error">${escapeHtml(error.message)}</div></div>`;
      }
    }

    function showResult(url, documentType, jobId, expectedJson) {
      document.body.classList.add("result-mode");
      document.body.classList.add("dialog-minimized");
      document.body.classList.remove("dialog-hidden");
      resultNode.innerHTML = `<a class="button" href="${url}" target="_blank" rel="noreferrer">뷰어 열기</a>`;
      workspaceRoot.hidden = true;
      viewerHost.hidden = false;
      if (documentType === "parser_test") {
        viewerHost.innerHTML = `
          <div class="result-layout">
            <iframe class="result-frame" src="${url}" title="Document parse viewer"></iframe>
            <aside class="parser-compare-panel" id="parserComparePanel">
              <div class="compare-card"><h2>JSON 검증</h2><div class="json-status">결과 대기 중</div></div>
            </aside>
          </div>
        `;
        runParserComparison(jobId, expectedJson);
      } else {
        viewerHost.innerHTML = `<iframe class="result-frame" src="${url}" title="Document parse viewer"></iframe>`;
      }
      if (documentType === "approval_rule") loadApprovalRules();
    }

    async function pollJob(jobId, expectedJson) {
      const response = await fetch(`/api/jobs/${jobId}`);
      const job = await response.json();
      setStatus(job.status, job.message || "");
      if (job.status === "done") {
        showResult(job.result_url, job.document_type, job.job_id, expectedJson);
        return;
      }
      if (job.status === "error") {
        resultNode.textContent = job.error || "Failed";
        return;
      }
      window.setTimeout(() => pollJob(jobId), 1500);
    }

    async function uploadFile(file) {
      if (!file) return;
      document.body.classList.remove("dialog-hidden", "dialog-minimized");
      fileName.textContent = file.name;
      setStatus("uploading", "Uploading");
      workspaceRoot.hidden = true;
      viewerHost.hidden = false;
      const isParserTest = currentDocumentType === "parser_test";
      const expectedJson = isParserTest ? getExpectedJson() : null;
      if (expectedJson === undefined) {
        setStatus("error", "기대 JSON 형식을 확인하세요.");
        return;
      }
      viewerHost.innerHTML = `<div class="empty"><div><strong>${isParserTest ? "Parser Test 실행 중" : "전결규정 파싱 중"}</strong>OCR 완료 후 이 화면에 뷰어가 열립니다.</div></div>`;
      resultNode.innerHTML = `<button class="button secondary" type="button" disabled>처리 중</button>`;
      const response = await fetch("/api/upload", {
        method: "POST",
        headers: {
          "X-File-Name": encodeURIComponent(file.name),
          "X-Document-Type": currentDocumentType,
          "Content-Type": "application/octet-stream"
        },
        body: file
      });
      const payload = await response.json();
      if (!response.ok) {
        setStatus("error", payload.error || "Upload failed");
        resultNode.textContent = payload.error || "Upload failed";
        return;
      }
      jobNode.textContent = payload.job_id;
      setStatus(payload.status, payload.message);
      pollJob(payload.job_id, expectedJson);
    }

    tabButtons.forEach((button) => button.addEventListener("click", () => showView(button.dataset.viewTarget)));
    uploadTriggers.forEach((button) => {
      button.addEventListener("click", () => openUploadDialog(button.dataset.documentType));
    });
    refreshRulesButton.addEventListener("click", loadApprovalRules);
    closeDialogButton.addEventListener("click", hideUploadDialog);
    approvalList.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-result-url]");
      if (!button) return;
      showResult(button.dataset.resultUrl, "approval_rule", "", null);
    });
    uploadBackdrop.addEventListener("click", (event) => {
      if (event.target === uploadBackdrop) hideUploadDialog();
    });
    uploadDialog.addEventListener("click", () => {
      if (document.body.classList.contains("dialog-minimized")) openUploadDialog(currentDocumentType);
    });
    fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));
    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
      uploadFile(event.dataTransfer.files[0]);
    });
    loadApprovalRules();
  </script>
</body>
</html>
"""


def serve(workspace: Path, host: str, port: int) -> None:
    workspace = workspace.resolve()
    os.chdir(workspace)
    store = JobStore()
    server = ThreadingHTTPServer((host, port), make_handler(workspace, store))
    print(f"Upload app: http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local document upload and parse app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8770, type=int)
    parser.add_argument("--workspace", default=Path("."), type=Path)
    args = parser.parse_args()
    serve(args.workspace, args.host, args.port)


if __name__ == "__main__":
    main()
