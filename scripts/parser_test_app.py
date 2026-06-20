#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    from scripts.upload_app import (
        DOCUMENT_TYPE_PARSER_TEST,
        MAX_UPLOAD_BYTES,
        JobStore,
        build_parse_result_summary,
        build_result_url,
        run_uploaded_document_pipeline,
        sanitize_upload_filename,
        validate_upload_filename,
    )
except ModuleNotFoundError:
    from upload_app import (
        DOCUMENT_TYPE_PARSER_TEST,
        MAX_UPLOAD_BYTES,
        JobStore,
        build_parse_result_summary,
        build_result_url,
        run_uploaded_document_pipeline,
        sanitize_upload_filename,
        validate_upload_filename,
    )


def _json_bytes(payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> tuple[int, bytes, str]:
    return status.value, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


def _html_bytes() -> tuple[int, bytes, str]:
    return HTTPStatus.OK.value, PARSER_TEST_HTML.encode("utf-8"), "text/html; charset=utf-8"


def _not_found() -> tuple[int, bytes, str]:
    return _json_bytes({"error": "Not found"}, HTTPStatus.NOT_FOUND)


def make_handler(workspace: Path, store: JobStore):
    class ParserTestHandler(BaseHTTPRequestHandler):
        server_version = "OCRParserTest/0.1"

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
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                try:
                    self._send(*_json_bytes(store.to_dict(job_id)))
                except KeyError:
                    self._send(*_not_found())
                return
            if parsed.path.startswith("/api/parse-results/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                try:
                    self._send(*_json_bytes(build_parse_result_summary(workspace, job_id)))
                except FileNotFoundError:
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
            filename = sanitize_upload_filename(unquote(raw_filename))
            try:
                validate_upload_filename(filename)
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    raise ValueError("Upload body is empty.")
                if content_length > MAX_UPLOAD_BYTES:
                    raise ValueError("Upload is too large.")
                content = self.rfile.read(content_length)
            except ValueError as exc:
                self._send(*_json_bytes({"error": str(exc)}, HTTPStatus.BAD_REQUEST))
                return

            job = store.create(filename, document_type=DOCUMENT_TYPE_PARSER_TEST)
            raw_dir = workspace / "data/private/uploads" / job.job_id / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / filename
            raw_file.write_bytes(content)

            thread = threading.Thread(target=self._run_job, args=(job.job_id, raw_file), daemon=True)
            thread.start()

            self._send(*_json_bytes(store.to_dict(job.job_id), HTTPStatus.ACCEPTED))

        def _run_job(self, job_id: str, raw_file: Path) -> None:
            try:
                run_uploaded_document_pipeline(workspace, job_id, raw_file, store, approval_registry=None)
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
            self._send(HTTPStatus.OK.value, target.read_bytes(), content_type)

    return ParserTestHandler


PARSER_TEST_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Parser Test</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      color: #172033;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: #f6f8fb;
    }
    button, textarea, input { font: inherit; }
    .shell {
      display: grid;
      grid-template-rows: 54px minmax(0, 1fr);
      min-height: 100vh;
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 0 16px;
      border-bottom: 1px solid #dbe3ee;
      background: #fff;
    }
    .title {
      display: grid;
      gap: 2px;
      min-width: 0;
    }
    h1 {
      margin: 0;
      font-size: 15px;
      line-height: 1.2;
    }
    .status {
      color: #65758c;
      font-size: 12px;
      font-weight: 650;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .button {
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid #172033;
      border-radius: 7px;
      background: #172033;
      color: #fff;
      font-size: 13px;
      font-weight: 750;
      cursor: pointer;
      white-space: nowrap;
    }
    .button.secondary {
      border-color: #cfd8e6;
      background: #fff;
      color: #26364d;
    }
    .button:disabled {
      cursor: not-allowed;
      opacity: .55;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: calc(100vh - 54px);
    }
    .viewer {
      min-width: 0;
      min-height: calc(100vh - 54px);
      background: #fff;
    }
    .viewer iframe {
      width: 100%;
      height: calc(100vh - 54px);
      border: 0;
      display: block;
    }
    .empty {
      display: grid;
      place-items: center;
      height: calc(100vh - 54px);
      padding: 24px;
      color: #65758c;
      text-align: center;
    }
    .side {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 12px;
      min-height: calc(100vh - 54px);
      padding: 14px;
      border-left: 1px solid #dbe3ee;
      background: #f8fafc;
    }
    .panel {
      display: grid;
      gap: 10px;
      padding: 12px;
      border: 1px solid #dbe3ee;
      border-radius: 8px;
      background: #fff;
    }
    .panel h2 {
      margin: 0;
      font-size: 14px;
      line-height: 1.3;
    }
    .json-input {
      width: 100%;
      min-height: 180px;
      resize: vertical;
      padding: 10px;
      border: 1px solid #cfd8e6;
      border-radius: 7px;
      background: #fbfdff;
      color: #172033;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.55;
    }
    .json-status {
      min-height: 20px;
      color: #65758c;
      font-size: 12px;
      font-weight: 650;
    }
    .json-status.error { color: #b42318; }
    .compare-list {
      display: grid;
      gap: 8px;
      overflow: auto;
      min-height: 0;
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
      color: #65758c;
      word-break: break-word;
    }
    .dialog-backdrop {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: none;
      place-items: center;
      padding: 18px;
      background: rgba(23, 32, 51, .32);
    }
    body.dialog-open .dialog-backdrop { display: grid; }
    .upload-dialog {
      width: min(520px, 100%);
      display: grid;
      gap: 12px;
      padding: 16px;
      border: 1px solid #dbe3ee;
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 22px 60px rgba(23, 32, 51, .20);
    }
    .dialog-head {
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
    }
    .dialog-head h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.3;
    }
    .dropzone {
      display: grid;
      place-items: center;
      gap: 10px;
      min-height: 142px;
      padding: 16px;
      border: 1px dashed #9fb0c5;
      border-radius: 8px;
      background: #f8fafc;
      text-align: center;
    }
    .dropzone.dragover {
      border-color: #172033;
      background: #eef4ff;
    }
    .file-input {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
    }
    .file-name {
      color: #65758c;
      font-size: 13px;
      word-break: break-word;
    }
    .job-grid {
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 7px 10px;
      color: #65758c;
      font-size: 12px;
    }
    .job-grid strong { color: #172033; }
    @media (max-width: 900px) {
      .workspace { grid-template-columns: 1fr; }
      .side {
        grid-row: 1;
        min-height: auto;
        border-left: 0;
        border-bottom: 1px solid #dbe3ee;
      }
      .viewer, .viewer iframe, .empty { height: 68vh; min-height: 68vh; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="toolbar">
      <div class="title">
        <h1>Parser Test</h1>
        <span class="status" id="viewerStatus">Local OCR ready</span>
      </div>
      <button class="button" id="openDialogButton" type="button">문서 업로드</button>
    </header>
    <section class="workspace">
      <section class="viewer" id="viewerHost">
        <div class="empty">PDF 또는 이미지를 업로드하면 파싱 뷰어가 여기에 열립니다.</div>
      </section>
      <aside class="side">
        <section class="panel">
          <h2>기대 JSON</h2>
          <textarea class="json-input" id="expectedJsonInput" spellcheck="false">{
  "merchant": "",
  "total": "",
  "date": ""
}</textarea>
          <div class="json-status" id="expectedJsonStatus">업로드 전 JSON 형식을 확인합니다.</div>
        </section>
        <section class="panel">
          <h2>검증 결과</h2>
          <div class="compare-list" id="compareList">
            <div class="compare-row">
              <div class="compare-key">대기</div>
              <div class="compare-values">파싱 완료 후 fields와 비교합니다.</div>
            </div>
          </div>
        </section>
      </aside>
    </section>
  </main>

  <div class="dialog-backdrop" id="uploadBackdrop">
    <section class="upload-dialog" role="dialog" aria-modal="true" aria-labelledby="uploadTitle">
      <div class="dialog-head">
        <h2 id="uploadTitle">문서 업로드</h2>
        <button class="button secondary" id="closeDialogButton" type="button">닫기</button>
      </div>
      <div class="dropzone" id="dropzone">
        <div class="file-name" id="fileName">PDF / PNG / JPG / JPEG / WebP</div>
        <label class="button" for="fileInput">파일 선택</label>
        <input class="file-input" id="fileInput" type="file" accept=".pdf,.png,.jpg,.jpeg,.webp,application/pdf,image/png,image/jpeg,image/webp">
      </div>
      <div class="job-grid">
        <strong>상태</strong><span id="jobStatus">Idle</span>
        <strong>메시지</strong><span id="jobMessage">Ready</span>
        <strong>Job</strong><span id="jobId">-</span>
      </div>
    </section>
  </div>

  <script>
    const openDialogButton = document.getElementById("openDialogButton");
    const closeDialogButton = document.getElementById("closeDialogButton");
    const uploadBackdrop = document.getElementById("uploadBackdrop");
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const fileName = document.getElementById("fileName");
    const jobStatus = document.getElementById("jobStatus");
    const jobMessage = document.getElementById("jobMessage");
    const jobId = document.getElementById("jobId");
    const viewerStatus = document.getElementById("viewerStatus");
    const viewerHost = document.getElementById("viewerHost");
    const expectedJsonInput = document.getElementById("expectedJsonInput");
    const expectedJsonStatus = document.getElementById("expectedJsonStatus");
    const compareList = document.getElementById("compareList");

    let activeExpectedJson = {};

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[char]));
    }

    function parseExpectedJson() {
      try {
        const text = expectedJsonInput.value.trim();
        const parsed = text ? JSON.parse(text) : {};
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
          throw new Error("JSON object만 입력할 수 있습니다.");
        }
        expectedJsonStatus.classList.remove("error");
        expectedJsonStatus.textContent = `${Object.keys(parsed).length} fields`;
        return parsed;
      } catch (error) {
        expectedJsonStatus.classList.add("error");
        expectedJsonStatus.textContent = error.message;
        return undefined;
      }
    }

    function valueText(value) {
      if (value === undefined || value === null) return "";
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    }

    function renderComparison(actual) {
      const expected = activeExpectedJson || {};
      const keys = [...new Set([...Object.keys(expected), ...Object.keys(actual || {})])];
      if (!keys.length) {
        compareList.innerHTML = `<div class="compare-row"><div class="compare-key">결과 없음</div><div class="compare-values">layout.json의 fields가 비어 있습니다.</div></div>`;
        return;
      }
      compareList.innerHTML = keys.map((key) => {
        const expectedValue = valueText(expected[key]);
        const actualValue = valueText(actual ? actual[key] : undefined);
        const status = actualValue === "" ? "missing" : expectedValue === "" || expectedValue === actualValue ? "match" : "mismatch";
        return `
          <div class="compare-row ${status}">
            <div class="compare-key">${escapeHtml(key)} · ${status}</div>
            <div class="compare-values">expected: ${escapeHtml(expectedValue || "-")}</div>
            <div class="compare-values">actual: ${escapeHtml(actualValue || "-")}</div>
          </div>
        `;
      }).join("");
    }

    function openDialog() {
      document.body.classList.add("dialog-open");
    }

    function closeDialog() {
      document.body.classList.remove("dialog-open");
    }

    async function pollJob(id) {
      while (true) {
        const response = await fetch(`/api/jobs/${id}`);
        const job = await response.json();
        jobStatus.textContent = job.status;
        jobMessage.textContent = job.message || "-";
        viewerStatus.textContent = `${job.status} · ${job.message || ""}`;
        if (job.status === "done") {
          closeDialog();
          viewerHost.innerHTML = `<iframe src="${job.result_url}" title="Document parse viewer"></iframe>`;
          const parseResponse = await fetch(`/api/parse-results/${id}`);
          if (parseResponse.ok) {
            const payload = await parseResponse.json();
            renderComparison(payload.merged_fields || {});
          }
          return;
        }
        if (job.status === "error") {
          viewerStatus.textContent = job.error || "parse failed";
          compareList.innerHTML = `<div class="compare-row missing"><div class="compare-key">오류</div><div class="compare-values">${escapeHtml(job.error || "parse failed")}</div></div>`;
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 900));
      }
    }

    async function uploadFile(file) {
      const expected = parseExpectedJson();
      if (expected === undefined) return;
      activeExpectedJson = expected;
      fileName.textContent = file.name;
      jobStatus.textContent = "uploading";
      jobMessage.textContent = "Uploading";
      viewerStatus.textContent = "uploading";
      viewerHost.innerHTML = `<div class="empty">OCR 처리 중입니다.</div>`;
      compareList.innerHTML = `<div class="compare-row"><div class="compare-key">처리 중</div><div class="compare-values">파싱 결과를 기다리고 있습니다.</div></div>`;

      const response = await fetch("/api/upload", {
        method: "POST",
        headers: {
          "Content-Type": file.type || "application/octet-stream",
          "X-File-Name": encodeURIComponent(file.name),
        },
        body: file,
      });
      const payload = await response.json();
      if (!response.ok) {
        jobStatus.textContent = "error";
        jobMessage.textContent = payload.error || "Upload failed";
        return;
      }
      jobId.textContent = payload.job_id;
      await pollJob(payload.job_id);
    }

    openDialogButton.addEventListener("click", openDialog);
    closeDialogButton.addEventListener("click", closeDialog);
    uploadBackdrop.addEventListener("click", (event) => {
      if (event.target === uploadBackdrop) closeDialog();
    });
    expectedJsonInput.addEventListener("input", parseExpectedJson);
    fileInput.addEventListener("change", () => {
      const [file] = fileInput.files || [];
      if (file) uploadFile(file);
    });
    dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
    dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
      const [file] = event.dataTransfer.files || [];
      if (file) uploadFile(file);
    });
    parseExpectedJson();
  </script>
</body>
</html>
"""


def serve(workspace: Path, host: str, port: int) -> None:
    store = JobStore()
    server = ThreadingHTTPServer((host, port), make_handler(workspace, store))
    print(f"Parser test app: http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight local parser test app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8771, type=int)
    parser.add_argument("--workspace", default=Path("."), type=Path)
    args = parser.parse_args()
    serve(args.workspace.resolve(), args.host, args.port)


if __name__ == "__main__":
    main()
