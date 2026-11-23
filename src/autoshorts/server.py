"""`autoshorts serve`: a small local HTTP API for running jobs and fetching their clips.

Standard library only. It binds to 127.0.0.1 by default and is meant for one person on their
own machine, but it still refuses the two ways a web page could poke at it from a browser:
- POSTs must be application/json, which a cross-origin page can't send without a CORS
  preflight (and we never answer preflights)
- the Host header must be localhost / 127.0.0.1, which stops DNS rebinding

Endpoints:
  GET  /                               the review page (web/index.html)
  GET  /api/jobs                      all jobs, newest first
  POST /api/jobs                       {"video": path, ...options} -> {"id": ...}
  GET  /api/jobs/<id>                  state, stage, fraction, log tail, manifest or error
  GET  /api/jobs/<id>/files/<name>     a file from the job's output folder (Range supported)
"""

from __future__ import annotations

import json
import mimetypes
import queue
import re
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .pipeline import Job, run
from .render import RenderOptions
from .segment import Options

PAGE = Path(__file__).with_name("web") / "index.html"
MAX_BODY = 64 * 1024
LOG_LINES = 200
CHUNK = 256 * 1024
MODEL_NAME = re.compile(r"[A-Za-z0-9._-]{1,40}")
LAYOUTS = ("blur", "crop", "fit")


class BadRequest(ValueError):
    pass


class FolderBusy(RuntimeError):
    pass


@dataclass
class JobState:
    id: str
    video: Path
    out_dir: Path
    job: Job
    created: float = field(default_factory=time.time)
    state: str = "queued"  # queued | running | done | error
    stage: str = ""
    fraction: float = 0.0
    error: str | None = None
    manifest: dict[str, Any] | None = None
    superseded_by: str | None = None  # a later job that wrote to the same folder
    log: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_LINES))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_log(self, line: str) -> None:
        self.log.append(line)

    def set_progress(self, stage: str, fraction: float) -> None:
        self.stage, self.fraction = stage, fraction

    def to_json(self, log_lines: int = 50) -> dict[str, Any]:
        return {
            "id": self.id, "video": str(self.video), "out_dir": str(self.out_dir), "created": self.created,
            "state": self.state, "stage": self.stage, "fraction": round(self.fraction, 4),
            "error": self.error, "log": list(self.log)[-log_lines:] if log_lines else [],
            "manifest": self.manifest, "superseded_by": self.superseded_by,
        }


def _num(body: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    v = body.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float) or not lo <= v <= hi:
        raise BadRequest(f"{key} must be a number between {lo:g} and {hi:g}")
    return float(v)


def parse_job(body: dict[str, Any]) -> Job:
    """Validates a POST /api/jobs body into a Job. Same defaults and limits as the CLI."""
    video = body.get("video")
    if not isinstance(video, str) or not video.strip():
        raise BadRequest("video must be a path to a video file")
    # people paste paths straight from Explorer's "Copy as path", which adds quotes
    src = Path(video.strip().strip('"')).expanduser().resolve()
    if not src.is_file():
        raise BadRequest(f"{src} not found")
    count = _num(body, "count", 6, 1, 50)
    if count != int(count):
        raise BadRequest("count must be a whole number")
    min_len = _num(body, "min", 20, 1, 600)
    max_len = _num(body, "max", 55, 1, 600)
    if max_len < min_len:
        raise BadRequest("max must be >= min")
    layout = body.get("layout", "blur")
    if layout not in LAYOUTS:
        raise BadRequest(f"layout must be one of {', '.join(LAYOUTS)}")
    captions = body.get("captions", True)
    if not isinstance(captions, bool):
        raise BadRequest("captions must be true or false")
    model = body.get("model", "small")
    if not isinstance(model, str) or not MODEL_NAME.fullmatch(model):
        raise BadRequest("model must be a whisper model name like small or large-v3")
    out = body.get("out")
    if out is not None and not isinstance(out, str):
        raise BadRequest("out must be a folder path")
    return Job(
        source=src,
        out_dir=Path(out).expanduser().resolve() if out else src.with_name(f"{src.stem}_shorts"),
        clips=Options(count=int(count), min_len=min_len, max_len=max_len),
        render=RenderOptions(layout=layout),
        captions=captions,
        whisper_model=model,
    )


class App:
    """Job registry plus one worker thread. Jobs run one at a time: two whisper models and two
    ffmpeg encodes fighting over the same GPU is slower than doing them in order."""

    def __init__(self) -> None:
        self.jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()  # submit() runs on request threads
        self._queue: queue.Queue[JobState] = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="autoshorts-worker", daemon=True)
        self._worker.start()

    def submit(self, job: Job) -> JobState:
        """Queues a job. Raises FolderBusy if a queued or running job already writes to the same folder.
        Finished jobs in that folder get `superseded_by`, since the new run will overwrite their clips."""
        folder = job.out_dir.resolve()
        with self._lock:
            same = [s for s in self.jobs.values() if s.out_dir.resolve() == folder]
            busy = next((s for s in same if s.state in ("queued", "running")), None)
            if busy:
                raise FolderBusy(f"job {busy.id} is still writing to {folder}")
            st = JobState(id=uuid.uuid4().hex[:12], video=job.source, out_dir=job.out_dir, job=job)
            for old in same:
                old.superseded_by = st.id
            self.jobs[st.id] = st
        self._queue.put(st)
        return st

    def _work(self) -> None:
        while True:
            self.execute(self._queue.get())

    def execute(self, st: JobState) -> None:
        st.state = "running"
        try:
            with st.lock:
                manifest = run(st.job, log=st.add_log, progress=st.set_progress)
                st.manifest = json.loads(manifest.read_text(encoding="utf-8"))
            st.state = "done"
        except Exception as e:  # anything the pipeline throws ends up on the page, not in a dead thread
            st.error = str(e) or type(e).__name__
            st.add_log("".join(traceback.format_exception_only(e)).strip())
            st.state = "error"


class Handler(BaseHTTPRequestHandler):
    server_version = "autoshorts"
    app: App  # set on the subclass made by make_server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # the default prints every request, including each Range chunk of a playing video

    # --- helpers ---

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in {"localhost", "127.0.0.1", "::1"}

    def _read_json(self) -> dict[str, Any]:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise _HttpError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send JSON with Content-Type: application/json")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise _HttpError(HTTPStatus.BAD_REQUEST, "bad Content-Length") from None
        if length < 0:
            # rfile.read(-1) would wait for the client to close the connection
            raise _HttpError(HTTPStatus.BAD_REQUEST, "bad Content-Length")
        if length > MAX_BODY:
            raise _HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "body is not valid JSON") from None
        if not isinstance(body, dict):
            raise _HttpError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        return body

    def _job(self, job_id: str) -> JobState:
        st = self.app.jobs.get(job_id)
        if st is None:
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such job")
        return st

    # --- routing ---

    def _dispatch(self, method: str) -> None:
        if not self._host_ok():
            self._error(HTTPStatus.FORBIDDEN, "only localhost may talk to this server")
            return
        path = self.path.split("?", 1)[0]
        # decode after splitting, so an encoded %2F stays inside one part (and gets rejected as a file name)
        parts = [unquote(p) for p in path.split("/") if p]
        try:
            if method == "GET" and parts in ([], ["index.html"]):
                self._send_page()
            elif method == "GET" and parts == ["api", "jobs"]:
                jobs = sorted(self.app.jobs.values(), key=lambda s: s.created, reverse=True)
                self._send_json(HTTPStatus.OK, [s.to_json(log_lines=0) for s in jobs])
            elif method == "POST" and parts == ["api", "jobs"]:
                try:
                    job = parse_job(self._read_json())
                except BadRequest as e:
                    raise _HttpError(HTTPStatus.BAD_REQUEST, str(e)) from None
                try:
                    st = self.app.submit(job)
                except FolderBusy as e:
                    raise _HttpError(HTTPStatus.CONFLICT, str(e)) from None
                self._send_json(HTTPStatus.CREATED, {"id": st.id})
            elif method == "GET" and len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                self._send_json(HTTPStatus.OK, self._job(parts[2]).to_json())
            elif method == "GET" and len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "files":
                self._send_file(self._job(parts[2]), parts[4])
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except _HttpError as e:
            self._error(e.status, e.message)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    # --- files ---

    def _send_page(self) -> None:
        body = PAGE.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        # everything the page needs comes from here, and it never needs to be framed
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                         "frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, st: JobState, name: str) -> None:
        # one plain file name inside the output folder, nothing else
        if "/" in name or "\\" in name or ":" in name or name in {"", ".", ".."}:
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such file")
        f = st.out_dir / name
        if not f.is_file() or f.resolve().parent != st.out_dir.resolve():
            raise _HttpError(HTTPStatus.NOT_FOUND, "no such file")
        size = f.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        rng = self.headers.get("Range")
        if rng:
            parsed = parse_range(rng, size)
            if parsed is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = parsed
            status = HTTPStatus.PARTIAL_CONTENT
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with f.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = fh.read(min(CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browsers drop video requests all the time when seeking


class _HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Single byte range ("bytes=a-b", "bytes=a-", "bytes=-n") as inclusive (start, end), None if unsatisfiable."""
    m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", header)
    if not m or (not m.group(1) and not m.group(2)) or size == 0:
        return None
    if not m.group(1):  # suffix: last n bytes
        n = int(m.group(2))
        return (max(0, size - n), size - 1) if n > 0 else None
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else size - 1
    if start >= size or end < start:
        return None
    return start, min(end, size - 1)


def make_server(host: str = "127.0.0.1", port: int = 8765, app: App | None = None) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app or App()})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(host, port)
    print(f"autoshorts is running at http://{host}:{server.server_address[1]}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
