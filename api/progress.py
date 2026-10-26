"""In-memory OCR jobs with live progress (stage, page, lines read per model).

Processing code calls :func:`report` wherever it makes progress. The call is a
no-op outside a job, so the synchronous ``/api/v1/ocr`` path is unaffected.
Inside a job it updates the job's snapshot and raises :class:`JobCancelled`
when the user has cancelled, so long recognitions stop between line batches.

Jobs and their results live only in memory and expire after ``TTL_SECONDS``.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

TTL_SECONDS = 600
MAX_JOBS = 32
STAGE_LABELS = {
    "queued": "Waiting to start",
    "loading": "Loading model",
    "decode": "Read file",
    "preprocess": "Clean up image",
    "detect": "Find text lines",
    "recognize": "Recognise text",
    "vote": "Vote on each word",
    "layout": "Rebuild layout",
    "postprocess": "Correct errors",
    "done": "Done",
}

_local = threading.local()


class JobCancelled(Exception):
    """Raised inside the worker when the job was cancelled by the client."""


@dataclass
class Job:
    filename: str
    engine: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: float = field(default_factory=time.time)
    status: str = "queued"          # queued | running | done | error | cancelled
    stage: str = "queued"
    page: int = 0
    pages: int = 0
    lines_done: int = 0
    lines_total: int = 0
    members: dict[str, dict] = field(default_factory=dict)  # per-model lines done / total
    stage_times: dict[str, float] = field(default_factory=dict)
    stage_started: float = field(default_factory=time.perf_counter)
    started: float = field(default_factory=time.perf_counter)
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    error_status: int | None = None
    cancelled: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, stage: str | None = None, done: int | None = None, total: int | None = None,
               member: str | None = None, page: int | None = None, pages: int | None = None) -> None:
        with self.lock:
            if self.cancelled:
                raise JobCancelled()
            now = time.perf_counter()
            if self.status == "queued":
                self.status = "running"
            if stage and stage != self.stage:
                self.stage_times[self.stage] = self.stage_times.get(self.stage, 0.0) + (now - self.stage_started) * 1000
                self.stage, self.stage_started = stage, now
                if stage in ("preprocess", "detect"):  # a new page starts: reset line counters
                    self.lines_done, self.lines_total, self.members = 0, 0, {}
            if page is not None:
                self.page = page
            if pages is not None:
                self.pages = pages
            if member is not None:
                entry = self.members.setdefault(member, {"done": 0, "total": 0})
                if done is not None:
                    entry["done"] = done
                if total is not None:
                    entry["total"] = total
                self.lines_done = min(m["done"] for m in self.members.values())
                self.lines_total = max(m["total"] for m in self.members.values())
            else:
                if done is not None:
                    self.lines_done = done
                if total is not None:
                    self.lines_total = total

    def finish(self, status: str, result: dict | None = None, error: str | None = None, error_status: int | None = None) -> None:
        with self.lock:
            now = time.perf_counter()
            self.stage_times[self.stage] = self.stage_times.get(self.stage, 0.0) + (now - self.stage_started) * 1000
            self.status, self.result, self.error, self.error_status = status, result, error, error_status
            self.stage = "done" if status == "done" else self.stage
            self.finished = now

    def snapshot(self, include_result: bool = True) -> dict:
        with self.lock:
            end = self.finished or time.perf_counter()
            payload = {
                "id": self.id,
                "filename": self.filename,
                "engine": self.engine,
                "status": self.status,
                "stage": self.stage,
                "stage_label": STAGE_LABELS.get(self.stage, self.stage),
                "page": self.page,
                "pages": self.pages,
                "lines_done": self.lines_done,
                "lines_total": self.lines_total,
                "members": {k: dict(v) for k, v in self.members.items()},
                "stage_times": {k: round(v, 1) for k, v in self.stage_times.items() if k != "queued"},
                "elapsed_ms": round((end - self.started) * 1000, 1),
                "error": self.error,
                "error_status": self.error_status,
            }
            if include_result and self.result is not None:
                payload["result"] = self.result
            return payload


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> Job:
        with self._lock:
            self._expire()
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._expire()
            return self._jobs.get(job_id)

    def _expire(self) -> None:
        now = time.time()
        for job_id in [j for j, job in self._jobs.items() if now - job.created > TTL_SECONDS]:
            self._jobs.pop(job_id, None)
        while len(self._jobs) > MAX_JOBS:
            oldest = min(self._jobs.values(), key=lambda job: job.created)
            self._jobs.pop(oldest.id, None)


def bind(job: Job | None) -> None:
    """Attach (or detach) the job that this worker thread is processing."""
    _local.job = job


def report(stage: str | None = None, done: int | None = None, total: int | None = None,
           member: str | None = None, page: int | None = None, pages: int | None = None) -> None:
    job = getattr(_local, "job", None)
    if job is not None:
        job.update(stage, done, total, member, page, pages)
