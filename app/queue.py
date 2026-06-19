"""A minimal single-worker job queue.

The GPU can run one inference pipeline at a time, so jobs execute sequentially.
Each job is a callable; status and result are tracked by job id.
"""
from __future__ import annotations

import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Job:
    id: str
    fn: Callable[..., Any]
    status: str = "queued"          # queued | running | done | error
    progress: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def set_progress(self, msg: str) -> None:
        self.progress = msg


class JobQueue:
    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, fn: Callable[[Job], Any], meta: dict | None = None) -> str:
        """fn receives the Job so it can report progress via job.set_progress()."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, fn=fn, meta=meta or {})
        with self._lock:
            self._jobs[job_id] = job
        self._q.put(job_id)
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def position(self, job_id: str) -> int:
        """Approximate number of jobs ahead of this one (0 = next/running)."""
        with self._lock:
            ahead = [j for j in self._jobs.values() if j.status == "queued"]
        ahead.sort(key=lambda j: j.id)
        ids = [j.id for j in ahead]
        return ids.index(job_id) if job_id in ids else 0

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            job = self.get(job_id)
            if job is None:
                continue
            job.status = "running"
            try:
                job.result = job.fn(job)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.error = str(exc)
                traceback.print_exc()
            finally:
                self._q.task_done()


# Process-wide singleton.
JOBS = JobQueue()
