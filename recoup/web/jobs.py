"""Running an evaluation in the background, so the browser does not have to wait.

Policy Studio is the write path. Every other screen reads a completed run in
milliseconds; this one asks for a new one, which takes ~35 seconds. Those are
different requirements and they get different mechanisms — reads and writes rarely
want the same thing, and forcing both through one path means one of them is badly
served.

Deliberately a thread and a dictionary rather than a job queue. Celery or RQ earn
their operational cost when work must survive process death, be retried, or be
spread across machines. None of that applies to one deterministic run on a
single-user demo, and a broker plus a worker pool to schedule it would be
architecture as decoration.

What it does keep from a real queue, because these are the parts that matter at
any size:

* **Progress**, so a 35-second wait is legible rather than a spinner.
* **Failure captured as state**, not as a traceback into a dead thread.
* **Bounded history**, so a long session cannot grow this without limit.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# A studio session might explore a dozen settings. Keeping every ledger would
# grow ~10MB at a time, so old runs are evicted with their files.
MAX_JOBS = 6


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | failed
    stage: str = ""
    progress: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    ledger_path: str = ""

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "results": self.results,
            "overrides": self.overrides,
        }


class JobRegistry:
    def __init__(self, directory: str | Path = "data/studio", max_jobs: int = MAX_JOBS) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_jobs

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self) -> Job | None:
        with self._lock:
            return next(reversed(self._jobs.values()), None)

    def _evict(self) -> None:
        """Drop the oldest jobs, and their ledgers with them.

        Forgetting the job but leaving a 10MB file behind is the standard way a
        cache becomes a disk-space incident.
        """
        while len(self._jobs) > self._max:
            _, job = self._jobs.popitem(last=False)
            if job.ledger_path:
                Path(job.ledger_path).unlink(missing_ok=True)

    def submit(self, overrides: dict[str, Any], work) -> Job:
        """Register a job and run `work(job)` on a background thread."""
        job = Job(
            id=uuid.uuid4().hex[:12],
            status="queued",
            started_at=datetime.now(UTC).isoformat(),
            overrides=overrides,
        )
        job.ledger_path = str(self.directory / f"{job.id}.db")

        with self._lock:
            self._jobs[job.id] = job
            self._evict()

        def run() -> None:
            job.status = "running"
            try:
                work(job)
                job.status = "done"
                job.progress = 1.0
                job.stage = "complete"
            except Exception as exc:  # noqa: BLE001 — a failed run is a state, not a crash
                # Captured onto the job rather than raised. A traceback on a
                # background thread reaches nobody: the browser would poll a job
                # that never finishes and the user would learn nothing.
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.stage = "failed"
                traceback.print_exc()
            finally:
                job.finished_at = datetime.now(UTC).isoformat()

        threading.Thread(target=run, daemon=True, name=f"studio-{job.id}").start()
        return job
