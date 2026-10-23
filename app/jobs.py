"""In-process job queue: one lattice build at a time, results kept on disk.

Good enough for one user; swap for a real queue (e.g. Redis + workers) before
the app is shared.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import threading
import time
import traceback
import uuid

import trimesh

from lattice_helmet.build import LatticeSpec, build, report


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | failed
    log: list = field(default_factory=list)
    report: dict = None
    error: str = None
    created: float = field(default_factory=time.time)
    source: dict = None  # where the shell came from (Onshape ids), if anywhere

    def public(self):
        return {"id": self.id, "status": self.status, "log": self.log[-20:],
                "report": self.report, "error": self.error, "source": self.source}


class JobQueue:
    def __init__(self, workdir, workers=1):
        self.dir = Path(workdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def submit(self, shell_bytes, recipe, filetype="stl", source=None):
        spec = LatticeSpec.from_dict(recipe)  # validate before queueing
        job = Job(id=uuid.uuid4().hex[:12], source=source)
        with self.lock:
            self.jobs[job.id] = job
        (self.dir / f"{job.id}.json").write_text(spec.to_json())
        self.pool.submit(self._run, job, shell_bytes, recipe, filetype)
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def result_path(self, job_id):
        return self.dir / f"{job_id}.3mf"

    def _run(self, job, shell_bytes, recipe, filetype):
        job.status = "running"
        try:
            shell = trimesh.load(io.BytesIO(shell_bytes), file_type=filetype, force="mesh")
            spec = LatticeSpec.from_dict(recipe, log=job.log.append)
            part = build(shell, spec)
            part.export(self.result_path(job.id))
            job.report = report(shell, part)
            (self.dir / f"{job.id}.report.json").write_text(json.dumps(job.report))
            job.status = "done"
        except Exception as e:  # report any failure to the panel
            job.error = f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc(limit=3))
            job.status = "failed"
