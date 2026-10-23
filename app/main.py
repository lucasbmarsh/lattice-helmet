"""Compute service for the Onshape lattice panel.

    uvicorn app.main:app --reload

The panel (app/panel/index.html) is served at /panel and is what the Onshape
right-panel extension loads, with the document ids in its query string.
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lattice_helmet.build import LatticeSpec
from lattice_helmet.fields import lattice_kinds
from lattice_helmet.maps import MAPS

from .jobs import JobQueue
from .onshape import Onshape, OnshapeError

WORKDIR = Path(os.environ.get("LATTICE_WORKDIR", "jobs"))


def make_app(onshape_factory=Onshape, workdir=WORKDIR):
    app = FastAPI(title="Lattice helmet service")
    queue = JobQueue(workdir)
    app.state.queue = queue

    def onshape():
        try:
            return onshape_factory()
        except OnshapeError as e:
            raise HTTPException(503, str(e))

    @app.get("/")
    def root():
        return RedirectResponse("/panel/")

    @app.get("/api/options")
    def options():
        """What the panel's dropdowns offer, and the default recipe."""
        return {"lattices": lattice_kinds(), "cell_maps": list(MAPS),
                "gradings": ["uniform", "z", "depth", "point", "expr"],
                "skin_surfaces": ["all", "outer", "inner", "rim"],
                "defaults": json.loads(LatticeSpec().to_json())}

    @app.get("/api/parts")
    def parts(did: str, wvm: str, wvmid: str, eid: str):
        try:
            return onshape().list_parts(did, wvm, wvmid, eid)
        except OnshapeError as e:
            raise HTTPException(502, str(e))

    class OnshapeJob(BaseModel):
        did: str
        wvm: str = "w"
        wvmid: str
        eid: str
        part_id: str
        recipe: dict = {}

    @app.post("/api/jobs/onshape")
    def job_from_onshape(req: OnshapeJob):
        try:
            stl = onshape().export_stl(req.did, req.wvm, req.wvmid, req.eid, req.part_id)
        except OnshapeError as e:
            raise HTTPException(502, str(e))
        return _submit(stl, req.recipe, "stl", source=req.model_dump(exclude={"recipe"}))

    @app.post("/api/jobs/upload")
    async def job_from_upload(file: UploadFile = File(...), recipe: str = Form("{}")):
        filetype = Path(file.filename or "shell.stl").suffix.lstrip(".").lower() or "stl"
        return _submit(await file.read(), json.loads(recipe), filetype)

    def _submit(data, recipe, filetype, source=None):
        try:
            job = queue.submit(data, recipe, filetype, source)
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))
        return job.public()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = queue.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.get("/api/jobs/{job_id}/result.3mf")
    def job_result(job_id: str):
        job = queue.get(job_id)
        if not job or job.status != "done":
            raise HTTPException(404, "result not ready")
        return FileResponse(queue.result_path(job_id), media_type="model/3mf",
                            filename=f"lattice-{job_id}.3mf")

    @app.post("/api/jobs/{job_id}/import")
    def job_import(job_id: str, translate: bool = True):
        """Put the result back into the Onshape document it came from."""
        job = queue.get(job_id)
        if not job or job.status != "done":
            raise HTTPException(404, "result not ready")
        if not job.source or job.source.get("wvm") != "w":
            raise HTTPException(400, "only jobs started from an Onshape workspace can be imported")
        data = queue.result_path(job_id).read_bytes()
        try:
            return onshape().upload_file(job.source["did"], job.source["wvmid"],
                                         f"lattice-{job_id}.3mf", data, translate=translate)
        except OnshapeError as e:
            raise HTTPException(502, str(e))

    app.mount("/panel", StaticFiles(directory=Path(__file__).parent / "panel", html=True), name="panel")
    return app


app = make_app()
