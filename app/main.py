"""Compute service for the Onshape lattice panel.

    uvicorn app.main:app --reload

The panel (app/panel/index.html) is served at /panel and is what the Onshape
right-panel extension loads, with the document ids in its query string.
With ONSHAPE_CLIENT_ID set, every call that reads Onshape or starts a build
needs the user's OAuth token (see app/auth.py).
"""

import html
import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

from lattice_helmet.build import GRADINGS, LatticeSpec, simulate
from lattice_helmet.fea import MATERIALS, SimSpec, TopoSpec, density_mesh
from lattice_helmet.fields import lattice_kinds
from lattice_helmet.maps import MAPS

from .auth import OAuth, OAuthError
from .jobs import JobQueue
from .onshape import Onshape, OnshapeError
from .shells import ShellStore, Superseded, pack

WORKDIR = Path(os.environ.get("LATTICE_WORKDIR", "jobs"))


_UNSET = object()

CALLBACK_PAGE = """<!doctype html><meta charset="utf-8"><title>Lattice helmet</title>
<body style="font:14px system-ui;padding:24px">{message}
<script>{script}</script></body>"""


def _onshape_failed(e):
    """Pass Onshape's 401/403 through so the panel can ask the user to sign in again."""
    return HTTPException(e.status if e.status in (401, 403) else 502, str(e))


def make_app(onshape_factory=Onshape, workdir=WORKDIR, oauth=_UNSET):
    app = FastAPI(title="Lattice helmet service")

    @app.middleware("http")
    async def panel_revalidates(request: Request, call_next):
        # Without this, browsers reuse a cached panel for days after a deploy
        # (heuristic caching from Last-Modified). no-cache still lets them
        # keep it, but they check the ETag first, so a deploy shows at once.
        response = await call_next(request)
        if request.url.path.startswith("/panel"):
            response.headers["Cache-Control"] = "no-cache"
        return response
    queue = JobQueue(workdir)
    shells = ShellStore()
    oauth = OAuth.from_env() if oauth is _UNSET else oauth
    app.state.queue, app.state.oauth, app.state.shells = queue, oauth, shells

    def make_client(token=None):
        try:
            return onshape_factory(token=token) if token else onshape_factory()
        except OnshapeError as e:
            raise HTTPException(503, str(e))

    def user_token(request: Request):
        """The caller's Onshape token when OAuth is on, else None (server credentials)."""
        if not oauth:
            return None
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "sign in with Onshape first")
        try:
            oauth.check(token, make_client(token).session_info)
        except OnshapeError as e:
            raise _onshape_failed(e)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        return token

    @app.get("/")
    def root():
        return RedirectResponse("/panel/")

    @app.get("/api/options")
    def options():
        """What the panel's dropdowns offer, and the default recipe."""
        return {"lattices": lattice_kinds(), "cell_maps": list(MAPS),
                "gradings": list(GRADINGS), "materials": {k: {"E": e, "nu": n} for k, (e, n) in MATERIALS.items()},
                "skin_surfaces": ["all", "outer", "inner", "rim"],
                "auth": "oauth" if oauth else "server",
                "defaults": json.loads(LatticeSpec().to_json())}

    @app.get("/api/parts")
    def parts(did: str, wvm: str, wvmid: str, eid: str, token=Depends(user_token)):
        try:
            return make_client(token).list_parts(did, wvm, wvmid, eid)
        except OnshapeError as e:
            raise _onshape_failed(e)

    class OnshapeJob(BaseModel):
        did: str
        wvm: str = "w"
        wvmid: str
        eid: str
        part_id: str
        recipe: dict = {}

    @app.post("/api/jobs/onshape")
    def job_from_onshape(req: OnshapeJob, token=Depends(user_token)):
        try:
            stl = make_client(token).export_stl(req.did, req.wvm, req.wvmid, req.eid, req.part_id)
        except OnshapeError as e:
            raise _onshape_failed(e)
        return _submit(stl, req.recipe, "stl", source=req.model_dump(exclude={"recipe"}))

    @app.post("/api/jobs/upload")
    async def job_from_upload(file: UploadFile = File(...), recipe: str = Form("{}"),
                              token=Depends(user_token)):
        filetype = Path(file.filename or "shell.stl").suffix.lstrip(".").lower() or "stl"
        return _submit(await file.read(), json.loads(recipe), filetype)

    def _submit(data, recipe, filetype, source=None):
        try:
            job = queue.submit(data, recipe, filetype, source)
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))
        return job.public()

    # Shells for the live view: load once, then preview and build by id.

    class OnshapePart(BaseModel):
        did: str
        wvm: str = "w"
        wvmid: str
        eid: str
        part_id: str

    def _add_shell(data, filetype, source=None):
        try:
            return shells.add(data, filetype, source).public()
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.post("/api/shells/onshape")
    def shell_from_onshape(req: OnshapePart, token=Depends(user_token)):
        try:
            stl = make_client(token).export_stl(req.did, req.wvm, req.wvmid, req.eid, req.part_id)
        except OnshapeError as e:
            raise _onshape_failed(e)
        return _add_shell(stl, "stl", req.model_dump())

    @app.post("/api/shells/upload")
    async def shell_from_upload(file: UploadFile = File(...), token=Depends(user_token)):
        filetype = Path(file.filename or "shell.stl").suffix.lstrip(".").lower() or "stl"
        return _add_shell(await file.read(), filetype)

    def _shell(shell_id):
        shell = shells.get(shell_id)
        if not shell:
            raise HTTPException(404, "no such shell; load the part again")
        return shell

    @app.get("/api/shells/{shell_id}/mesh")
    def shell_mesh(shell_id: str):
        return Response(pack(_shell(shell_id).mesh), media_type="application/octet-stream")

    class PreviewReq(BaseModel):
        recipe: dict = {}
        detail: float = 1.0  # 1 = about 1.5 M voxels

    @app.post("/api/shells/{shell_id}/preview")
    def shell_preview(shell_id: str, req: PreviewReq, token=Depends(user_token)):
        """A coarse mesh of the lattice part, in the same packed format as /mesh."""
        shell = _shell(shell_id)
        try:
            mesh, info = shells.preview(shell, req.recipe, min(max(req.detail, 0.25), 4.0))
        except Superseded:
            raise HTTPException(409, "superseded by a newer preview")
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))
        return Response(pack(mesh), media_type="application/octet-stream",
                        headers={"X-Preview": json.dumps(info)})

    class ShellJob(BaseModel):
        recipe: dict = {}

    @app.post("/api/shells/{shell_id}/build")
    def shell_build(shell_id: str, req: ShellJob, token=Depends(user_token)):
        """The full-resolution build of a loaded shell (same job flow as /api/jobs)."""
        shell = _shell(shell_id)
        return _submit(shell.data, req.recipe, shell.filetype, shell.source)

    # Simulation: a static analysis or a topology optimisation of a loaded
    # shell. The result lands in the shell's preview cache too, so grading
    # "stress" or "topopt" with the same settings previews straight away.

    class SimReq(BaseModel):
        simulation: dict
        mode: str = "analysis"  # analysis | topopt
        topopt: dict = None

    @app.post("/api/shells/{shell_id}/simulate")
    def shell_simulate(shell_id: str, req: SimReq, token=Depends(user_token)):
        shell = _shell(shell_id)
        if req.mode not in ("analysis", "topopt"):
            raise HTTPException(422, "mode must be analysis or topopt")
        recipe = {"grading": "stress" if req.mode == "analysis" else "topopt",
                  "simulation": req.simulation, "topopt": req.topopt}
        try:
            sim = SimSpec.from_dict(req.simulation)
            sim.elastic()
            if not sim.fixed or not sim.loads:
                raise ValueError("add at least one fixed zone and one load")
            TopoSpec(**(req.topopt or {}))
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e))

        def work(job):
            spec = LatticeSpec.from_dict(recipe, log=job.log.append)
            _, res, values = simulate(shell.mesh, spec, shell.cache)
            job.report = dict(res.summary(), mode=req.mode)
            if req.mode == "analysis":
                mesh = shell.mesh
            else:
                mesh = density_mesh(res, values, shell.mesh)
                mesh.export(queue.result_path(job.id))
                job.report["volume_fraction"] = round(float(mesh.volume / shell.mesh.volume), 3)
            job.data = {"mesh": mesh, "displacement": res.sample(mesh.vertices, "displacement"),
                        "stress": res.sample(mesh.vertices, "stress")}
            for k in ("displacement", "stress"):
                v = job.data[k]
                job.report[f"{k}_range"] = [float(v.min()), float(v.max())]

        return queue.run(work, req.mode, shell.source).public()

    @app.get("/api/sims/{job_id}/mesh")
    def sim_mesh(job_id: str, field: str = "displacement"):
        """The analysed (or optimised) mesh, packed as /mesh, followed by one
        float32 per vertex: displacement in mm or von Mises stress in MPa."""
        job = queue.get(job_id)
        if not job or job.status != "done" or "mesh" not in job.data:
            raise HTTPException(404, "result not ready")
        if field not in ("displacement", "stress"):
            raise HTTPException(422, "field must be displacement or stress")
        values = np.ascontiguousarray(job.data[field], dtype="<f4")
        return Response(pack(job.data["mesh"]) + values.tobytes(), media_type="application/octet-stream")

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
    def job_import(job_id: str, translate: bool = True, token=Depends(user_token)):
        """Put the result back into the Onshape document it came from."""
        job = queue.get(job_id)
        if not job or job.status != "done":
            raise HTTPException(404, "result not ready")
        if not job.source or job.source.get("wvm") != "w":
            raise HTTPException(400, "only jobs started from an Onshape workspace can be imported")
        data = queue.result_path(job_id).read_bytes()
        try:
            return make_client(token).upload_file(job.source["did"], job.source["wvmid"],
                                                  f"lattice-{job_id}.3mf", data, translate=translate)
        except OnshapeError as e:
            raise _onshape_failed(e)

    def need_oauth():
        if not oauth:
            raise HTTPException(404, "OAuth is not configured on this service")
        return oauth

    @app.get("/oauth/login")
    def oauth_login(request: Request, state: str):
        """Opened by the panel in a popup; sends the user to Onshape to sign in."""
        o = need_oauth()
        try:
            return RedirectResponse(o.start(state, o.redirect_uri(request.base_url)))
        except OAuthError as e:
            raise HTTPException(400, str(e))

    @app.get("/oauth/callback", response_class=HTMLResponse)
    def oauth_callback(request: Request, state: str = "", code: str = "", error: str = ""):
        o = need_oauth()
        if error or not code:
            return CALLBACK_PAGE.format(message=f"Onshape sign-in failed: {html.escape(error or 'no code')}", script="")
        try:
            o.finish(state, code, o.redirect_uri(request.base_url))
        except OAuthError as e:
            return HTMLResponse(CALLBACK_PAGE.format(message=html.escape(str(e)), script=""), 400)
        return CALLBACK_PAGE.format(message="Signed in to Onshape. You can close this window.",
                                    script="setTimeout(() => window.close(), 800)")

    class StateReq(BaseModel):
        state: str

    @app.post("/oauth/poll")
    def oauth_poll(req: StateReq):
        """The tokens once the popup's sign-in is done; 202 while still waiting."""
        try:
            tokens = need_oauth().poll(req.state)
        except KeyError:
            raise HTTPException(404, "unknown or expired sign-in")
        return tokens or JSONResponse({"status": "pending"}, 202)

    class RefreshReq(BaseModel):
        refresh_token: str

    @app.post("/oauth/refresh")
    def oauth_refresh(req: RefreshReq):
        try:
            return need_oauth().refresh(req.refresh_token)
        except OAuthError as e:
            raise HTTPException(401, str(e))

    app.mount("/panel", StaticFiles(directory=Path(__file__).parent / "panel", html=True), name="panel")
    return app


app = make_app()
