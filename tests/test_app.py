import io
import json
import struct
import time

import httpx
import pytest
import trimesh
from fastapi.testclient import TestClient

from app.auth import OAuth
from app.main import make_app
from app.onshape import _metres_to_mm

FAST = {"cell": 7.5, "pitch": 1.0, "wall_min": 1.2, "wall_max": 1.2, "cell_map": "conformal",
        "skin_surfaces": ["outer", "inner"]}


def _dome_stl():
    outer = trimesh.creation.icosphere(3, radius=60)
    inner = trimesh.creation.icosphere(3, radius=45)
    cut = trimesh.creation.box((200, 200, 100))
    cut.apply_translation((0, 0, -50))
    dome = outer.difference(inner, engine="manifold").difference(cut, engine="manifold")
    return dome.export(file_type="stl")


class FakeOnshape:
    uploads = []

    def __init__(self, token=None):
        self.token = token

    def session_info(self):
        return {"id": "u1", "email": {"tok-lucas": "Lucas@example.com"}.get(self.token, "other@example.com")}

    def list_parts(self, did, wvm, wvmid, eid):
        return [{"partId": "JHD", "name": "Liner"}]

    def export_stl(self, did, wvm, wvmid, eid, part_id):
        assert part_id == "JHD"
        return _dome_stl()

    def upload_file(self, did, wid, filename, data, translate=False):
        self.uploads.append((did, wid, filename, len(data), translate))
        return {"id": "blob1"}


@pytest.fixture
def client(tmp_path):
    return TestClient(make_app(onshape_factory=FakeOnshape, workdir=tmp_path, oauth=None))


def _wait(client, job_id):
    for _ in range(300):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.5)
    raise TimeoutError


def test_options_and_panel(client):
    o = client.get("/api/options").json()
    assert "gyroid" in o["lattices"] and "conformal" in o["cell_maps"]
    assert "Lattice helmet" in client.get("/panel/").text


def test_onshape_round_trip(client):
    assert client.get("/api/parts", params=dict(did="d", wvm="w", wvmid="w1", eid="e")).json()[0]["name"] == "Liner"
    job = client.post("/api/jobs/onshape", json=dict(did="d", wvmid="w1", eid="e", part_id="JHD", recipe=FAST)).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done", j
    assert j["report"]["watertight"]
    r = client.get(f"/api/jobs/{job['id']}/result.3mf")
    assert r.status_code == 200 and len(r.content) > 1000
    assert client.post(f"/api/jobs/{job['id']}/import").status_code == 200
    assert FakeOnshape.uploads[-1][:2] == ("d", "w1")


def test_upload_and_bad_recipe(client):
    files = {"file": ("shell.stl", io.BytesIO(_dome_stl()), "model/stl")}
    bad = client.post("/api/jobs/upload", files=files, data={"recipe": '{"bogus": 1}'})
    assert bad.status_code == 422
    files = {"file": ("shell.stl", io.BytesIO(_dome_stl()), "model/stl")}
    ok = client.post("/api/jobs/upload", files=files, data={"recipe": __import__("json").dumps(FAST)}).json()
    assert _wait(client, ok["id"])["status"] == "done"


def _unpack(data):
    nv, nf = struct.unpack_from("<II", data)
    assert len(data) == 8 + 12 * nv + 12 * nf
    return nv, nf


def test_live_preview_then_build(client):
    files = {"file": ("shell.stl", io.BytesIO(_dome_stl()), "model/stl")}
    s = client.post("/api/shells/upload", files=files).json()
    assert s["watertight"] and s["bounds_mm"][1][2] == pytest.approx(60, abs=1)
    assert _unpack(client.get(f"/api/shells/{s['id']}/mesh").content)[1] == s["triangles"]

    r = client.post(f"/api/shells/{s['id']}/preview", json={"recipe": FAST, "detail": 0.25})
    assert r.status_code == 200 and _unpack(r.content)[1] == json.loads(r.headers["X-Preview"])["triangles"]
    assert client.post(f"/api/shells/{s['id']}/preview", json={"recipe": {"lattice": "nope"}}).status_code == 422
    assert client.post("/api/shells/nope/preview", json={}).status_code == 404

    job = client.post(f"/api/shells/{s['id']}/build", json={"recipe": FAST}).json()
    assert _wait(client, job["id"])["status"] == "done"


def test_onshape_shell_keeps_its_source_for_import(client):
    s = client.post("/api/shells/onshape", json=dict(did="d", wvmid="w1", eid="e", part_id="JHD")).json()
    assert s["source"]["did"] == "d"
    job = client.post(f"/api/shells/{s['id']}/build", json={"recipe": FAST}).json()
    assert _wait(client, job["id"])["status"] == "done"
    assert client.post(f"/api/jobs/{job['id']}/import").status_code == 200


def test_stl_metres_to_mm():
    box = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    out = _metres_to_mm(box.export(file_type="stl"))
    scaled = trimesh.load(io.BytesIO(out), file_type="stl")
    assert scaled.extents == pytest.approx((100, 200, 300))


def _token_endpoint(request):
    form = dict(httpx.QueryParams(request.content.decode()))
    assert form["client_id"] == "cid" and form["client_secret"] == "shh"
    if form["grant_type"] == "authorization_code":
        assert form["redirect_uri"] == "https://lattice.example/oauth/callback"
        token = {"good-code": "tok-lucas", "other-code": "tok-other"}[form["code"]]
        return httpx.Response(200, json={"access_token": token, "refresh_token": "r1", "expires_in": 3600})
    assert form["grant_type"] == "refresh_token" and form["refresh_token"] == "r1"
    return httpx.Response(200, json={"access_token": "tok-lucas", "expires_in": 3600})


@pytest.fixture
def oauth_client(tmp_path):
    oauth = OAuth("cid", "shh", ["lucas@example.com"], public_url="https://lattice.example",
                  http=httpx.Client(transport=httpx.MockTransport(_token_endpoint)))
    return TestClient(make_app(onshape_factory=FakeOnshape, workdir=tmp_path, oauth=oauth))


def _sign_in(client, code, state="s" * 30):
    login = client.get("/oauth/login", params={"state": state}, follow_redirects=False)
    assert login.status_code == 307
    assert login.headers["location"].startswith("https://oauth.onshape.com/oauth/authorize?")
    assert "client_id=cid" in login.headers["location"]
    assert client.post("/oauth/poll", json={"state": state}).status_code == 202
    page = client.get("/oauth/callback", params={"state": state, "code": code})
    assert "Signed in" in page.text
    tokens = client.post("/oauth/poll", json={"state": state}).json()
    assert client.post("/oauth/poll", json={"state": state}).status_code == 404  # handed over once
    return tokens


def test_oauth_sign_in_and_allow_list(oauth_client):
    c = oauth_client
    assert c.get("/api/options").json()["auth"] == "oauth"
    ids = dict(did="d", wvm="w", wvmid="w1", eid="e")
    assert c.get("/api/parts", params=ids).status_code == 401
    assert c.post("/api/shells/onshape", json=dict(ids, part_id="JHD")).status_code == 401
    assert c.get("/oauth/login", params={"state": "short"}).status_code == 400

    tokens = _sign_in(c, "good-code")
    assert tokens["access_token"] == "tok-lucas" and tokens["refresh_token"] == "r1"
    auth = {"Authorization": "Bearer tok-lucas"}
    assert c.get("/api/parts", params=ids, headers=auth).json()[0]["name"] == "Liner"
    assert c.post("/oauth/refresh", json={"refresh_token": "r1"}).json()["refresh_token"] == "r1"

    other = _sign_in(c, "other-code", state="t" * 30)
    r = c.get("/api/parts", params=ids, headers={"Authorization": f"Bearer {other['access_token']}"})
    assert r.status_code == 403 and "other@example.com" in r.json()["detail"]


def test_oauth_callback_rejects_unknown_state(oauth_client):
    r = oauth_client.get("/oauth/callback", params={"state": "x" * 30, "code": "good-code"})
    assert r.status_code == 400


SIM = {"material": "PETG", "voxel": 4.0,
       "fixed": [{"where": "8 - z"}], "loads": [{"sphere": [0, 0, 60, 15], "force": [0, 0, -500]}]}


def test_simulation_then_stress_graded_preview(client):
    files = {"file": ("shell.stl", io.BytesIO(_dome_stl()), "model/stl")}
    s = client.post("/api/shells/upload", files=files).json()
    recipe = dict(FAST, grading="stress", simulation=SIM, wall_min=0.6, wall_max=2.0)
    # Not analysed yet: the preview asks for the analysis instead of running it.
    r = client.post(f"/api/shells/{s['id']}/preview", json={"recipe": recipe, "detail": 0.25})
    assert r.status_code == 422 and "Run analysis" in r.json()["detail"]
    assert client.post(f"/api/shells/{s['id']}/simulate", json={"simulation": {"fixed": []}}).status_code == 422

    job = client.post(f"/api/shells/{s['id']}/simulate", json={"simulation": SIM}).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done", j
    assert j["report"]["max_displacement_mm"] > 0 and j["report"]["mode"] == "analysis"
    data = client.get(f"/api/sims/{job['id']}/mesh", params={"field": "stress"}).content
    nv, nf = struct.unpack_from("<II", data)
    assert len(data) == 8 + 12 * nv + 12 * nf + 4 * nv
    r = client.post(f"/api/shells/{s['id']}/preview", json={"recipe": recipe, "detail": 0.25})
    assert r.status_code == 200


def test_topopt_job_gives_a_part(client):
    s = client.post("/api/shells/onshape", json=dict(did="d", wvmid="w1", eid="e", part_id="JHD")).json()
    job = client.post(f"/api/shells/{s['id']}/simulate",
                      json={"simulation": SIM, "mode": "topopt", "topopt": {"volume_fraction": 0.4, "iterations": 4}}).json()
    j = _wait(client, job["id"])
    assert j["status"] == "done", j
    assert 0.2 < j["report"]["volume_fraction"] < 0.7
    assert client.get(f"/api/jobs/{job['id']}/result.3mf").status_code == 200
    assert client.post(f"/api/jobs/{job['id']}/import").status_code == 200


def test_zones_and_new_lattices_in_options(client):
    o = client.get("/api/options").json()
    assert {"voronoi", "honeycomb_tri", "honeycomb_hex"} <= set(o["lattices"])
    assert "stress" in o["gradings"] and "PETG" in o["materials"]
