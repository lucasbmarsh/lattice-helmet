import io
import time

import pytest
import trimesh
from fastapi.testclient import TestClient

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
    return TestClient(make_app(onshape_factory=FakeOnshape, workdir=tmp_path))


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


def test_stl_metres_to_mm():
    box = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    out = _metres_to_mm(box.export(file_type="stl"))
    scaled = trimesh.load(io.BytesIO(out), file_type="stl")
    assert scaled.extents == pytest.approx((100, 200, 300))
