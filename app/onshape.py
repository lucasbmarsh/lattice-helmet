"""Minimal Onshape REST client.

Inside Onshape the service passes each user's OAuth token (`token=`, see
app/auth.py). Without OAuth it falls back to API keys with HTTP Basic auth
(ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY). With ONSHAPE_AUTH=proxy no
credentials are sent, for environments whose egress proxy adds the
Authorization header.

list_parts, export_stl and upload_file were checked live against
cad.onshape.com on 2026-10-26.
"""

import os
import time

import httpx
import numpy as np

BASE_URL = os.environ.get("ONSHAPE_BASE_URL", "https://cad.onshape.com")


class OnshapeError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


_STL_FACET = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def _faces_to_stl(resp):
    """Binary STL (mm) from a /tessellatedfaces response (metres). Faces are
    tessellated separately but share their edge points, so merging equal
    vertices stitches them into a closed mesh."""
    import io

    import numpy as np
    import trimesh

    bodies = resp["bodies"] if isinstance(resp, dict) else resp
    def xyz(v):  # {"x", "y", "z"} in newer API versions, [x, y, z] in older ones
        return [v["x"], v["y"], v["z"]] if isinstance(v, dict) else v

    tris = [[xyz(v) for v in f["vertices"]] for b in bodies for face in b["faces"] for f in face["facets"]]
    if not tris:
        raise ValueError("no facets")
    t = np.asarray(tris, dtype=np.float64).reshape(-1, 3) * 1000.0
    mesh = trimesh.Trimesh(t, np.arange(len(t)).reshape(-1, 3), process=False)
    mesh.merge_vertices(digits_vertex=4)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        raise ValueError("tessellated faces did not stitch into a closed mesh")
    if mesh.volume < 0:
        mesh.invert()
    out = io.BytesIO()
    mesh.export(out, file_type="stl")
    return out.getvalue()


def _metres_to_mm(stl):
    n = int.from_bytes(stl[80:84], "little")
    if len(stl) != 84 + n * _STL_FACET.itemsize:
        raise OnshapeError("STL export is not binary STL")
    facets = np.frombuffer(stl, _STL_FACET, n, 84).copy()
    facets["v"] *= 1000
    return stl[:84] + facets.tobytes()


class Onshape:
    def __init__(self, access_key=None, secret_key=None, token=None, base_url=BASE_URL, timeout=120):
        if token:
            auth, headers = None, {"Authorization": f"Bearer {token}"}
        elif os.environ.get("ONSHAPE_AUTH") == "proxy":
            auth, headers = None, {}
        else:
            access_key = access_key or os.environ.get("ONSHAPE_ACCESS_KEY")
            secret_key = secret_key or os.environ.get("ONSHAPE_SECRET_KEY")
            if not (access_key and secret_key):
                raise OnshapeError("set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY, or pass an OAuth token, or set ONSHAPE_AUTH=proxy")
            auth, headers = (access_key, secret_key), {}
        headers["Accept"] = "application/json"
        self.http = httpx.Client(base_url=base_url, auth=auth, headers=headers,
                                 timeout=timeout, follow_redirects=True)

    def _get(self, path, **params):
        r = self.http.get(path, params=params)
        if r.status_code >= 400:
            raise OnshapeError(f"GET {path} -> {r.status_code}: {r.text[:300]}", r.status_code)
        return r

    def session_info(self):
        """The signed-in user: {"id", "email", "name", ...}."""
        return self._get("/api/users/sessioninfo").json()

    def list_parts(self, did, wvm, wvmid, eid):
        """Parts in a Part Studio: [{"partId", "name"}, ...]."""
        r = self._get(f"/api/parts/d/{did}/{wvm}/{wvmid}/e/{eid}")
        return [{"partId": p["partId"], "name": p.get("name", p["partId"])} for p in r.json()]

    def export_stl(self, did, wvm, wvmid, eid, part_id, poll=1.0, max_wait=300):
        """Binary STL of one part, in millimetres, tessellated finely.

        The translation API ignores tessellation settings and gives coarse
        facets (a helmet came out at 2,000 triangles, which show as flats in
        the lattice), so ask for the part's faces at 0.2 mm chord tolerance
        and stitch them; fall back to the translation API if that fails.
        """
        try:
            r = self._get(f"/api/partstudios/d/{did}/{wvm}/{wvmid}/e/{eid}/tessellatedfaces",
                          partId=part_id, chordTolerance=0.0002, angleTolerance=0.1,
                          outputFaceAppearances="false")
            return _faces_to_stl(r.json())
        except (OnshapeError, KeyError, ValueError):
            return self._translate_stl(did, wvm, wvmid, eid, part_id, poll, max_wait)

    def _translate_stl(self, did, wvm, wvmid, eid, part_id, poll=1.0, max_wait=300):
        """Binary STL of one part via the translation API, in millimetres.

        The translation API writes STL in metres whatever `unit` says, so the
        vertices are scaled by 1000 here.

        Uses the async translation API rather than the synchronous .../stl
        endpoint, which redirects to a regional host (cad-usw2.onshape.com
        /modelexport) that a proxy scoped to cad.onshape.com/api/ refuses.
        """
        r = self.http.post(
            f"/api/partstudios/d/{did}/{wvm}/{wvmid}/e/{eid}/translations",
            json={"formatName": "STL", "partIds": part_id, "storeInDocument": False,
                  "stlMode": "BINARY", "grouping": True},
        )
        if r.status_code >= 400:
            raise OnshapeError(f"STL export -> {r.status_code}: {r.text[:300]}", r.status_code)
        t = r.json()
        deadline = time.monotonic() + max_wait
        while t.get("requestState") == "ACTIVE":
            if time.monotonic() > deadline:
                raise OnshapeError(f"STL export timed out after {max_wait}s")
            time.sleep(poll)
            t = self._get(f"/api/translations/{t['id']}").json()
        if t.get("requestState") != "DONE":
            raise OnshapeError(f"STL export failed: {t.get('failureReason')}")
        r = self.http.get(f"/api/documents/d/{did}/externaldata/{t['resultExternalDataIds'][0]}",
                          headers={"Accept": "application/octet-stream"})
        if r.status_code >= 400:
            raise OnshapeError(f"STL download -> {r.status_code}: {r.text[:300]}", r.status_code)
        return _metres_to_mm(r.content)

    def upload_file(self, did, wid, filename, data, translate=False):
        """Add a file to the document as a blob element.

        translate=True asks Onshape to import it, so a mesh lands in a new
        Part Studio as a preview.
        """
        r = self.http.post(
            f"/api/blobelements/d/{did}/w/{wid}",
            files={"file": (filename, data, "application/octet-stream")},
            data={"encodedFilename": filename, "translate": str(translate).lower()},
        )
        if r.status_code >= 400:
            raise OnshapeError(f"upload -> {r.status_code}: {r.text[:300]}", r.status_code)
        return r.json()
