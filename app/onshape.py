"""Minimal Onshape REST client.

Uses API keys with HTTP Basic auth for now (ONSHAPE_ACCESS_KEY and
ONSHAPE_SECRET_KEY). An OAuth bearer token can replace them once the app is
registered, by passing `token=`.

Endpoint paths follow Onshape's public REST API but have not yet been run
against cad.onshape.com from this environment; check them in the first live
test.
"""

import os

import httpx

BASE_URL = os.environ.get("ONSHAPE_BASE_URL", "https://cad.onshape.com")


class OnshapeError(RuntimeError):
    pass


class Onshape:
    def __init__(self, access_key=None, secret_key=None, token=None, base_url=BASE_URL, timeout=120):
        if token:
            auth, headers = None, {"Authorization": f"Bearer {token}"}
        else:
            access_key = access_key or os.environ.get("ONSHAPE_ACCESS_KEY")
            secret_key = secret_key or os.environ.get("ONSHAPE_SECRET_KEY")
            if not (access_key and secret_key):
                raise OnshapeError("set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY, or pass an OAuth token")
            auth, headers = (access_key, secret_key), {}
        headers["Accept"] = "application/json"
        self.http = httpx.Client(base_url=base_url, auth=auth, headers=headers,
                                 timeout=timeout, follow_redirects=True)

    def _get(self, path, **params):
        r = self.http.get(path, params=params)
        if r.status_code >= 400:
            raise OnshapeError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
        return r

    def list_parts(self, did, wvm, wvmid, eid):
        """Parts in a Part Studio: [{"partId", "name"}, ...]."""
        r = self._get(f"/api/parts/d/{did}/{wvm}/{wvmid}/e/{eid}")
        return [{"partId": p["partId"], "name": p.get("name", p["partId"])} for p in r.json()]

    def export_stl(self, did, wvm, wvmid, eid, part_id):
        """Binary STL of one part, in millimetres."""
        r = self.http.get(
            f"/api/partstudios/d/{did}/{wvm}/{wvmid}/e/{eid}/stl",
            params={"partIds": part_id, "mode": "binary", "units": "millimeter", "grouping": "true"},
            headers={"Accept": "application/vnd.onshape.v1+octet-stream"},
        )
        if r.status_code >= 400:
            raise OnshapeError(f"STL export -> {r.status_code}: {r.text[:300]}")
        return r.content

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
            raise OnshapeError(f"upload -> {r.status_code}: {r.text[:300]}")
        return r.json()
