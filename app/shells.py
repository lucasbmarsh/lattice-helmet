"""Shells the panel is looking at, kept in memory so the live preview can
re-run on every settings change without re-uploading or re-exporting.

Each shell keeps the preview's SDF cache, so after the first preview only
the lattice itself is recomputed.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import io
import struct
import threading
import uuid

import numpy as np
import trimesh

from lattice_helmet.build import SIM_GRADINGS, LatticeSpec, axis_name, preview, sim_key
from lattice_helmet.surfaces import detect_up


@dataclass
class Shell:
    id: str
    data: bytes  # the file as uploaded or exported, for the full build
    filetype: str
    mesh: trimesh.Trimesh
    source: dict = None  # Onshape ids, when it came from Onshape
    cache: dict = field(default_factory=dict)

    def public(self):
        if "up" not in self.cache:
            self.cache["up"] = detect_up(self.mesh)  # the same cache key build.resolve_up reads
        return {"id": self.id, "triangles": len(self.mesh.faces), "watertight": bool(self.mesh.is_watertight),
                "up": axis_name(self.cache["up"]),
                "bounds_mm": np.round(self.mesh.bounds, 1).tolist(), "source": self.source}


def pack(mesh):
    """Mesh as the panel's viewer reads it: little-endian uint32 vertex and
    face counts, then float32 xyz per vertex, then uint32 indices per face."""
    v = np.ascontiguousarray(mesh.vertices, dtype="<f4")
    f = np.ascontiguousarray(mesh.faces, dtype="<u4")
    return struct.pack("<II", len(v), len(f)) + v.tobytes() + f.tobytes()


class Superseded(Exception):
    """A newer preview of the same shell arrived while this one waited."""


class ShellStore:
    def __init__(self, keep=4):
        self.keep = keep  # each shell's SDF cache is tens of MB, so keep a few
        self.shells = OrderedDict()
        self.lock = threading.Lock()
        self.preview_lock = threading.Lock()  # one preview at a time
        self.latest = {}  # shell id -> number of its newest preview request

    def add(self, data, filetype="stl", source=None):
        mesh = trimesh.load(io.BytesIO(data), file_type=filetype, force="mesh")
        if not len(mesh.faces):
            raise ValueError("the file has no triangles")
        shell = Shell(uuid.uuid4().hex[:12], data, filetype, mesh, source)
        with self.lock:
            self.shells[shell.id] = shell
            while len(self.shells) > self.keep:
                self.shells.popitem(last=False)
        return shell

    def get(self, shell_id):
        with self.lock:
            shell = self.shells.get(shell_id)
            if shell:
                self.shells.move_to_end(shell_id)
            return shell

    def preview(self, shell, recipe, detail=1.0):
        """Run a preview; raises Superseded if a newer one for this shell came in
        first, so a burst of slider changes only computes the last one."""
        spec = LatticeSpec.from_dict(recipe, log=lambda *a: None)
        if spec.grading in SIM_GRADINGS and sim_key(spec) not in shell.cache:
            # An analysis takes seconds and an optimisation minutes: too long
            # to hold up the live view, so the panel runs them as jobs first.
            what = "Run analysis" if spec.grading == "stress" else "Optimise shape"
            raise ValueError(f"press {what} under Simulation first; the preview then uses its result")
        with self.lock:
            n = self.latest.get(shell.id, 0) + 1
            self.latest[shell.id] = n
        with self.preview_lock:
            if self.latest.get(shell.id) != n:
                raise Superseded()
            return preview(shell.mesh, spec, voxels=1.5e6 * detail, max_faces=int(400_000 * detail),
                           cache=shell.cache)
