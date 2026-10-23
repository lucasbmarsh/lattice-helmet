"""Unit cells and field helpers.

A unit cell is evaluated in lattice coordinates (u, v, w), which a cell map
produces from world coordinates. Each cell returns a raw field R; build.py
turns it into a world-space distance with R / |grad R|, so the same cell works
under any map and wall thickness stays in mm.
"""

import numpy as np

_TAU = np.float32(2 * np.pi)


def _trig(u, v, w, cell):
    k = _TAU / np.float32(cell)
    return np.sin(k * u), np.cos(k * u), np.sin(k * v), np.cos(k * v), np.sin(k * w), np.cos(k * w)


def gyroid(u, v, w, cell):
    sx, cx, sy, cy, sz, cz = _trig(u, v, w, cell)
    return sx * cy + sy * cz + sz * cx


def schwarz_p(u, v, w, cell):
    _, cx, _, cy, _, cz = _trig(u, v, w, cell)
    return cx + cy + cz


def diamond(u, v, w, cell):
    sx, cx, sy, cy, sz, cz = _trig(u, v, w, cell)
    return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz


def neovius(u, v, w, cell):
    _, cx, _, cy, _, cz = _trig(u, v, w, cell)
    return 3 * (cx + cy + cz) + 4 * cx * cy * cz


TPMS = {"gyroid": gyroid, "schwarz_p": schwarz_p, "diamond": diamond, "neovius": neovius}

# |f| / |grad f| underestimates distance away from the surface, so sheets come
# out thin. These factors were measured against the known surface area per cell
# (gyroid 3.092, Schwarz P 2.345, diamond 3.838 per a^2) at 1 mm walls.
SHEET_CAL = {"gyroid": 1.055, "schwarz_p": 1.025, "diamond": 1.075, "neovius": 1.05}


# Strut graphs in unit-cell coordinates [0, 1]^3, as (start, end) pairs.
_C = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
_MID = (0.5, 0.5, 0.5)
_FACES = [(0.5, 0.5, 0), (0.5, 0.5, 1), (0.5, 0, 0.5), (0.5, 1, 0.5), (0, 0.5, 0.5), (1, 0.5, 0.5)]


def _cube_edges():
    return [(a, b) for i, a in enumerate(_C) for b in _C[i + 1:] if sum(x != y for x, y in zip(a, b)) == 1]


def _face_diagonals():
    out = []
    for f in _FACES:
        axis = [i for i in range(3) if f[i] in (0, 1)][0]
        corners = [c for c in _C if c[axis] == f[axis]]
        out += [(c, f) for c in corners]
    return out


def _octahedron():
    return [(a, b) for i, a in enumerate(_FACES) for b in _FACES[i + 1:]
            if sum(x != y for x, y in zip(a, b)) == 2]


GRAPHS = {
    "sc": _cube_edges(),
    "bcc": [(c, _MID) for c in _C],
    "fcc": _face_diagonals(),
    "octet": _face_diagonals() + _octahedron(),
}


def graph_distance(u, v, w, cell, segments):
    """Distance (lattice units) from each point to the nearest strut axis."""
    c = np.float32(cell)
    p = [np.mod(a, c) / c for a in (u, v, w)]  # position inside the cell, [0, 1)
    seg = np.asarray(segments, dtype=np.float32)
    best = None
    for a, b in seg:
        d = b - a
        dd = float(d @ d)
        q = [p[i] - a[i] for i in range(3)]
        t = np.clip((q[0] * d[0] + q[1] * d[1] + q[2] * d[2]) / dd, 0, 1)
        r2 = sum((q[i] - t * d[i]) ** 2 for i in range(3))
        best = r2 if best is None else np.minimum(best, r2)
    return np.sqrt(best) * c


def lattice_kinds():
    sheets = sorted(TPMS)
    return sheets + [f"{s}_network" for s in sheets] + sorted(GRAPHS) + ["custom"]


def raw_field(kind, u, v, w, cell, graph=None):
    """Return (R, (mode, cal)) for a lattice kind.

    mode says how R becomes a distance; cal scales the requested wall.
    """
    if kind in TPMS:
        return TPMS[kind](u, v, w, cell), ("sheet", SHEET_CAL[kind])
    if kind.endswith("_network") and kind[:-8] in TPMS:
        return TPMS[kind[:-8]](u, v, w, cell), ("network", 1.0)
    if kind in GRAPHS:
        return graph_distance(u, v, w, cell, GRAPHS[kind]), ("beam", 1.0)
    if kind == "custom":
        if not graph:
            raise ValueError("lattice 'custom' needs a strut graph")
        return graph_distance(u, v, w, cell, graph), ("beam", 1.0)
    raise ValueError(f"unknown lattice {kind!r}; choose from {lattice_kinds()}")


def to_distance(R, grad, mode, wall, cell):
    """World-space signed distance (mm, negative = material)."""
    mode, cal = mode
    wall = wall * cal
    if mode == "sheet":
        g = np.maximum(grad, 0.1 * _TAU / cell)
        return np.abs(R) / g - 0.5 * wall
    if mode == "network":
        # wall = offset of the solid's boundary from the minimal surface, mm
        g = np.maximum(grad, 0.1 * _TAU / cell)
        return R / g - wall
    if mode == "beam":
        return R / np.maximum(grad, 0.2) - 0.5 * wall
    raise ValueError(mode)


def smooth_min(a, b, r):
    """Polynomial smooth union; r is the blend (fillet) radius in mm."""
    if r <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / r, 0.0, 1.0)
    return b + (a - b) * h - r * h * (1.0 - h)
