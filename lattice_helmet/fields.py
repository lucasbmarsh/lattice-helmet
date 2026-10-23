"""Lattice and grading fields.

Every lattice function returns an approximate signed distance in mm (negative
= material), so lattices, skins and booleans can be combined with plain
min/max like any other implicit solid.
"""

import numpy as np

_EPS = 1e-6


def _trig(x, y, z, cell):
    k = np.float32(2 * np.pi / cell)
    kx, ky, kz = k * x, k * y, k * z
    return k, np.sin(kx), np.cos(kx), np.sin(ky), np.cos(ky), np.sin(kz), np.cos(kz)


def gyroid(x, y, z, cell):
    k, sx, cx, sy, cy, sz, cz = _trig(x, y, z, cell)
    f = sx * cy + sy * cz + sz * cx
    gx = cx * cy - sz * sx
    gy = cy * cz - sx * sy
    gz = cz * cx - sy * sz
    return f, k * np.sqrt(gx * gx + gy * gy + gz * gz)


def schwarz_p(x, y, z, cell):
    k, sx, cx, sy, cy, sz, cz = _trig(x, y, z, cell)
    f = cx + cy + cz
    return f, k * np.sqrt(sx * sx + sy * sy + sz * sz)


def diamond(x, y, z, cell):
    k, sx, cx, sy, cy, sz, cz = _trig(x, y, z, cell)
    f = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
    gx = cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz
    gy = sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz
    gz = sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz
    return f, k * np.sqrt(gx * gx + gy * gy + gz * gz)


TPMS = {"gyroid": gyroid, "schwarz_p": schwarz_p, "diamond": diamond}


def tpms_sheet(kind, x, y, z, cell, wall):
    """Sheet TPMS: a wall of thickness `wall` (mm, scalar or field) around f = 0."""
    f, g = TPMS[kind](x, y, z, cell)
    return np.abs(f) / np.maximum(g, _EPS) - 0.5 * wall


def bcc_beams(x, y, z, cell, wall):
    """Body-centred cubic strut lattice; `wall` is the strut diameter (mm)."""
    c = np.float32(cell)
    px = np.mod(x, c) - 0.5 * c
    py = np.mod(y, c) - 0.5 * c
    pz = np.mod(z, c) - 0.5 * c
    r2 = px * px + py * py + pz * pz
    best = None
    s = np.float32(1 / np.sqrt(3))
    for ux, uy, uz in ((1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)):
        t = (px * ux + py * uy + pz * uz) * s
        d2 = r2 - t * t
        best = d2 if best is None else np.minimum(best, d2)
    return np.sqrt(np.maximum(best, 0)) - 0.5 * wall


LATTICES = {
    "gyroid": lambda x, y, z, c, w: tpms_sheet("gyroid", x, y, z, c, w),
    "diamond": lambda x, y, z, c, w: tpms_sheet("diamond", x, y, z, c, w),
    "schwarz_p": lambda x, y, z, c, w: tpms_sheet("schwarz_p", x, y, z, c, w),
    "bcc": bcc_beams,
}


def smooth_min(a, b, r):
    """Polynomial smooth union; r is the blend (fillet) radius in mm."""
    if r <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / r, 0.0, 1.0)
    return b + (a - b) * h - r * h * (1.0 - h)


def grading(mode, x, y, z, sdf, bounds, point=None, depth_range=None):
    """Scalar field in [0, 1] that drives variable wall thickness.

    uniform: 0 everywhere
    z:       0 at the bottom of the part, 1 at the top
    depth:   0 at the part surface, 1 at `depth_range` mm or deeper
    point:   0 at `point` (e.g. an impact zone), 1 at the farthest corner
    """
    lo, hi = bounds
    if mode == "uniform":
        return np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), np.float32)
    if mode == "z":
        t = (z - lo[2]) / max(hi[2] - lo[2], _EPS)
        return np.broadcast_to(np.clip(t, 0, 1), sdf.shape).astype(np.float32)
    if mode == "depth":
        dr = depth_range or 10.0
        return np.clip(-sdf / dr, 0, 1).astype(np.float32)
    if mode == "point":
        p = np.asarray(point, dtype=np.float32)
        d = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2)
        far = np.max(np.linalg.norm(np.array([lo, hi]) - p, axis=1))
        return np.clip(d / max(far, _EPS), 0, 1).astype(np.float32)
    raise ValueError(f"unknown grading mode {mode!r}")
