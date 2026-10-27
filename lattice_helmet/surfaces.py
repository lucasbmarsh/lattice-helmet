"""Split a helmet shell's surface into outer, inner and rim, and measure
distance to each on the grid.

A face is "outer" when its normal points away from the helmet centre, "inner"
when it points toward it, and "rim" otherwise (edges, brow and ear cut-outs).
"""

import numpy as np
import trimesh
from scipy import ndimage

SURFACES = ("outer", "inner", "rim")


AXES = {f"{s}{a}": (i, 1 if s == "+" else -1) for i, a in enumerate("xyz") for s in "+-"}


def parse_up(up):
    """'+z', '-y', 'x' ... as (axis index, sign)."""
    up = up if up[0] in "+-" else "+" + up
    if up not in AXES:
        raise ValueError(f"up must be one of {sorted(AXES)} or 'auto'")
    return AXES[up]


def detect_up(mesh, n=64):
    """Which way the crown points, as (axis index, sign).

    The head goes in through the biggest opening, so look along each of the
    six axis directions from the middle of the bounding box: the direction
    in which most straight lines escape without crossing the shell is down.
    """
    from .sdf import inside_mask, make_grid

    pitch = float(np.ptp(mesh.bounds, axis=0).max()) / n
    grid = make_grid(mesh, pitch, margin=pitch)
    solid = inside_mask(mesh, grid)
    c = [s // 2 for s in solid.shape]
    best, down = -1.0, (2, -1)
    for i in range(3):
        o = [j for j in range(3) if j != i]
        # a bundle of lines through the middle 40% of the other two axes
        sl = [slice(None)] * 3
        for j in o:
            w = max(1, int(0.2 * solid.shape[j]))
            sl[j] = slice(c[j] - w, c[j] + w)
        for sign in (-1, 1):
            sl[i] = slice(0, c[i]) if sign < 0 else slice(c[i], None)
            free = ~solid[tuple(sl)].any(axis=i)
            score = float(free.mean())
            # prefer -z, then -y, on a tie (the usual CAD conventions)
            score += 1e-3 * ((i == 2) + 0.5 * (i == 1)) * (sign < 0)
            if score > best:
                best, down = score, (i, sign)
    return down[0], -down[1]


def default_center(mesh, up=(2, 1)):
    """Centre of the head: the centre of the sphere that best fits the
    shell's surface (area-weighted least squares), kept inside the bounding
    box. For a plain dome that is the middle of the rim; for a helmet with
    ear flaps it sits near ear level, so radial directions stay close to the
    surface normal over the flaps too."""
    p = mesh.triangles_center
    w = mesh.area_faces / mesh.area_faces.sum()
    # |p|^2 = 2 c.p + (r^2 - |c|^2): linear in c and k
    A = np.hstack([2 * p, np.ones((len(p), 1))]) * np.sqrt(w)[:, None]
    b = (p * p).sum(axis=1) * np.sqrt(w)
    c = np.linalg.lstsq(A, b, rcond=None)[0][:3]
    lo, hi = mesh.bounds
    return np.clip(c, lo, hi)


def classify_faces(mesh, center, threshold=0.7):
    d = mesh.triangles_center - center
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    dot = np.einsum("ij,ij->i", d, mesh.face_normals)
    labels = np.full(len(mesh.faces), "rim", dtype=object)
    labels[dot > threshold] = "outer"
    labels[dot < -threshold] = "inner"
    return labels


def surface_distances(mesh, grid, center, smooth=1.0):
    """Unsigned distance (mm) from every voxel to the outer, inner and rim surfaces.

    Each surface is resampled finer than a voxel, its points mark the voxels
    they fall in, and a Euclidean distance transform spreads the distance out.
    """
    labels = classify_faces(mesh, center)
    out = {}
    for name in SURFACES:
        sel = np.nonzero(labels == name)[0]
        if len(sel) == 0:
            out[name] = None
            continue
        sub = mesh.submesh([sel], append=True)
        v, _ = trimesh.remesh.subdivide_to_size(sub.vertices, sub.faces, max_edge=0.7 * grid.pitch)
        idx = np.round((v - grid.origin) / grid.pitch).astype(np.int64)
        idx = np.clip(idx, 0, np.array(grid.shape) - 1)
        seed = np.ones(grid.shape, dtype=bool)
        seed[idx[:, 0], idx[:, 1], idx[:, 2]] = False
        d = ndimage.distance_transform_edt(seed).astype(np.float32) * grid.pitch
        if smooth:
            d = ndimage.gaussian_filter(d, sigma=smooth)
        out[name] = d
    return out
