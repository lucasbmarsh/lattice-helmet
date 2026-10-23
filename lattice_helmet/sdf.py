"""Turn a closed triangle mesh into a signed distance field on a regular grid.

Convention used everywhere in this package: negative inside, positive outside,
distances in millimetres.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Grid:
    origin: np.ndarray  # (3,) world position of voxel (0, 0, 0), mm
    pitch: float  # voxel size, mm
    shape: tuple  # (nx, ny, nz)

    def axis(self, i):
        return self.origin[i] + self.pitch * np.arange(self.shape[i], dtype=np.float32)

    def coords(self, k0=0, k1=None):
        """Broadcastable x, y, z coordinate arrays for z-slab [k0, k1)."""
        k1 = self.shape[2] if k1 is None else k1
        x = self.axis(0)[:, None, None]
        y = self.axis(1)[None, :, None]
        z = self.axis(2)[k0:k1][None, None, :]
        return x, y, z


def make_grid(mesh, pitch, margin=None):
    margin = 3 * pitch if margin is None else margin
    lo = mesh.bounds[0] - margin
    hi = mesh.bounds[1] + margin
    shape = tuple(int(n) for n in np.ceil((hi - lo) / pitch).astype(int) + 1)
    return Grid(origin=lo.astype(np.float64), pitch=float(pitch), shape=shape)


def inside_mask(mesh, grid):
    """Voxel inside/outside test by z-ray parity, one ray per (x, y) column.

    Every triangle is rasterised onto the columns its xy-projection covers, the
    crossing heights are binned into z cells, and a running parity along z gives
    the inside set. Column centres are nudged by an irrational fraction of a
    voxel so a ray never lands exactly on a shared edge or vertex.
    """
    nx, ny, nz = grid.shape
    p = grid.pitch
    jitter = np.array([np.sqrt(2) - 1.0, np.sqrt(3) - 1.5]) * 1e-3 * p
    o = grid.origin[:2] + jitter

    tri = mesh.triangles.astype(np.float64)  # (m, 3, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

    # Column-index bounding box of each triangle's projection.
    lo = np.ceil((tri[:, :, :2].min(axis=1) - o) / p).astype(np.int64)
    hi = np.floor((tri[:, :, :2].max(axis=1) - o) / p).astype(np.int64)
    lo = np.clip(lo, 0, [nx - 1, ny - 1])
    hi = np.clip(hi, -1, [nx - 1, ny - 1])
    wx = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0)
    wy = np.maximum(hi[:, 1] - lo[:, 1] + 1, 0)
    counts = wx * wy

    toggles = np.zeros((nx, ny, nz + 1), dtype=np.uint8)
    # Process triangles in batches so the candidate list stays a manageable size.
    batch_limit = 20_000_000
    order = np.arange(len(tri))
    start = 0
    while start < len(order):
        csum = np.cumsum(counts[start:])
        end = start + max(1, int(np.searchsorted(csum, batch_limit)))
        idx = order[start:end]
        n = counts[idx]
        t = np.repeat(idx, n)
        if len(t):
            # Local offset of each candidate inside its triangle's bbox.
            first = np.repeat(np.cumsum(n) - n, n)
            local = np.arange(len(t)) - first
            i = lo[t, 0] + local % wx[t]
            j = lo[t, 1] + local // wx[t]
            px = o[0] + i * p
            py = o[1] + j * p

            # 2D barycentric coordinates of the column centre.
            ax, ay = a[t, 0], a[t, 1]
            v0x, v0y = b[t, 0] - ax, b[t, 1] - ay
            v1x, v1y = c[t, 0] - ax, c[t, 1] - ay
            v2x, v2y = px - ax, py - ay
            den = v0x * v1y - v1x * v0y
            ok = np.abs(den) > 1e-18
            den = np.where(ok, den, 1.0)
            u = (v2x * v1y - v1x * v2y) / den
            v = (v0x * v2y - v2x * v0y) / den
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)

            t, i, j, u, v = t[hit], i[hit], j[hit], u[hit], v[hit]
            z = a[t, 2] + u * (b[t, 2] - a[t, 2]) + v * (c[t, 2] - a[t, 2])
            # Voxel centres strictly above the crossing get toggled.
            k = np.ceil((z - grid.origin[2]) / p).astype(np.int64)
            k = np.clip(k, 0, nz)
            np.add.at(toggles, (i, j, k), 1)
        start = end

    inside = np.cumsum(toggles, axis=2, dtype=np.uint8)[:, :, :nz] & 1
    return inside.astype(bool)


def mesh_sdf(mesh, grid, smooth=0.7):
    """Approximate signed distance field (mm) of a watertight mesh.

    Built from Euclidean distance transforms of the voxelised solid, so it is
    accurate to roughly half a voxel at the surface and exact-ish in the bulk.
    A light Gaussian blur removes the voxel staircase.
    """
    inside = inside_mask(mesh, grid)
    if not inside.any():
        raise ValueError("No voxels inside the mesh. Is it closed and watertight?")
    p = grid.pitch
    d_out = ndimage.distance_transform_edt(~inside).astype(np.float32)
    d_in = ndimage.distance_transform_edt(inside).astype(np.float32)
    sdf = np.where(inside, -(d_in - 0.5), d_out - 0.5) * p
    del d_out, d_in, inside
    if smooth:
        sdf = ndimage.gaussian_filter(sdf, sigma=smooth)
    return sdf.astype(np.float32)
