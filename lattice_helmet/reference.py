"""Lay cells out over a reference body (e.g. the headform the helmet was
built on) instead of over a sphere.

A sphere only fits a head roughly, so cells laid over one stretch where the
head is longer or flatter than the sphere, and holes that run along its
radius go through the shell at a slant. With a reference body:

geodesic_hex  the hexagon centres are spread evenly over the reference
              surface itself (a geodesic sphere pushed onto it, then relaxed
              so every cell has the same area), and every point of the shell
              takes the cell of its nearest point on the reference. So each
              hexagon has the set size on the head, and its hole runs along
              the head's normal, straight out through the shell.
conformal     distance from the crown is measured along the reference
              surface (its meridians) rather than as an angle on a sphere,
              so rows of cells keep their size down the front, back and
              sides of a long or flat head.

The reference can be a closed body or an open surface; it only needs to
cover the part of the head the helmet sits on, in the same coordinates as
the shell.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree


class Reference:
    def __init__(self, mesh, spacing=1.5, max_points=600_000, cap=True):
        """mesh: the reference body or surface, in mm, in the shell's coordinates.
        cap: close openings in an open surface (see `cap_openings`), so a
        headform cut off flat at the back still has a back for cells to sit on."""
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
        if not len(mesh.faces):
            raise ValueError("the reference body has no triangles")
        # Sample the surface densely enough that its nearest sample's tangent
        # plane is a good stand-in for the surface itself.
        spacing = max(spacing, float(np.sqrt(mesh.area / max_points)))
        v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=spacing)
        fine = trimesh.Trimesh(v, f, process=True)
        self.mesh = mesh
        self.points = np.asarray(fine.vertices, dtype=np.float64)
        n = np.asarray(fine.vertex_normals, dtype=np.float64)
        self.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        # Area each sample stands for (a third of each face it touches).
        w = np.zeros(len(self.points))
        np.add.at(w, fine.faces.ravel(), np.repeat(fine.area_faces / 3, 3))
        self.weights = w
        self.capped = 0
        if cap:
            caps = cap_openings(mesh, spacing)
            for p, n, a in caps:
                self.points = np.vstack([self.points, p])
                self.normals = np.vstack([self.normals, n])
                self.weights = np.concatenate([self.weights, a])
            self.capped = len(caps)
        self.area = float(self.weights.sum())
        # The middle of the head: the centre of the sphere that best fits the
        # surface. Cells are laid out from here, not from the part's own
        # centre, so every panel of a split helmet gets the same cells.
        w = np.sqrt(self.weights / self.area)
        A = np.hstack([2 * self.points, np.ones((len(self.points), 1))]) * w[:, None]
        b = (self.points ** 2).sum(axis=1) * w
        c = np.linalg.lstsq(A, b, rcond=None)[0][:3]
        self.center = np.clip(c, self.points.min(axis=0), self.points.max(axis=0))
        self.tree = cKDTree(self.points)
        self._seeds = {}
        self._meridians = {}
        self._flows = {}

    @classmethod
    def load(cls, data, filetype="stl"):
        import io

        mesh = trimesh.load(io.BytesIO(data), file_type=filetype, force="mesh")
        return cls(mesh)

    # -- nearest point on the reference -------------------------------------

    def foot(self, p):
        """Nearest point on the reference surface to each row of p (n, 3):
        the nearest sample, slid along its tangent plane under p."""
        _, i = self.tree.query(p, workers=-1)
        q, n = self.points[i], self.normals[i]
        return p - n * ((p - q) * n).sum(axis=1, keepdims=True)

    def landing(self, p, bounds, smooth=10.0, steps=14):
        """Where each point of p (n, 3) lands on the reference when it moves
        down a smoothed distance field: close to the surface that is its
        nearest point, so walls run along the surface normal; further out
        the paths bend gently instead of all crowding into a hollow (like the
        nape of the neck) the way nearest points do. `bounds` (2, 3) is the
        region the points come from; `smooth` the blur, mm."""
        dist, grad, origin, h = self._flow(bounds, smooth)
        from scipy import ndimage

        p = np.array(p, dtype=np.float64)
        start = p.copy()
        moving = np.arange(len(p))
        for _ in range(steps):
            q = p[moving]
            idx = [(q[:, k] - origin[k]) / h for k in range(3)]
            d = ndimage.map_coordinates(dist, idx, order=1, mode="nearest")
            # Within a few voxels the field is too coarse to steer by, and
            # the nearest point is already right, so stop there.
            far = d > 3 * h
            moving, q, d = moving[far], q[far], d[far]
            if not len(moving):
                break
            idx = [a[far] for a in idx]
            g = np.stack([ndimage.map_coordinates(gk, idx, order=1, mode="nearest") for gk in grad], axis=1)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            p[moving] = q - g * (0.6 * d)[:, None]
        else:
            # Paths still far out after all the steps stalled where the
            # smoothed field is flat; those points take their nearest point.
            if len(moving):
                p[moving] = start[moving]
        return self.foot(p)

    def _flow(self, bounds, smooth):
        from scipy import ndimage

        lo, hi = np.asarray(bounds, dtype=np.float64)
        key = (tuple(np.round(lo)), tuple(np.round(hi)), smooth)
        if key in self._flows:
            return self._flows[key]
        lo, hi = lo - 2 * smooth, hi + 2 * smooth
        h = float(np.clip(np.cbrt(np.prod(hi - lo) / 4e6), 1.0, 3.0))
        shape = tuple(int(n) for n in np.ceil((hi - lo) / h) + 1)
        idx = np.round((self.points - lo) / h).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < shape), axis=1)
        if not ok.any():
            raise ValueError("the reference body is nowhere near the part; are they in the same coordinates?")
        free = np.ones(shape, dtype=bool)
        free[tuple(idx[ok].T)] = False
        dist = (ndimage.distance_transform_edt(free) * h).astype(np.float32)
        blurred = ndimage.gaussian_filter(dist, sigma=smooth / h)
        grad = [g.astype(np.float32) for g in np.gradient(blurred, h)]
        out = dist, grad, lo, h
        self._flows = {key: out}  # one region at a time; each is tens of MB
        return out

    def gap(self, shell, n=20000):
        """Typical distance (mm) from the reference out to the middle of the
        shell, so cell sizes can be set where the part is, not on the head."""
        key = ("gap", id(shell))
        if key not in self._seeds:
            p = trimesh.sample.sample_surface(shell, n, seed=0)[0]
            d, _ = self.tree.query(p, workers=-1)
            self._seeds[key] = float(np.median(d))
        return self._seeds[key]

    # -- hexagonal cells --------------------------------------------------------

    def hex_seeds(self, cell, center, up, iterations=20):
        """Cell centres spread evenly over the surface, about `cell` mm apart
        (hexagons `cell` across the flats). Starts from a geodesic sphere
        pushed out onto the surface from `center` and thinned to the number
        of cells the surface's area holds, keeping one on the crown, then
        evens them out by Lloyd relaxation on the surface."""
        from .fields import geodesic_points

        key = (round(float(cell), 3), tuple(np.round(center, 2)), tuple(up))
        if key in self._seeds:
            return self._seeds[key]
        rel = self.points - center
        dirs = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-9)
        dir_tree = cKDTree(dirs)

        def seeds_for(n):
            """About n points over the surface, from the finest geodesic
            sphere needed, thinned evenly from the crown down."""
            freq = max(1, int(np.sqrt(max(n - 2, 1) / 10)))
            while True:
                g = geodesic_points(freq, up)
                dist, i = dir_tree.query(g)
                # Directions the surface doesn't reach (e.g. below an open
                # neck) pick up a far-off sample; leave those out.
                _, first = np.unique(i, return_index=True)
                first = np.sort(first[dist[first] < 0.5 / freq])
                if len(first) >= n or freq > 200:
                    break
                freq += 1
            pts = self.points[i[first]]
            return pts[_farthest_first(pts, min(n, len(pts)), start=int(np.argmax((pts - center) @ axis)))]

        axis = np.zeros(3)
        axis[up[0]] = up[1]
        # A hexagon `cell` across the flats covers sqrt(3)/2 cell^2.
        n = max(4, int(round(self.area / (np.sqrt(3) / 2 * cell * cell))))
        seeds = self._relax(seeds_for(n), iterations)
        # Cells don't tile a curved, possibly open surface exactly; check the
        # spacing they came out at and correct the count once.
        spacing = self._spacing(seeds)
        if abs(spacing / cell - 1) > 0.02:
            seeds = self._relax(seeds_for(max(4, int(round(n * (spacing / cell) ** 2)))), iterations)
        seeds = self._even(seeds)
        self._seeds[key] = seeds
        return seeds

    def _even(self, seeds, iterations=200, rate=0.3):
        """Push apart neighbours closer than the typical spacing. Lloyd leaves
        a few tight pairs, and the tightest pair sets how big same-size holes
        can be."""
        for _ in range(iterations):
            d, nb = cKDTree(seeds).query(seeds, k=min(7, len(seeds)))
            target = np.median(d[:, 1:])
            v = seeds[:, None] - seeds[nb[:, 1:]]
            dd = d[:, 1:, None]
            push = np.clip(target - dd, 0, None) * v / np.maximum(dd, 1e-9)
            seeds = self.foot(seeds + rate * push.sum(axis=1))
        return seeds

    def _relax(self, seeds, iterations):
        for _ in range(iterations):
            _, owner = cKDTree(seeds).query(self.points)
            mass = np.bincount(owner, self.weights, minlength=len(seeds))
            keep = mass > 0
            c = np.stack([np.bincount(owner, self.weights * self.points[:, k], minlength=len(seeds))
                          for k in range(3)], axis=1)
            seeds = self.foot(c[keep] / mass[keep, None])
        return seeds

    @staticmethod
    def _spacing(seeds):
        """Typical distance between neighbouring cell centres."""
        d, _ = cKDTree(seeds).query(seeds, k=4)
        return float(np.median(d[:, 1:]))

    def hex_holes(self, cell, center, up, wall, same_size=False):
        """fields.HexHoles for the cells: a regular hexagonal hole per cell,
        keeping at least `wall` mm between holes."""
        from .fields import HexHoles

        key = ("holes", round(float(cell), 3), tuple(np.round(center, 2)), tuple(up), round(float(wall), 3),
               same_size)
        if key not in self._seeds:
            seeds = self.hex_seeds(cell, center, up)
            _, i = self.tree.query(seeds)
            self._seeds[key] = HexHoles(seeds, self.normals[i], wall, same_size=same_size)
        return self._seeds[key]

    def hex_field(self, x, y, z, cell, center, up, bounds, mask=None, holes_wall=None, same_size=False,
                  block=1 << 19):
        """Distance (mm, on the reference surface) from where each point
        lands on the reference (see `landing`) to the walls between the
        hexagonal cells. Points outside `mask` get `cell` (far from any
        wall); `bounds` is the region all the points come from.

        With `holes_wall`, the distance to the nearest regular hexagonal
        hole instead (see fields.HexHoles), positive between holes."""
        seeds = self.hex_seeds(cell, center, up)
        tree = cKDTree(seeds)
        holes = None if holes_wall is None else self.hex_holes(cell, center, up, holes_wall, same_size)
        x, y, z = np.broadcast_arrays(x, y, z)
        shape = x.shape
        out = np.full(int(np.prod(shape)), np.float32(cell), dtype=np.float32)
        sel = np.arange(out.size) if mask is None else np.flatnonzero(np.broadcast_to(mask, shape))
        if not len(sel):
            return out.reshape(shape)
        xs, ys, zs = (a.reshape(-1) for a in (x, y, z))
        k = min(7, len(seeds))
        for s0 in range(0, len(sel), block):
            idx = sel[s0:s0 + block]
            p = np.stack([xs[idx], ys[idx], zs[idx]], axis=1).astype(np.float64)
            f = self.landing(p, bounds)
            if holes is not None:
                out[idx] = holes.field(f, p)
                continue
            d, nb = tree.query(f, k=k)
            s1 = seeds[nb[:, 0]]
            best = np.full(len(f), np.inf)
            for j in range(1, k):
                sj = seeds[nb[:, j]]
                e = sj - s1
                # distance from f to the plane halfway between the two centres
                best = np.minimum(best, (d[:, j] ** 2 - d[:, 0] ** 2) / (2 * np.maximum(np.linalg.norm(e, axis=1), 1e-9)))
            out[idx] = best
        return out.reshape(shape)

    # -- distance from the crown along the surface ---------------------------

    def meridians(self, center, up, offset=0.0, n_az=96, n_pol=120):
        """Arc length (mm) from the crown down the surface, raised `offset`
        mm off it, as a table over the azimuth around `up` and the angle from
        it. Each meridian's profile r(angle) is the mean radius of the
        surface samples in that bin, filled and smoothed, then integrated."""
        from scipy import ndimage

        key = (tuple(np.round(center, 2)), tuple(up), round(float(offset), 2))
        if key in self._meridians:
            return self._meridians[key]
        a, b, h = _frame(self.points, center, up)
        r = np.sqrt(a * a + b * b + h * h)
        theta = np.arctan2(b, a)
        alpha = np.arccos(np.clip(h / np.maximum(r, 1e-9), -1, 1))
        ti = np.clip(((theta + np.pi) / (2 * np.pi) * n_az).astype(int), 0, n_az - 1)
        ai = np.clip((alpha / np.pi * n_pol).astype(int), 0, n_pol - 1)
        # median radius per bin (weighted mean is close enough and vectorises)
        flat = ti * n_pol + ai
        wsum = np.bincount(flat, self.weights, minlength=n_az * n_pol)
        rsum = np.bincount(flat, self.weights * r, minlength=n_az * n_pol)
        prof = np.where(wsum > 0, rsum / np.maximum(wsum, 1e-12), np.nan).reshape(n_az, n_pol)
        have = ~np.isnan(prof)
        if not have.any():
            raise ValueError("the reference body is empty around the centre")
        # Fill empty bins from the nearest filled one (wraps nothing; fine for a head).
        _, (ii, jj) = ndimage.distance_transform_edt(~have, return_indices=True)
        prof = prof[ii, jj]
        # Smooth around the azimuth (periodic) and down each meridian.
        prof = ndimage.gaussian_filter(prof, sigma=(1.5, 1.0), mode=("wrap", "nearest")) + offset
        da = np.pi / n_pol
        dr = np.gradient(prof, da, axis=1)
        ds = np.sqrt(prof ** 2 + dr ** 2) * da
        s = np.concatenate([np.zeros((n_az, 1)), np.cumsum(ds, axis=1)], axis=1)  # at bin edges
        alpha_edges = np.linspace(0, np.pi, n_pol + 1)
        table = (s.astype(np.float32), alpha_edges.astype(np.float32), n_az)
        self._meridians[key] = table
        return table

    def arc_from_crown(self, theta, alpha, center, up, offset=0.0):
        """Distance down the reference surface (raised `offset` mm) from the
        crown, mm, for points at azimuth `theta` and angle `alpha` from the up
        axis (radians)."""
        from scipy import ndimage

        s, edges, n_az = self.meridians(center, up, offset)
        n_pol = len(edges) - 1
        ti = (np.asarray(theta) + np.pi) / (2 * np.pi) * n_az - 0.5
        ai = np.asarray(alpha) / np.pi * n_pol
        ti, ai = np.broadcast_arrays(ti, ai)
        # periodic in azimuth: pad one column each side
        pad = np.concatenate([s[-1:], s, s[:1]], axis=0)
        out = ndimage.map_coordinates(pad, [ti.ravel() + 1, ai.ravel()], order=1, mode="nearest")
        return out.reshape(ti.shape).astype(np.float32)


def _farthest_first(pts, n, start=0):
    """Indices of n of the points, each as far as possible from those before."""
    pick = [start]
    d = np.linalg.norm(pts - pts[start], axis=1)
    for _ in range(n - 1):
        k = int(np.argmax(d))
        pick.append(k)
        d = np.minimum(d, np.linalg.norm(pts - pts[k], axis=1))
    return np.array(pick)


def cap_openings(mesh, spacing, min_size=20.0, bulge=0.5):
    """Fill each opening in an open surface with a rounded cap, as samples:
    a list of (points, normals, areas), one per opening at least `min_size`
    mm across.

    The cap is a dome over the opening's best-fit plane, bulging away from the
    rest of the surface, that starts square to the plane at the edge (where
    a head cut flat at the back meets its cut) and rises to `bulge` times
    the opening's inner radius in the middle, which is about how far the back
    of a head rounds out. The cut is rarely the part the helmet is about, so
    a plausible, smooth stand-in beats cells fanning out from the edge.
    """
    from scipy import ndimage
    from skimage.draw import polygon as fill_polygon

    try:
        loops = [np.asarray(d) for d in mesh.outline().discrete]
    except (IndexError, ValueError):
        return []
    caps = []
    middle = mesh.vertices.mean(axis=0)
    for loop in loops:
        if len(loop) < 8:
            continue
        c = loop.mean(axis=0)
        _, sv, vt = np.linalg.svd(loop - c, full_matrices=False)
        e1, e2, nrm = vt
        if sv[1] / np.sqrt(len(loop)) < min_size / 4:
            continue
        if np.dot(middle - c, nrm) > 0:
            nrm = -nrm  # bulge away from the rest of the surface
        e2 = np.cross(nrm, e1)
        uv = np.stack([(loop - c) @ e1, (loop - c) @ e2], axis=1)
        h = spacing
        lo = uv.min(axis=0) - 2 * h
        shape = tuple(int(n) for n in np.ceil((uv.max(axis=0) + 2 * h - lo) / h) + 1)
        inside = np.zeros(shape, dtype=bool)
        rr, cc = fill_polygon((uv[:, 0] - lo[0]) / h, (uv[:, 1] - lo[1]) / h, shape)
        inside[rr, cc] = True
        if inside.sum() < 4:
            continue
        d = ndimage.distance_transform_edt(inside) * h
        dmax = float(d.max())
        if 2 * dmax < min_size:
            continue
        t = np.clip(d / dmax, 0, 1)
        height = ndimage.gaussian_filter(bulge * dmax * np.sqrt(2 * t - t * t), sigma=max(1.0, 0.05 * dmax / h))
        height[~inside] = 0
        gu, gv = np.gradient(height, h)
        iu, iv = np.nonzero(inside)
        u, v = lo[0] + iu * h, lo[1] + iv * h
        pts = c + np.outer(u, e1) + np.outer(v, e2) + np.outer(height[iu, iv], nrm)
        n = nrm[None] - np.outer(gu[iu, iv], e1) - np.outer(gv[iu, iv], e2)
        slope = np.linalg.norm(n, axis=1)
        caps.append((pts, n / slope[:, None], h * h * slope))
    return caps


def _frame(p, center, up):
    """(a, b, h) of points p relative to center, with h along up (axis, sign)."""
    i, sign = up
    q = p - np.asarray(center)
    a, b = [q[:, j] for j in range(3) if j != i]
    return a, b * sign, q[:, i] * sign
