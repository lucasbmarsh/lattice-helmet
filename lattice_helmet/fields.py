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


# Honeycombs are 2.5D: walls in the (u, v) plane, extruded along w. Under the
# conformal map w runs through the shell, so the walls stand normal to the skins
# like a sandwich-panel core.

_SQ3 = np.float32(np.sqrt(3))


def honeycomb_tri(u, v, w, cell):
    """Distance to the walls of a triangle grid with side `cell`."""
    h = np.float32(cell) * _SQ3 / 2  # spacing of each family of parallel lines
    best = None
    for nu, nv in ((0, 1), (_SQ3 / 2, -0.5), (_SQ3 / 2, 0.5)):
        t = nu * u + nv * v
        d = np.abs(t - h * np.round(t / h))
        best = d if best is None else np.minimum(best, d)
    return best


def honeycomb_hex(u, v, w, cell):
    """Distance to the walls of a hexagon grid, `cell` across the flats."""
    a = np.float32(cell) / 2  # apothem
    # Hexagon centres lie on two rectangular grids offset by half a cell; the
    # nearer of the two candidates is the centre of the hexagon we are in.
    px, py = np.float32(cell), np.float32(cell) * _SQ3
    best = None
    for ox, oy in ((0, 0), (px / 2, py / 2)):
        qu = u - ox - px * np.round((u - ox) / px)
        qv = v - oy - py * np.round((v - oy) / py)
        # hexagonal norm with flats facing +-u
        r = np.maximum(np.abs(qu), np.abs(qu) * 0.5 + np.abs(qv) * (_SQ3 / 2))
        best = r if best is None else np.minimum(best, r)
    return a - best


def _hash3(i, j, k, seed):
    """Three pseudo-random floats in [0, 1) per integer cell index."""
    h = (i.astype(np.int64) * 73856093) ^ (j.astype(np.int64) * 19349663) ^ (k.astype(np.int64) * 83492791)
    h = (h ^ (seed * 2654435761)) & 0xFFFFFFFF
    out = []
    for m in (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D):
        x = (h * m) & 0xFFFFFFFF
        x ^= x >> 15
        x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
        x ^= x >> 12
        out.append((x & 0xFFFFFF).astype(np.float32) / np.float32(1 << 24))
    return out


def _voronoi_setup(u, v, w, cell, seed, jitter, margin=1):
    """Flattened points, their cell indices, and a table of the seed point of
    every cell in range (one per cell, moved randomly by up to `jitter` of a cell)."""
    u, v, w = np.broadcast_arrays(u, v, w)
    pts = [np.ascontiguousarray(a, dtype=np.float32).reshape(-1) for a in (u, v, w)]
    c = np.float32(cell)
    cells = [np.floor(a / c).astype(np.int64) for a in pts]
    lo = [int(a.min()) - margin for a in cells]
    hi = [int(a.max()) + margin for a in cells]
    idx = np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij")
    r = _hash3(*idx, seed)
    table = np.stack([(n + 0.5 + jitter * (q - 0.5)) * c for n, q in zip(idx, r)], axis=-1).astype(np.float32)
    cells = [a - l for a, l in zip(cells, lo)]
    return u.shape, pts, cells, table


_OFFS = [(a, b, d) for a in (-1, 0, 1) for b in (-1, 0, 1) for d in (-1, 0, 1)]


def _blocks(pts, cells, table, block):
    """Per block of points: (slice, points, the 27 candidate seeds, squared distances)."""
    for s0 in range(0, len(pts[0]), block):
        sl = slice(s0, s0 + block)
        p = [a[sl] for a in pts]
        base = [a[sl] for a in cells]
        seeds = np.stack([table[base[0] + a, base[1] + b, base[2] + d] for a, b, d in _OFFS])  # (27, n, 3)
        d2 = sum((p[i] - seeds[:, :, i]) ** 2 for i in range(3))
        yield sl, p, seeds, d2


def voronoi_surface(u, v, w, cell, seed=0, jitter=1.0, block=1 << 18):
    """Closed-cell foam: distance to the nearest Voronoi face.

    The nearest seed is found among the 27 cells around the point; the
    distance to the bisector plane between it and each other seed is exact,
    and the nearest of those planes is the cell's face.
    """
    shape, pts, cells, table = _voronoi_setup(u, v, w, cell, seed, jitter)
    out = np.empty_like(pts[0])
    for sl, p, seeds, d2 in _blocks(pts, cells, table, block):
        near = np.argmin(d2, axis=0)
        s1 = np.take_along_axis(seeds, near[None, :, None], axis=0)[0]
        e = seeds - s1  # (27, n, 3)
        ln = np.sqrt((e * e).sum(axis=2))
        # distance from p to each bisector of s1 and another seed: (|p-s|^2 - |p-s1|^2) / 2|s-s1|
        f = (d2 - np.take_along_axis(d2, near[None], axis=0)) / np.maximum(2 * ln, 1e-6)
        f[near, np.arange(len(near))] = np.inf
        out[sl] = f.min(axis=0)
    return out.reshape(shape)


def voronoi_beam(u, v, w, cell, seed=0, jitter=1.0):
    """Open-cell foam: struts on the Voronoi edges, where three cells meet.

    The Voronoi diagram of the seeds in range (plus a margin) is built with
    scipy, its edges are binned into cells, and each point takes the exact
    distance to the edges listed for its cell. Edges farther than half a cell
    are never listed, so the result is min(distance, cell / 2), which is
    still continuous.
    """
    from scipy.spatial import Voronoi

    shape, pts, cells, table = _voronoi_setup(u, v, w, cell, seed, jitter, margin=3)
    c = np.float32(cell)
    reach = 0.3 * c  # struts up to 0.6 of a cell across
    vor = Voronoi(table.reshape(-1, 3).astype(np.float64))
    edges = set()
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        for i, j in zip(ridge, ridge[1:] + ridge[:1]):
            edges.add((min(i, j), max(i, j)))
    e = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
    va = vor.vertices.astype(np.float32)
    # Positions relative to the table origin (cell (0, 0, 0) of the table).
    origin = (np.array([pts[i].min() for i in range(3)], dtype=np.float32) // c - 3) * c
    size = np.array(table.shape[:3]) * c
    A, B = va[e[:, 0]] - origin, va[e[:, 1]] - origin
    # Keep edges in the trusted middle of the table, away from its margin.
    inner = np.all((np.minimum(A, B) > c) & (np.maximum(A, B) < size - c), axis=1)
    A, B = A[inner], B[inner]

    # Bin every edge into each half-cell bin its bounding box, grown by
    # `reach`, touches.
    bs = c / 2
    nb = tuple(int(n) for n in np.ceil(size / bs))
    lo = np.floor((np.minimum(A, B) - reach) / bs).astype(np.int64).clip(0, np.array(nb) - 1)
    hi = np.floor((np.maximum(A, B) + reach) / bs).astype(np.int64).clip(0, np.array(nb) - 1)
    span = hi - lo + 1
    owner = np.repeat(np.arange(len(A)), span.prod(axis=1))
    first = np.repeat(np.cumsum(span.prod(axis=1)) - span.prod(axis=1), span.prod(axis=1))
    r = np.arange(len(owner)) - first  # index within each edge's box of bins
    sp = span[owner]
    bi = lo[owner] + np.stack([r // (sp[:, 1] * sp[:, 2]), (r // sp[:, 2]) % sp[:, 1], r % sp[:, 2]], axis=1)
    flat = np.ravel_multi_index(bi.T, nb)
    order = np.argsort(flat, kind="stable")
    flat, owner = flat[order], owner[order]
    edge_start = np.searchsorted(flat, np.arange(int(np.prod(nb)) + 1))

    # Group the points by bin and measure each group against its own edges.
    rel = np.stack([pts[i] - origin[i] for i in range(3)], axis=1)
    bin_of = np.ravel_multi_index([np.clip((rel[:, i] // bs).astype(np.int64), 0, nb[i] - 1)
                                   for i in range(3)], nb)
    porder = np.argsort(bin_of, kind="stable")
    bins, pstart = np.unique(bin_of[porder], return_index=True)
    pstart = np.append(pstart, len(porder))
    out = np.full(len(porder), reach, dtype=np.float32)
    for n, bn in enumerate(bins):
        ks = owner[edge_start[bn]:edge_start[bn + 1]]
        if len(ks) == 0:
            continue
        idx = porder[pstart[n]:pstart[n + 1]]
        q = rel[idx][:, None, :] - A[ks][None]  # (m, k, 3)
        d = (B[ks] - A[ks])[None]
        t = np.clip((q * d).sum(axis=2) / np.maximum((d * d).sum(axis=2), 1e-12), 0, 1)
        d2 = ((q - t[..., None] * d) ** 2).sum(axis=2).min(axis=1)
        out[idx] = np.sqrt(np.minimum(d2, reach * reach))
    return out.reshape(shape)


def geodesic_points(freq, up=(2, 1)):
    """Unit vectors of a class-I geodesic sphere of frequency `freq`
    (10 freq^2 + 2 points): an icosahedron with each face cut into a
    triangular grid, pushed out onto the sphere. Its Voronoi cells are
    hexagons, plus 12 pentagons at the icosahedron's corners; one corner
    points along `up`, so the crown gets a pentagon and the next five sit
    63 degrees down."""
    import trimesh

    ico = trimesh.creation.icosahedron()
    v = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    # Rotate so vertex 0 points along `up`.
    target = np.zeros(3)
    target[up[0]] = up[1]
    R = trimesh.geometry.align_vectors(v[0], target)[:3, :3]
    v = v @ R.T
    pts = []
    n = max(1, int(freq))
    for a, b, c in ico.faces:
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                pts.append((i * v[a] + j * v[b] + k * v[c]) / n)
    pts = np.asarray(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return np.unique(np.round(pts, 9), axis=0)


def geodesic_hex(x, y, z, cell, center, up, ref_radius):
    """Distance (mm) to the walls of hexagonal cells laid over a sphere
    around `center` and run straight out from it.

    Every wall lies in a plane through the centre (the bisector of two
    neighbouring cell directions), so each hexagon sits flat on the dome
    and its hole runs through the shell along the radius, which on a helmet
    is close to the surface normal. `cell` is the hexagon size across the
    flats at `ref_radius` (the shell's typical radius); cells grow with
    radius, as on any radial pattern.
    """
    from scipy.spatial import cKDTree

    freq = max(1, int(round(1.1071 * ref_radius / cell)))  # icosahedron edge = 1.1071 rad
    seeds = geodesic_points(freq, up)
    tree = cKDTree(seeds)
    x, y, z = np.broadcast_arrays(x, y, z)
    shape = x.shape
    p = np.stack([(a - c).reshape(-1) for a, c in zip((x, y, z), center)], axis=1).astype(np.float32)
    out = np.empty(len(p), dtype=np.float32)
    block = 1 << 19
    for s0 in range(0, len(p), block):
        q = p[s0:s0 + block]
        d = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-6)
        _, idx = tree.query(d, k=7)  # nearest seed and its (5 or) 6 neighbours
        s1 = seeds[idx[:, 0]]
        best = np.full(len(q), np.inf, dtype=np.float32)
        for k in range(1, 7):
            e = s1 - seeds[idx[:, k]]
            n = e / np.linalg.norm(e, axis=1, keepdims=True)
            best = np.minimum(best, (q * n).sum(axis=1))  # distance to the plane through the centre
        out[s0:s0 + block] = best
    return out.reshape(shape)


# Lattices laid out around the helmet centre in world coordinates, so they
# ignore the cell map: name -> (function, mode).
RADIAL = {"geodesic_hex": (geodesic_hex, "sheet")}


# Lattices that are not TPMS or strut graphs: name -> (function, mode).
# Their fields are already close to true distances, so no calibration.
OTHER = {
    "honeycomb_tri": (honeycomb_tri, "sheet"),
    "honeycomb_hex": (honeycomb_hex, "sheet"),
    "voronoi": (voronoi_beam, "beam"),
    "voronoi_surface": (voronoi_surface, "sheet"),
}
RANDOM = {"voronoi", "voronoi_surface"}


def lattice_kinds():
    sheets = sorted(TPMS)
    return (sheets + [f"{s}_network" for s in sheets] + sorted(GRAPHS) + sorted(OTHER) + sorted(RADIAL)
            + ["custom"])


def raw_field(kind, u, v, w, cell, graph=None, seed=0, jitter=1.0):
    """Return (R, (mode, cal)) for a lattice kind.

    mode says how R becomes a distance; cal scales the requested wall.
    seed and jitter only affect the random (Voronoi) lattices.
    """
    if kind in OTHER:
        fn, mode = OTHER[kind]
        kw = dict(seed=seed, jitter=jitter) if kind in RANDOM else {}
        return fn(u, v, w, cell, **kw), (mode, 1.0)
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
