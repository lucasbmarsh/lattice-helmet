"""Fill a closed shell mesh with a lattice and mesh the result."""

from dataclasses import asdict, dataclass, field, fields as dc_fields, replace
import json
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from . import expr, fields, maps
from .sdf import make_grid, mesh_sdf
from .surfaces import SURFACES, default_center, surface_distances


@dataclass
class LatticeSpec:
    """Everything that defines a lattice part. Saved and loaded as a JSON recipe."""

    lattice: str = "gyroid"  # see fields.lattice_kinds()
    cell: float = 10.0  # unit cell size, mm
    graph: list = None  # strut segments in unit-cell coords, for lattice="custom"
    cell_map: str = "rectangular"  # rectangular | cylindrical | spherical | conformal
    center: list = None  # helmet centre for curved maps and surfaces (default: rim centre)
    pole: str = None  # spherical: pole axis (default x, poles at the ears); conformal: crown axis (default z)
    layers: int = None  # cells through the shell for the conformal map (default: auto)
    wall_min: float = 0.8  # wall / strut thickness where grading = 0, mm
    wall_max: float = 0.8  # wall / strut thickness where grading = 1, mm
    grading: str = "uniform"  # uniform | z | depth | point | expr
    grade_point: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # used by grading="point"
    grade_depth: float = 10.0  # used by grading="depth", mm
    grade_expr: str = None  # used by grading="expr"; result is clipped to [0, 1]
    skin: float = 1.2  # solid skin thickness, mm (0 = none)
    skin_surfaces: list = field(default_factory=lambda: ["all"])  # "all" or any of outer, inner, rim
    holes: list = field(default_factory=list)  # drain holes through the skin, each [x, y, z, radius] in mm
    fillet: float = 0.5  # blend radius between lattice and skin, mm
    pitch: float = 0.5  # voxel size, mm
    slab: int = 64  # z-slices meshed per chunk
    decimate: float = 0.0  # fraction of triangles to remove (0 = keep all)
    log: callable = field(default=print, repr=False, compare=False)

    def to_json(self):
        d = asdict(self)
        d.pop("log")
        return json.dumps(d, indent=2)

    @classmethod
    def from_dict(cls, d, **extra):
        known = {f.name for f in dc_fields(cls)} - {"log"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown recipe keys: {sorted(unknown)}")
        return cls(**d, **extra)


def _grading(spec, x, y, z, s, aux, bounds):
    """Scalar field in [0, 1] that drives wall thickness."""
    lo, hi = bounds
    if spec.grading == "uniform":
        return np.float32(0)
    if spec.grading == "z":
        return np.clip((z - lo[2]) / max(hi[2] - lo[2], 1e-6), 0, 1)
    if spec.grading == "depth":
        return np.clip(-s / spec.grade_depth, 0, 1)
    if spec.grading == "point":
        p = np.asarray(spec.grade_point, dtype=np.float32)
        d = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2)
        far = np.max(np.linalg.norm(np.array([lo, hi]) - p, axis=1))
        return np.clip(d / max(far, 1e-6), 0, 1)
    if spec.grading == "expr":
        c = aux["center"]
        env = {
            "x": x, "y": y, "z": z, "depth": -s,
            "r": np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2),
        }
        if aux.get("shell_t") is not None:
            env["shell"] = aux["shell_t"]
        return np.clip(np.asarray(expr.evaluate(aux["expr"], env), dtype=np.float32), 0, 1)
    raise ValueError(f"unknown grading {spec.grading!r}")


def _validate(spec):
    if spec.lattice not in fields.lattice_kinds():
        raise ValueError(f"lattice must be one of {fields.lattice_kinds()}")
    if spec.cell_map not in maps.MAPS:
        raise ValueError(f"cell_map must be one of {maps.MAPS}")
    surf = set(spec.skin_surfaces)
    if surf != {"all"} and not surf <= set(SURFACES):
        raise ValueError(f"skin_surfaces must be 'all' or a subset of {SURFACES}")
    if spec.grading == "expr":
        if not spec.grade_expr:
            raise ValueError("grading 'expr' needs grade_expr")
        return expr.compile_expr(spec.grade_expr)
    return None


def _prepare(shell, spec, cache=None):
    """Grid, shell SDF and surface distances: the part of a build that only
    depends on the shell, the voxel size and the centre. `cache` (a dict) lets
    repeated previews of one shell skip it."""
    margin = 3 * spec.pitch + spec.fillet
    key = (spec.pitch, margin)
    if cache is not None and key in cache:
        grid, sdf = cache[key]
    else:
        grid = make_grid(shell, spec.pitch, margin=margin)
        spec.log(f"grid {grid.shape} at {spec.pitch} mm ({int(np.prod(grid.shape)) / 1e6:.1f} M voxels)")
        sdf = mesh_sdf(shell, grid)
        if cache is not None:
            cache[key] = grid, sdf
    center = np.asarray(spec.center if spec.center is not None else default_center(shell), dtype=np.float32)

    def surfaces():
        k = key + (tuple(center.tolist()),)
        if cache is not None and k in cache:
            return cache[k]
        surf = surface_distances(shell, grid, center)
        if cache is not None:
            cache[k] = surf
        return surf

    return grid, sdf, center, surfaces


def _chunks(shell, spec, compiled, cache=None):
    """Mesh the part slab by slab; yields (vertices, faces) per chunk."""
    t0 = time.time()
    grid, sdf, center, surfaces = _prepare(shell, spec, cache)
    spec.log(f"shell SDF done in {time.time() - t0:.1f}s")

    aux = {"center": center, "expr": compiled, "shell_t": None}
    surf = None
    need_shell_t = spec.cell_map == "conformal" or (compiled is not None and "shell" in expr.names_used(compiled))
    if need_shell_t or set(spec.skin_surfaces) != {"all"}:
        surf = surfaces()
        spec.log("surfaces: " + ", ".join(f"{k}={'yes' if v is not None else 'none'}" for k, v in surf.items()))
    layers = spec.layers
    if need_shell_t:
        d_o, d_i = surf["outer"], surf["inner"]
        if d_o is None or d_i is None:
            raise ValueError("conformal map needs both outer and inner surfaces; check `center`")
        aux["shell_t"] = np.clip(d_o / np.maximum(d_o + d_i, 1e-6), 0, 1)
        if layers is None:
            thick = (d_o + d_i)[sdf < 0]
            layers = max(1, int(round(float(np.median(thick)) / spec.cell)))
        spec.log(f"conformal: {layers} cell layer(s) through the shell")
    ref_radius = float(np.median(np.linalg.norm(shell.vertices - center, axis=1)))

    bounds = shell.bounds.astype(np.float32)
    nz = grid.shape[2]
    seam = None
    count = 0
    for k0 in range(0, nz - 1, spec.slab):
        k1 = min(k0 + spec.slab + 1, nz)  # one slice of overlap so chunks stitch
        s = sdf[:, :, k0:k1]
        if s.min() > spec.fillet:  # slab entirely outside the part
            continue

        # Lattice on the slab plus a one-slice halo, so the gradient is centred.
        h0, h1 = max(k0 - 1, 0), min(k1 + 1, nz)
        hx, hy, hz = grid.coords(h0, h1)
        shell_t = None if aux["shell_t"] is None else aux["shell_t"][:, :, h0:h1]
        u, v, w = maps.map_coords(spec.cell_map, hx, hy, hz, spec.cell, center, ref_radius,
                                  pole=spec.pole, shell_t=shell_t, layers=layers or 1)
        R, mode = fields.raw_field(spec.lattice, u, v, w, spec.cell, spec.graph)
        del u, v, w
        gx, gy, gz = np.gradient(R, spec.pitch)
        crop = slice(k0 - h0, k0 - h0 + (k1 - k0))
        grad = np.sqrt(gx * gx + gy * gy + gz * gz)[:, :, crop]
        R = R[:, :, crop]
        del gx, gy, gz

        x, y, z = grid.coords(k0, k1)
        slab_aux = dict(aux, shell_t=None if aux["shell_t"] is None else aux["shell_t"][:, :, k0:k1])
        t = _grading(spec, x, y, z, s, slab_aux, bounds)
        wall = spec.wall_min + (spec.wall_max - spec.wall_min) * t
        lat = fields.to_distance(R, grad, mode, wall, spec.cell)
        del R, grad

        if spec.skin > 0:
            if set(spec.skin_surfaces) == {"all"}:
                skin = -s - spec.skin
            else:
                near = [surf[n][:, :, k0:k1] for n in spec.skin_surfaces if surf[n] is not None]
                skin = np.minimum.reduce(near) - spec.skin if near else np.full_like(s, np.inf)
            for hx_, hy_, hz_, hr in spec.holes:
                skin = np.maximum(skin, hr - np.sqrt((x - hx_) ** 2 + (y - hy_) ** 2 + (z - hz_) ** 2))
            body = fields.smooth_min(skin, lat, spec.fillet)
        else:
            body = lat
        f = np.maximum(s, body)  # keep everything inside the shell
        # Reuse the previous chunk's last slice so the shared plane is
        # bit-identical; recomputing it can differ in the last ulp and tear
        # the seam.
        if seam is not None and seam[0] == k0:
            f[:, :, 0] = seam[1]
        # Push near-zero samples off the surface. A sample sitting on the
        # iso-level makes all its edge vertices coincide, and merging them
        # pinches the surface into a non-manifold edge.
        f[np.abs(f) < 1e-4] = np.float32(1e-4)
        seam = (k1 - 1, f[:, :, -1].copy())
        if f.min() >= 0 or f.max() <= 0:
            continue
        vtx, faces, _, _ = marching_cubes(f, level=0.0, spacing=(spec.pitch,) * 3)
        vtx += grid.origin + np.array([0, 0, k0 * spec.pitch])
        count += 1
        yield vtx, faces
    spec.log(f"meshed {count} chunks in {time.time() - t0:.1f}s")


def _concat(pieces):
    verts, faces, offset = [], [], 0
    for v, fc in pieces:
        verts.append(v)
        faces.append(fc + offset)
        offset += len(v)
    return np.vstack(verts), np.vstack(faces)


def build(shell: trimesh.Trimesh, spec: LatticeSpec) -> trimesh.Trimesh:
    compiled = _validate(spec)
    if not shell.is_watertight:
        spec.log("warning: input mesh is not watertight; inside test may leak")

    t0 = time.time()
    pieces = list(_chunks(shell, spec, compiled))
    if not pieces:
        raise ValueError("the lattice came out empty; check cell size and wall thickness")

    out = trimesh.Trimesh(*_concat(pieces), process=False)
    out.merge_vertices(digits_vertex=6)
    # Drop only faces that collapsed onto a repeated vertex. Removing every
    # zero-area face would also drop slivers whose edges other faces rely on.
    f = out.faces
    out.update_faces((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0]))
    out.remove_unreferenced_vertices()
    if spec.decimate > 0:
        import fast_simplification

        v, fc = fast_simplification.simplify(
            out.vertices.astype(np.float32), out.faces, target_reduction=spec.decimate
        )
        # Collapsing thin walls can leave back-to-back face pairs (zero-volume
        # slivers); dropping both copies restores a manifold surface.
        _, inv, cnt = np.unique(np.sort(fc, axis=1), axis=0, return_inverse=True, return_counts=True)
        out = trimesh.Trimesh(v, fc[cnt[inv.ravel()] == 1], process=True)
        out.remove_unreferenced_vertices()
    out = split_nonmanifold(out)
    out = drop_specks(out)
    if out.volume < 0:
        out.invert()
    spec.log(f"done in {time.time() - t0:.1f}s: {len(out.faces):,} triangles")
    return out


def preview(shell, spec, voxels=1.5e6, max_faces=400_000, cache=None):
    """A quick, coarse version of the part for the panel's live view.

    The voxel size is picked so the grid has about `voxels` samples, and the
    cleanup passes that make the real build printable are skipped. A wall
    thinner than a voxel would come out full of holes, so walls and skin are
    drawn as sqrt(t^2 + voxel^2) thick (struts sqrt(t^2 + (2 voxels)^2), as a
    thin rod breaks up sooner than a sheet): never too thin to show, and still
    thicker when the setting is thicker. Pass the same `cache` dict for every
    preview of one shell to reuse its SDF. Returns (mesh, info).
    """
    compiled = _validate(spec)
    t0 = time.time()
    size = np.ptp(shell.bounds, axis=0) + 2 * spec.fillet
    pitch = float(np.cbrt(np.prod(size) / voxels))
    pitch = max(spec.pitch, np.ceil(pitch * 4) / 4)  # quarter-mm steps, so the cache hits

    one = np.zeros((1, 1, 1), dtype=np.float32)
    _, (mode, _) = fields.raw_field(spec.lattice, one, one, one, spec.cell, spec.graph)
    floor = 2 * pitch if mode == "beam" else pitch

    def drawn(t, floor=floor):
        return float(np.hypot(t, floor))

    quick = replace(spec, pitch=pitch, slab=128, decimate=0,
                    wall_min=drawn(spec.wall_min), wall_max=drawn(spec.wall_max),
                    skin=drawn(spec.skin, pitch) if spec.skin > 0 else 0)
    pieces = list(_chunks(shell, quick, compiled, cache))
    if not pieces:
        raise ValueError("the lattice came out empty; check cell size and wall thickness")
    v, f = _concat(pieces)
    if len(f) > max_faces:
        try:
            import fast_simplification

            v, f = fast_simplification.simplify(v.astype(np.float32), f, agg=10,
                                                target_reduction=1 - max_faces / len(f))
        except ImportError:
            pass
    info = {"pitch": pitch, "triangles": int(len(f)), "seconds": round(time.time() - t0, 2)}
    return trimesh.Trimesh(v, f, process=False), info


def _components(mesh):
    e = mesh.edges_unique
    n = len(mesh.vertices)
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(adj, directed=False)


def _edge_counts(faces, n):
    f = faces.astype(np.int64)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    keys, counts = np.unique(e[:, 0] * n + e[:, 1], return_counts=True)
    return keys, counts


def split_nonmanifold(mesh):
    """Separate surface sheets that marching cubes joined along one edge.

    In rare saddle cases two sheets share an edge (four faces on it). Each
    vertex of such an edge gets one copy per fan of faces around it, so the
    sheets touch but no longer share topology.
    """
    n = len(mesh.vertices)
    keys, counts = _edge_counts(mesh.faces, n)
    bad = keys[counts > 2]
    if len(bad) == 0:
        return mesh
    manifold = set(keys[counts == 2].tolist())
    faces = mesh.faces.copy()
    verts = [mesh.vertices]
    extra = n
    for v in np.unique(np.concatenate([bad // n, bad % n])):
        inc = np.nonzero((faces == v).any(axis=1))[0]
        # Union faces around v that share a manifold edge through v.
        parent = {int(f): int(f) for f in inc}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        by_edge = {}
        for f in inc:
            for w in faces[f]:
                if w == v:
                    continue
                key = min(v, w) * n + max(v, w)
                if key in manifold:
                    by_edge.setdefault(key, []).append(int(f))
        for fs in by_edge.values():
            for f in fs[1:]:
                parent[find(f)] = find(fs[0])
        groups = {}
        for f in inc:
            groups.setdefault(find(int(f)), []).append(int(f))
        for fs in list(groups.values())[1:]:
            verts.append(mesh.vertices[v][None])
            for f in fs:
                faces[f][faces[f] == v] = extra
            extra += 1
    return trimesh.Trimesh(np.vstack(verts), faces, process=False)


def drop_specks(mesh, min_faces=500):
    """Remove tiny disconnected shells (slivers where a wall grazes the skin)."""
    _, lab = _components(mesh)
    face_lab = lab[mesh.faces[:, 0]]
    sizes = np.bincount(face_lab)
    keep = sizes[face_lab] >= min_faces
    if keep.all():
        return mesh
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def report(shell, part):
    return {
        "triangles": len(part.faces),
        "watertight": bool(part.is_watertight),
        # 1 = outer skin only; more = sealed internal voids (fine for FDM,
        # needs drain holes for resin or powder printing)
        "surface_shells": int(_components(part)[0]),
        "shell_volume_cm3": round(shell.volume / 1000, 1),
        "part_volume_cm3": round(part.volume / 1000, 1),
        "relative_density": round(part.volume / shell.volume, 3),
        "bounds_mm": np.round(part.bounds, 1).tolist(),
    }
