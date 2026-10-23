"""Fill a closed shell mesh with a lattice and mesh the result."""

from dataclasses import dataclass, field
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from . import fields
from .sdf import make_grid, mesh_sdf


@dataclass
class LatticeSpec:
    lattice: str = "gyroid"  # gyroid | diamond | schwarz_p | bcc
    cell: float = 10.0  # unit cell size, mm
    wall_min: float = 0.8  # wall / strut thickness where grading = 0, mm
    wall_max: float = 0.8  # wall / strut thickness where grading = 1, mm
    grading: str = "uniform"  # uniform | z | depth | point
    grade_point: tuple = (0.0, 0.0, 0.0)  # used by grading="point"
    grade_depth: float = 10.0  # used by grading="depth", mm
    skin: float = 1.2  # solid skin on every surface of the part, mm (0 = none)
    fillet: float = 0.5  # blend radius between lattice and skin, mm
    pitch: float = 0.5  # voxel size, mm
    slab: int = 64  # z-slices meshed per chunk
    decimate: float = 0.0  # fraction of triangles to remove (0 = keep all)
    log: callable = field(default=print, repr=False)


def build(shell: trimesh.Trimesh, spec: LatticeSpec) -> trimesh.Trimesh:
    if spec.lattice not in fields.LATTICES:
        raise ValueError(f"lattice must be one of {sorted(fields.LATTICES)}")
    if not shell.is_watertight:
        spec.log("warning: input mesh is not watertight; inside test may leak")

    t0 = time.time()
    grid = make_grid(shell, spec.pitch, margin=3 * spec.pitch + spec.fillet)
    nvox = int(np.prod(grid.shape))
    spec.log(f"grid {grid.shape} at {spec.pitch} mm ({nvox / 1e6:.1f} M voxels)")
    sdf = mesh_sdf(shell, grid)
    spec.log(f"shell SDF done in {time.time() - t0:.1f}s")

    lattice_fn = fields.LATTICES[spec.lattice]
    bounds = shell.bounds.astype(np.float32)
    nz = grid.shape[2]
    pieces = []
    for k0 in range(0, nz - 1, spec.slab):
        k1 = min(k0 + spec.slab + 1, nz)  # one slice of overlap so chunks stitch
        s = sdf[:, :, k0:k1]
        x, y, z = grid.coords(k0, k1)
        t = fields.grading(
            spec.grading, x, y, z, s, bounds,
            point=spec.grade_point, depth_range=spec.grade_depth,
        )
        wall = spec.wall_min + (spec.wall_max - spec.wall_min) * t
        lat = lattice_fn(x, y, z, spec.cell, wall)
        if spec.skin > 0:
            body = fields.smooth_min(-s - spec.skin, lat, spec.fillet)
        else:
            body = lat
        f = np.maximum(s, body)  # keep everything inside the shell
        if f.min() >= 0 or f.max() <= 0:
            continue
        v, faces, _, _ = marching_cubes(f, level=0.0, spacing=(spec.pitch,) * 3)
        v += grid.origin + np.array([0, 0, k0 * spec.pitch])
        pieces.append((v, faces))
    spec.log(f"meshed {len(pieces)} chunks in {time.time() - t0:.1f}s")

    verts, faces, offset = [], [], 0
    for v, fc in pieces:
        verts.append(v)
        faces.append(fc + offset)
        offset += len(v)
    out = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
    out.merge_vertices(digits_vertex=6)
    out.update_faces(out.nondegenerate_faces())
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
    out = drop_specks(out)
    if out.volume < 0:
        out.invert()
    spec.log(f"done in {time.time() - t0:.1f}s: {len(out.faces):,} triangles")
    return out


def _components(mesh):
    e = mesh.edges_unique
    n = len(mesh.vertices)
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(adj, directed=False)


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
