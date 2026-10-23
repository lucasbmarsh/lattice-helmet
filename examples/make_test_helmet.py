"""Build a simple helmet liner (thick ellipsoidal dome with a brow cut) as a test input."""

import sys

import manifold3d as m3d
import numpy as np
import trimesh


def helmet(thickness=25.0, outer=(110.0, 135.0, 125.0)):
    def ellipsoid(r):
        return m3d.Manifold.sphere(1.0, 128).scale(r)

    inner = tuple(v - thickness for v in outer)
    dome = ellipsoid(outer) - ellipsoid(inner)
    # Keep the top half, raised at the front for a brow opening.
    keep = m3d.Manifold.cube((400, 400, 200)).translate((-200, -200, 0))
    brow = m3d.Manifold.cube((400, 200, 60)).translate((-200, 60, 0))
    dome = (dome ^ keep) - brow
    mesh = dome.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test_helmet.stl"
    h = helmet()
    h.export(out)
    print(f"{out}: watertight={h.is_watertight} volume={h.volume / 1000:.0f} cm3 bounds={h.bounds.round(1).tolist()}")
