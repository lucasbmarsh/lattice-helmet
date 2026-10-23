"""Plot cross-sections of a lattice part (and its shell) to a PNG."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def section(ax, mesh, origin, normal, title, color):
    s = mesh.section(plane_origin=origin, plane_normal=normal)
    if s is None:
        return
    planar, _ = s.to_2D(to_2D=_frame(origin, normal))
    for poly in planar.discrete:
        ax.plot(poly[:, 0], poly[:, 1], color=color, lw=0.4)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def _frame(origin, normal):
    # Map the section plane onto XY so plots read with z up.
    n = np.asarray(normal, float)
    up = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(up, n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    T = np.eye(4)
    T[:3, :3] = np.vstack([x, y, n])
    T[:3, 3] = -T[:3, :3] @ np.asarray(origin, float)
    return T


if __name__ == "__main__":
    part = trimesh.load(sys.argv[1], force="mesh")
    out = sys.argv[2] if len(sys.argv) > 2 else "preview.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    section(axes[0], part, (0, 0, 0), (1, 0, 0), "side section (x = 0)", "#c0392b")
    section(axes[1], part, (0, 0, 0), (0, 1, 0), "front section (y = 0)", "#c0392b")
    section(axes[2], part, (0, 0, 60), (0, 0, 1), "plan section (z = 60 mm)", "#c0392b")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(out)
