"""Cell maps: world coordinates (x, y, z) to lattice coordinates (u, v, w).

rectangular  cells on a straight grid
cylindrical  cells wrap around a vertical axis
spherical    latitude/longitude around the helmet centre, layered by radius
conformal    layered by position through the shell's thickness, so every
             layer follows the skins and a whole number of cells spans the
             shell everywhere. Across the surface it uses an azimuthal
             equidistant projection around the crown, so its only singular
             point is straight below the centre, outside a helmet.

Spherical longitude is scaled so a whole number of cells fits around each
circle, which keeps the lattice seamless.
"""

import numpy as np

MAPS = ("rectangular", "cylindrical", "spherical", "conformal")

_AXES = {"x": 0, "y": 1, "z": 2}


def _frame(x, y, z, center, pole):
    """Coordinates relative to `center`, rotated so the pole axis is last.
    A pole like "-y" points the last axis down y (and flips one of the
    others, so the frame stays right-handed)."""
    p = [x - center[0], y - center[1], z - center[2]]
    sign = -1 if pole.startswith("-") else 1
    i = _AXES[pole.lstrip("+-")]
    a, b = [p[j] for j in range(3) if j != i]
    return a, b * sign, p[i] * sign


def _whole(length, cell):
    return max(1, int(round(length / cell)))


def map_coords(kind, x, y, z, cell, center, ref_radius, pole=None, shell_t=None, layers=1):
    """Return (u, v, w) in mm of lattice space.

    ref_radius  typical radius of the shell's mid-surface (sets cell count around)
    shell_t     0 at the outer surface, 1 at the inner surface (conformal only)
    layers      whole number of cells through the shell (conformal only)
    """
    x, y, z = np.broadcast_arrays(x, y, z)
    if kind == "rectangular":
        return x, y, z
    if kind == "cylindrical":
        a, b, h = _frame(x, y, z, center, pole or "z")
        n = _whole(2 * np.pi * ref_radius, cell)
        theta = np.arctan2(b, a)
        return theta * np.float32(n * cell / (2 * np.pi)), h, np.sqrt(a * a + b * b)
    if kind == "spherical":
        a, b, h = _frame(x, y, z, center, pole or "x")
        r = np.sqrt(a * a + b * b + h * h)
        n = _whole(2 * np.pi * ref_radius, cell)
        theta = np.arctan2(b, a)
        phi = np.arccos(np.clip(h / np.maximum(r, 1e-6), -1, 1))
        return theta * np.float32(n * cell / (2 * np.pi)), phi * np.float32(ref_radius), r
    if kind == "conformal":
        if shell_t is None:
            raise ValueError("conformal map needs the shell thickness field")
        a, b, h = _frame(x, y, z, center, pole or "z")
        rho = np.sqrt(a * a + b * b)
        alpha = np.arctan2(rho, h)  # angle from the crown
        k = np.float32(ref_radius) * alpha / np.maximum(rho, 1e-6)
        return a * k, b * k, shell_t * np.float32(layers * cell)
    raise ValueError(f"unknown cell map {kind!r}; choose from {MAPS}")
