"""Static linear-elastic analysis and topology optimisation on a voxel grid.

The part is voxelised into cube (hex8) elements. Each element's stiffness is
scaled by how full it is, measured on a finer grid, so a latticed part is
analysed as a homogenised material (E * density^n, Gibson-Ashby) rather than
strut by strut, and a solid shell simply comes out at full stiffness.

Where the part is held and where it is pushed are zones, written like lattice
regions (see build.REGION_KEYS): {"box": [...]}, {"sphere": [...]} or
{"where": "<expr>"}. A load also has "force": [Fx, Fy, Fz] in newtons, the
total spread evenly over the surface nodes in its zone. Units are mm, N and
MPa (N/mm^2), so displacements come out in mm and stresses in MPa.
"""

from dataclasses import asdict, dataclass, field, fields as dc_fields, replace
import time

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

from . import expr
from .sdf import make_grid, mesh_sdf

# Young's modulus (MPa) and Poisson's ratio of common printing materials,
# for printed (not moulded) parts.
MATERIALS = {
    "PLA": (3500.0, 0.36),
    "PETG": (2000.0, 0.38),
    "Nylon PA12": (1700.0, 0.40),
    "TPU 95A": (26.0, 0.45),
    "Resin (tough)": (2200.0, 0.38),
    "Aluminium": (70000.0, 0.33),
}


@dataclass
class SimSpec:
    fixed: list = field(default_factory=list)  # zones held in place
    loads: list = field(default_factory=list)  # zones with "force": [Fx, Fy, Fz], N
    material: str = "PETG"  # a key of MATERIALS, or "custom" with E and nu
    E: float = None  # Young's modulus, MPa (overrides the material)
    nu: float = None  # Poisson's ratio
    voxel: float = None  # element size, mm (default: picked from max_elements)
    max_elements: int = 80_000
    lattice_exponent: float = 2.0  # stiffness ~ density^n for part-full elements
    log: callable = field(default=print, repr=False, compare=False)

    @classmethod
    def from_dict(cls, d, **extra):
        known = {f.name for f in dc_fields(cls)} - {"log"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown simulation keys: {sorted(unknown)}")
        return cls(**d, **extra)

    def to_dict(self):
        d = asdict(self)
        d.pop("log")
        return d

    def elastic(self):
        if self.material not in MATERIALS and self.E is None:
            raise ValueError(f"material must be one of {sorted(MATERIALS)} or give E")
        E, nu = MATERIALS.get(self.material, (None, 0.35))
        return float(self.E if self.E is not None else E), float(self.nu if self.nu is not None else nu)


# ---- element ---------------------------------------------------------------

_CORNERS = np.array([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=float)


def _D(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    return D


def _B(xi):
    """Strain-displacement matrix of a unit hex8 at local point xi in [0, 1]^3."""
    c = _CORNERS
    dN = np.empty((8, 3))
    for a in range(3):
        f = np.where(c[:, a] == 1, 1.0, -1.0)
        o = [b for b in range(3) if b != a]
        dN[:, a] = f * np.prod(np.where(c[:, o] == 1, xi[o], 1 - xi[o]), axis=1)
    B = np.zeros((6, 24))
    for n in range(8):
        dx, dy, dz = dN[n]
        B[0, 3 * n], B[1, 3 * n + 1], B[2, 3 * n + 2] = dx, dy, dz
        B[3, 3 * n], B[3, 3 * n + 1] = dy, dx
        B[4, 3 * n + 1], B[4, 3 * n + 2] = dz, dy
        B[5, 3 * n], B[5, 3 * n + 2] = dz, dx
    return B


def element_stiffness(nu):
    """24x24 stiffness of a unit cube for E = 1. A cube of side h and modulus E
    has E * h times this (2x2x2 Gauss quadrature)."""
    D = _D(1.0, nu)
    g = 0.5 + np.array([-1, 1]) / (2 * np.sqrt(3))
    KE = np.zeros((24, 24))
    for a in g:
        for b in g:
            for c in g:
                B = _B(np.array([a, b, c]))
                KE += B.T @ D @ B / 8
    return KE


# ---- model -----------------------------------------------------------------


class Model:
    """Voxel model of a part: element grid, densities and node numbering."""

    def __init__(self, mesh, sim, design=None):
        self.sim = sim
        size = np.ptp(mesh.bounds, axis=0)
        h = sim.voxel or float(np.cbrt(mesh.volume / max(sim.max_elements, 1) * 1.0))
        h = max(h, float(np.cbrt(np.prod(size) / 4e6)))  # cap the grid size
        self.h = h
        # Element fill fraction from a 3x finer inside test.
        sub = 3
        fine = make_grid(mesh, h / sub, margin=h)
        sdf = mesh_sdf(mesh, fine)
        ne = [(n - 1) // sub for n in fine.shape]
        inside = (sdf[:ne[0] * sub, :ne[1] * sub, :ne[2] * sub] < 0).astype(np.float32)
        rho = inside.reshape(ne[0], sub, ne[1], sub, ne[2], sub).mean(axis=(1, 3, 5))
        self.origin = fine.origin  # corner node (0, 0, 0)
        self.rho = rho
        self.active = rho > 0.05
        sim.log(f"FEA: {int(self.active.sum()):,} elements of {h:.2f} mm")
        if not self.active.any():
            raise ValueError("the part is too thin for the analysis grid; set a smaller voxel")

        # Node numbering over nodes that touch an active element.
        nn = tuple(n + 1 for n in ne)
        used = np.zeros(nn, dtype=bool)
        e = np.argwhere(self.active)
        for c in _CORNERS.astype(int):
            used[tuple((e + c).T)] = True
        self.node_id = np.full(nn, -1, dtype=np.int64)
        self.node_id[used] = np.arange(int(used.sum()))
        self.nodes = np.argwhere(used)  # (n, 3) grid index per node
        self.elements = e
        self.conn = np.stack([self.node_id[tuple((e + c).T)] for c in _CORNERS.astype(int)], axis=1)
        self.dofs = (3 * self.conn[:, :, None] + np.arange(3)).reshape(-1, 24)
        # Nodes on the surface: fewer than 8 active elements around them.
        count = np.zeros(nn, dtype=np.int8)
        for c in _CORNERS.astype(int):
            count[tuple((e + c).T)] += 1
        self.surface = count[used] < 8

    def node_xyz(self):
        return self.origin + self.nodes * self.h

    def element_xyz(self):
        return self.origin + (self.elements + 0.5) * self.h

    def zone_nodes(self, zone, surface=True):
        from .build import _region_inside

        p = self.node_xyz().astype(np.float32)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        z_ = dict(zone)
        if "where" in z_:
            z_["_where"] = expr.compile_expr(z_["where"])
        # A zone is grown by half an element so a thin zone still catches nodes.
        inside = _region_inside(z_, x, y, z, {"x": x, "y": y, "z": z}) > -0.5 * self.h
        if surface:
            inside &= self.surface
        return np.nonzero(inside)[0]

    def stiffness(self, scale):
        """Global stiffness for per-element modulus `scale` (MPa).

        The sparsity pattern and where each element entry lands in it are
        worked out once, so re-assembling (once per optimisation step) is a
        single weighted bincount.
        """
        E, nu = self.sim.elastic()
        KE = (element_stiffness(nu) * self.h).ravel()
        if not hasattr(self, "_pattern"):
            n = 3 * len(self.nodes)
            rows = np.repeat(self.dofs, 24, axis=1).ravel()
            cols = np.tile(self.dofs, (1, 24)).ravel()
            keys, slot = np.unique(rows * n + cols, return_inverse=True)
            del rows, cols
            indptr = np.searchsorted(keys // n, np.arange(n + 1))
            self._pattern = (n, keys % n, indptr, slot.astype(np.int32).reshape(len(self.dofs), 576), len(keys))
        n, indices, indptr, slot, nnz = self._pattern
        data = np.bincount(slot.ravel(), weights=(scale[:, None] * KE[None]).ravel(), minlength=nnz)
        return sparse.csr_matrix((data, indices, indptr), shape=(n, n))

    def boundary(self):
        """Fixed dof mask and load vector from the simulation zones."""
        n = len(self.nodes)
        fixed = np.zeros(3 * n, dtype=bool)
        for z in self.sim.fixed:
            ids = self.zone_nodes(z)
            if len(ids) == 0:
                raise ValueError(f"fixed zone {z.get('name', z)} doesn't touch the part's surface")
            fixed[(3 * ids[:, None] + np.arange(3)).ravel()] = True
        if not fixed.any():
            raise ValueError("add at least one fixed zone, where the part is held")
        f = np.zeros(3 * n)
        for z in self.sim.loads:
            ids = self.zone_nodes(z)
            if len(ids) == 0:
                raise ValueError(f"load zone {z.get('name', z)} doesn't touch the part's surface")
            force = np.asarray(z.get("force", [0, 0, 0]), dtype=float)
            for a in range(3):
                f[3 * ids + a] += force[a] / len(ids)
        if not self.sim.loads:
            raise ValueError("add at least one load")
        return fixed, f


def _rigid_modes(xyz):
    """Six rigid-body modes, the near null space that makes AMG work for elasticity."""
    n = len(xyz)
    B = np.zeros((3 * n, 6))
    c = xyz - xyz.mean(axis=0)
    for a in range(3):
        B[a::3, a] = 1
    B[0::3, 3], B[1::3, 3] = -c[:, 1], c[:, 0]
    B[1::3, 4], B[2::3, 4] = -c[:, 2], c[:, 1]
    B[0::3, 5], B[2::3, 5] = c[:, 2], -c[:, 0]
    return B


def solve(K, f, fixed, xyz, log=print, tol=1e-6, x0=None):
    free = ~fixed
    Kf = K[free][:, free].tocsr()
    ff = f[free]
    u = np.zeros_like(f)
    t0 = time.time()
    try:
        import pyamg

        B = _rigid_modes(xyz)[free]
        ml = pyamg.smoothed_aggregation_solver(Kf, B=B, max_coarse=500, coarse_solver="splu")
        res = []
        uf = ml.solve(ff, x0=None if x0 is None else x0[free], tol=tol, accel="cg", maxiter=300, residuals=res)
        log(f"FEA: solved {Kf.shape[0]:,} unknowns in {len(res)} AMG iterations, {time.time() - t0:.1f}s")
    except ImportError:
        d = Kf.diagonal()
        M = sparse.diags(1 / d)
        uf, info = cg(Kf, ff, rtol=tol, maxiter=20_000, M=M, x0=None if x0 is None else x0[free])
        log(f"FEA: solved {Kf.shape[0]:,} unknowns with CG in {time.time() - t0:.1f}s")
    u[free] = uf
    return u


@dataclass
class Result:
    model: Model
    u: np.ndarray  # (nodes, 3) displacement, mm
    vm: np.ndarray  # (elements,) von Mises stress, MPa
    compliance: float  # f . u, N mm (lower = stiffer)

    def summary(self):
        mag = np.linalg.norm(self.u, axis=1)
        return {"max_displacement_mm": float(mag.max()), "max_stress_mpa": float(np.percentile(self.vm, 99.5)),
                "compliance_nmm": self.compliance, "elements": int(len(self.vm)),
                "element_mm": round(self.model.h, 3)}

    def node_grid(self, values):
        g = np.zeros(self.model.node_id.shape + values.shape[1:], dtype=np.float32)
        g[tuple(self.model.nodes.T)] = values
        return g

    def element_grid(self, values, fill=0.0):
        g = np.full(self.model.rho.shape, fill, dtype=np.float32)
        g[tuple(self.model.elements.T)] = values
        return g

    def sample(self, points, what="displacement"):
        """Interpolate displacement magnitude (mm) or von Mises stress (MPa) at points."""
        m = self.model
        if what == "displacement":
            grid = self.node_grid(np.linalg.norm(self.u, axis=1))
            idx = (np.asarray(points) - m.origin) / m.h
        else:
            grid = _spread(self.element_grid(self.vm), m.active)
            idx = (np.asarray(points) - m.origin) / m.h - 0.5
        return ndimage.map_coordinates(grid, idx.T, order=1, mode="nearest")


def _spread(values, mask, steps=2):
    """Fill cells outside `mask` from their neighbours, so interpolating near
    the surface doesn't blend in zeros."""
    v = np.where(mask, values, 0).astype(np.float32)
    w = mask.astype(np.float32)
    for _ in range(steps):
        vs = ndimage.uniform_filter(v, 3)
        ws = ndimage.uniform_filter(w, 3)
        grow = (w == 0) & (ws > 0)
        v = np.where(grow, vs / np.maximum(ws, 1e-9), v)
        w = np.where(grow, 1.0, w)
    return v


def _stress(model, u, scale):
    """Von Mises stress at element centres."""
    E, nu = model.sim.elastic()
    DB = _D(1.0, nu) @ _B(np.array([0.5, 0.5, 0.5])) / model.h
    ue = u.reshape(-1)[model.dofs]  # (elements, 24)
    s = (ue @ DB.T) * scale[:, None]
    sx, sy, sz, txy, tyz, txz = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + txz ** 2))


def element_modulus(model, density=None):
    E, _ = model.sim.elastic()
    rho = model.rho[tuple(model.elements.T)] if density is None else density
    return E * np.clip(rho, 1e-3, 1) ** model.sim.lattice_exponent


def analyze(mesh, sim: SimSpec, model=None):
    """Static analysis of a closed mesh. Returns a Result."""
    t0 = time.time()
    model = model or Model(mesh, sim)
    scale = element_modulus(model)
    fixed, f = model.boundary()
    K = model.stiffness(scale)
    u = solve(K, f, fixed, model.node_xyz(), sim.log)
    vm = _stress(model, u, scale)
    sim.log(f"FEA: done in {time.time() - t0:.1f}s")
    return Result(model, u.reshape(-1, 3), vm, float(f @ u))


# ---- topology optimisation -------------------------------------------------


@dataclass
class TopoSpec:
    volume_fraction: float = 0.3  # share of the design space to keep
    iterations: int = 30
    penal: float = 3.0  # SIMP penalty
    filter_radius: float = None  # mm (default: 1.5 elements)
    keep: list = field(default_factory=list)  # zones that stay solid (e.g. mounting points)
    move: float = 0.2  # largest density change per iteration
    max_elements: int = 30_000  # coarser than a single analysis, as it solves once per iteration


def topopt(mesh, sim: SimSpec, topo: TopoSpec, progress=None):
    """Minimum-compliance SIMP topology optimisation inside the part.

    Elements in the fixed and load zones, and in `topo.keep`, stay solid.
    Returns (Result of the final design, density per element in [0, 1]).
    """
    from .build import _region_inside

    t0 = time.time()
    if sim.voxel is None and sim.max_elements > topo.max_elements:
        sim = replace(sim, max_elements=topo.max_elements)  # each iteration is a full solve
    model = Model(mesh, sim)
    E, _ = sim.elastic()
    fixed, f = model.boundary()
    xyz = model.node_xyz()
    nel = len(model.elements)
    full = model.rho[tuple(model.elements.T)]  # element fill, caps the density

    # Solid near the boundary conditions and in keep zones.
    c = model.element_xyz().astype(np.float32)
    solid = np.zeros(nel, dtype=bool)
    for zone in list(sim.fixed) + list(sim.loads) + list(topo.keep):
        z = dict(zone)
        if "where" in z:
            z["_where"] = expr.compile_expr(z["where"])
        grow = 1.5 * model.h if zone not in topo.keep else 0
        solid |= _region_inside(z, c[:, 0], c[:, 1], c[:, 2], {"x": c[:, 0], "y": c[:, 1], "z": c[:, 2]}) > -grow

    # Density filter as a sparse matrix on the element grid (cone weights).
    r = (topo.filter_radius or 1.5 * model.h) / model.h
    R = int(np.ceil(r))
    idx = np.full(model.rho.shape, -1, dtype=np.int64)
    idx[tuple(model.elements.T)] = np.arange(nel)
    rows, cols, vals = [], [], []
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                w = r - np.sqrt(dx * dx + dy * dy + dz * dz)
                if w <= 0:
                    continue
                q = model.elements + [dx, dy, dz]
                ok = np.all((q >= 0) & (q < model.rho.shape), axis=1)
                j = np.full(nel, -1)
                j[ok] = idx[tuple(q[ok].T)]
                ok = j >= 0
                rows.append(np.nonzero(ok)[0]); cols.append(j[ok]); vals.append(np.full(ok.sum(), w))
    H = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nel, nel))
    Hs = np.asarray(H.sum(axis=1)).ravel()

    _, nu = sim.elastic()
    KE = element_stiffness(nu) * model.h
    target = topo.volume_fraction * full.sum()
    x = np.where(solid, full, np.clip(topo.volume_fraction, 0.01, 1) * full)
    u = None
    emin = 1e-6
    history = []
    for it in range(topo.iterations):
        xp = np.where(solid, full, (H @ x) / Hs)
        scale = E * (emin + (1 - emin) * xp ** topo.penal)
        K = model.stiffness(scale)
        u = solve(K, f, fixed, xyz, log=lambda *a: None, tol=1e-5, x0=u)
        ue = u[model.dofs]
        ce = np.einsum("ij,jk,ik->i", ue, KE, ue)
        comp = float(f @ u)
        dc = -topo.penal * E * (1 - emin) * xp ** (topo.penal - 1) * ce
        dc = (H.T @ (dc / Hs))
        dv = H.T @ (np.ones(nel) / Hs)
        # Optimality-criteria update with bisection on the volume multiplier.
        lo, hi = 1e-9, 1e9
        while hi / lo > 1 + 1e-4:
            mid = np.sqrt(lo * hi)
            xn = x * np.sqrt(np.maximum(-dc, 0) / (dv * mid))
            xn = np.clip(xn, np.maximum(0, x - topo.move * full), np.minimum(full, x + topo.move * full))
            xn = np.where(solid, full, xn)
            if xn.sum() > target:
                lo = mid
            else:
                hi = mid
        change = float(np.abs(xn - x).max())
        x = xn
        history.append(comp)
        msg = f"topopt: iteration {it + 1}/{topo.iterations}, compliance {comp:.4g} N mm, change {change:.3f}"
        sim.log(msg)
        if progress:
            progress(it + 1, topo.iterations, comp)
        if change < 0.01 and it > 5:
            break
    xp = np.where(solid, full, (H @ x) / Hs)
    scale = E * (emin + (1 - emin) * xp ** topo.penal)
    vm = _stress(model, u, scale)
    sim.log(f"topopt: done in {time.time() - t0:.1f}s, kept {xp.sum() / full.sum():.0%} of the material")
    return Result(model, u.reshape(-1, 3), vm, float(f @ u)), xp


def density_mesh(result, density, shell, pitch=None, iso=0.5):
    """Smooth mesh of an optimised density field, trimmed to the shell."""
    import trimesh
    from skimage.measure import marching_cubes

    m = result.model
    pitch = pitch or m.h / 2
    grid = make_grid(shell, pitch, margin=2 * pitch)
    sdf = mesh_sdf(shell, grid)
    rho = _spread(result.element_grid(density), m.active, steps=3)
    rho = ndimage.gaussian_filter(rho, 0.6)
    x, y, z = (grid.axis(i) for i in range(3))
    pts = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    idx = (pts - m.origin) / m.h - 0.5
    r = ndimage.map_coordinates(rho, idx.T, order=1, mode="constant", cval=0).reshape(grid.shape)
    f = np.maximum(sdf, (iso - r) * m.h * 2)
    f[np.abs(f) < 1e-4] = 1e-4
    v, fc, _, _ = marching_cubes(f, level=0.0, spacing=(pitch,) * 3)
    out = trimesh.Trimesh(v + grid.origin, fc, process=True)
    if out.volume < 0:
        out.invert()
    return out
