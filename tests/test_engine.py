import math

import numpy as np
import pytest
import trimesh

from lattice_helmet import LatticeSpec, build, preview
from lattice_helmet import expr
from lattice_helmet.fields import GRAPHS, lattice_kinds
from lattice_helmet.sdf import inside_mask, make_grid

QUIET = dict(log=lambda *a: None)


def _box_fraction(**kw):
    box = trimesh.creation.box((30, 30, 30))
    box.apply_translation((3.3, 4.1, 2.7))  # keep struts off the box faces
    spec = dict(skin=0, fillet=0, cell=10, wall_min=1.0, wall_max=1.0, pitch=0.25, **QUIET)
    spec.update(kw)
    return build(box, LatticeSpec(**spec)).volume / box.volume


@pytest.mark.parametrize("kind, area", [("gyroid", 3.0919), ("schwarz_p", 2.3451), ("diamond", 3.8377)])
def test_sheet_wall_thickness(kind, area):
    # Thin sheet volume fraction = wall * (surface area per cell) / cell^3.
    assert _box_fraction(lattice=kind) == pytest.approx(area / 10, rel=0.04)


def test_network_half_density_at_zero_offset():
    assert _box_fraction(lattice="gyroid_network", wall_min=0, wall_max=0, pitch=0.5) == pytest.approx(0.5, abs=0.02)


def test_strut_graph_counts():
    assert {k: len(v) for k, v in GRAPHS.items()} == {"sc": 12, "bcc": 8, "fcc": 24, "octet": 36}


def test_expr_rejects_code():
    for bad in ("__import__('os')", "x.__class__", "open('f')", "[1, 2]", "'a'", "f(x=1)"):
        with pytest.raises(expr.ExprError):
            expr.compile_expr(bad)
    node = expr.compile_expr("smoothstep(0, 10, dist(x, y, z, 0, 0, 0))")
    assert float(expr.evaluate(node, {"x": 5.0, "y": 0.0, "z": 0.0})) == pytest.approx(0.5)


def _dome():
    outer = trimesh.creation.icosphere(4, radius=60)
    inner = trimesh.creation.icosphere(4, radius=45)
    shell = outer.difference(inner, engine="manifold")
    cut = trimesh.creation.box((200, 200, 100))
    cut.apply_translation((0, 0, -50))
    return shell.difference(cut, engine="manifold")


def test_conformal_dome_is_printable():
    dome = _dome()
    part = build(dome, LatticeSpec(cell_map="conformal", cell=7.5, wall_min=1.0, wall_max=1.0,
                                   pitch=0.75, skin_surfaces=["outer", "inner"],
                                   holes=[[0, 0, 52, 4]], **QUIET))
    assert part.is_watertight
    assert 0.1 < part.volume / dome.volume < 0.6



def test_preview_is_quick_and_reuses_the_shell_sdf():
    dome, cache = _dome(), {}
    mesh, info = preview(dome, LatticeSpec(cell_map="conformal", cell=7.5, **QUIET), voxels=4e5, cache=cache)
    assert len(mesh.faces) == info["triangles"] > 1000
    assert info["pitch"] > 0.5  # coarser than the build's voxels
    assert np.allclose(mesh.bounds, dome.bounds, atol=2 * info["pitch"])
    n = len(cache)
    preview(dome, LatticeSpec(lattice="octet", cell=10, **QUIET), voxels=4e5, cache=cache)
    assert len(cache) == n  # same voxel size, so the SDF was reused


def test_recipe_round_trip():
    spec = LatticeSpec(lattice="octet", cell_map="conformal", holes=[[1, 2, 3, 4]], **QUIET)
    again = LatticeSpec.from_dict(__import__("json").loads(spec.to_json()), **QUIET)
    assert again == spec
    with pytest.raises(ValueError):
        LatticeSpec.from_dict({"bogus": 1})


def test_all_kinds_listed():
    assert {"gyroid", "gyroid_network", "octet", "custom"} <= set(lattice_kinds())


@pytest.mark.parametrize("kind, expect", [
    # wall length per unit plan area, times 1 mm walls, less the corner
    # overlaps: triangles 3 / (cell * sqrt(3) / 2) ~ 0.35, hexagons 2 / cell = 0.2
    ("honeycomb_tri", 0.32), ("honeycomb_hex", 0.18)])
def test_honeycomb_wall_fraction(kind, expect):
    assert _box_fraction(lattice=kind) == pytest.approx(expect, rel=0.12)


def test_voronoi_is_one_connected_foam_and_seeded():
    box = trimesh.creation.box((30, 30, 30))
    spec = dict(lattice="voronoi", cell=8, wall_min=1.2, wall_max=1.2, skin=0, fillet=0, pitch=0.4, **QUIET)
    a = build(box, LatticeSpec(**spec))
    assert a.is_watertight and 0.02 < a.volume / box.volume < 0.2
    b = build(box, LatticeSpec(**spec, seed=7))
    assert abs(a.volume - b.volume) > 1e-3 * a.volume  # a different random foam
    assert _box_fraction(lattice="voronoi_surface", cell=8) > 0.15


def test_regions_override_the_lattice_locally():
    box = trimesh.creation.box((60, 30, 30))
    base = dict(cell=10, wall_min=1.0, wall_max=1.0, skin=0, fillet=0, pitch=0.5, **QUIET)
    plain = build(box, LatticeSpec(**base)).volume
    half_solid = build(box, LatticeSpec(**base, regions=[{"where": "x", "lattice": "solid", "blend": 2}])).volume
    # the x > 0 half is solid, the other half keeps its gyroid
    assert half_solid == pytest.approx(box.volume / 2 + plain / 2, rel=0.05)
    sphere = build(box, LatticeSpec(**base, regions=[{"sphere": [0, 0, 0, 8], "lattice": "empty"}])).volume
    assert sphere < plain
    with pytest.raises(ValueError):
        LatticeSpec(regions=[{"lattice": "solid"}], **QUIET) and build(box, LatticeSpec(regions=[{"lattice": "solid"}], **base))


def _cantilever(size=(100, 10, 10)):
    b = trimesh.creation.box(size)
    b.apply_translation((size[0] / 2, 0, 0))
    h = [s / 2 + 1 for s in size]
    fixed = {"box": [-1, -h[1], -h[2], 0.1, h[1], h[2]]}
    tip = {"box": [size[0] - 0.1, -h[1], -h[2], size[0] + 1, h[1], h[2]], "force": [0, 0, -10]}
    return b, {"fixed": [fixed], "loads": [tip], "E": 2000, "nu": 0.3}


def test_fea_matches_beam_theory():
    from lattice_helmet.fea import SimSpec, analyze

    beam, sim = _cantilever()
    r = analyze(beam, SimSpec.from_dict(dict(sim, voxel=1.25), **QUIET))
    # tip deflection F L^3 / (3 E I) = 2.0 mm, bending stress at the root 6 F L / (b h^2) = 6 MPa
    assert r.summary()["max_displacement_mm"] == pytest.approx(2.0, rel=0.05)
    assert 3.5 < r.summary()["max_stress_mpa"] < 7


def test_topopt_keeps_the_volume_and_gets_stiffer():
    from lattice_helmet.fea import SimSpec, TopoSpec, analyze, density_mesh, topopt

    beam, sim = _cantilever((60, 10, 30))
    sim = SimSpec.from_dict(dict(sim, voxel=2.5), **QUIET)
    res, x = topopt(beam, sim, TopoSpec(volume_fraction=0.4, iterations=12))
    assert x.mean() == pytest.approx(0.4, abs=0.03)
    # The optimised 40% is stiffer than a uniformly 40%-dense part (E * 0.4^3 under SIMP).
    full = analyze(beam, sim).compliance
    assert full < res.compliance < full / 0.4 ** 3
    part = density_mesh(res, x, beam)
    assert part.is_watertight and 0.25 < part.volume / beam.volume < 0.55


def test_stress_grading_thickens_the_loaded_root():
    beam, sim = _cantilever((60, 20, 20))
    spec = dict(cell=6, wall_min=0.3, wall_max=2.0, skin=0, fillet=0, pitch=0.5, grading="stress",
                simulation=dict(sim, voxel=2.0), **QUIET)
    part = build(beam, LatticeSpec(**spec))
    grid = make_grid(part, 0.5)
    solid = inside_mask(part, grid)
    x = grid.axis(0)
    assert solid[x < 15].mean() > 1.5 * solid[x > 45].mean()  # more material near the fixed end


def _y_up(mesh):
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # +z -> +y
    return m


def test_up_direction_is_detected():
    from lattice_helmet.surfaces import detect_up

    dome = _dome()
    assert detect_up(dome) == (2, 1)
    assert detect_up(_y_up(dome)) == (1, 1)
    flipped = dome.copy()
    flipped.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    assert detect_up(flipped) == (2, -1)


def test_geodesic_hex_runs_straight_through_and_ignores_orientation():
    from lattice_helmet.fields import geodesic_hex, geodesic_points

    assert len(geodesic_points(4)) == 162
    # Walls are planes through the centre: the distance scales with radius
    # along a ray, so a hole lines up from the inside to the outside.
    d = np.array([0.3, 0.2, 0.93])
    d /= np.linalg.norm(d)
    near, far = (geodesic_hex(*(r * d), 10, (0, 0, 0), (2, 1), 60) for r in (50, 60))
    assert far == pytest.approx(near * 60 / 50, rel=1e-4)

    dome = _dome()
    spec = dict(lattice="geodesic_hex", cell=12, wall_min=1.5, wall_max=1.5, skin=0, fillet=0, pitch=0.75, **QUIET)
    a = build(dome, LatticeSpec(**spec))
    b = build(_y_up(dome), LatticeSpec(**spec))
    assert a.is_watertight and b.is_watertight
    assert b.volume == pytest.approx(a.volume, rel=0.03)  # same part whichever way it was modelled


def test_edge_zone_makes_a_solid_border():
    dome = _dome()
    spec = dict(lattice="geodesic_hex", cell=12, wall_min=1.5, wall_max=1.5, skin=0, fillet=0, pitch=0.75, **QUIET)
    plain = build(dome, LatticeSpec(**spec)).volume
    banded = build(dome, LatticeSpec(**spec, regions=[{"where": "5 - edge", "lattice": "solid", "blend": 1}]))
    assert banded.is_watertight
    # the rim is a 15 mm wide ring of radius 45..60; a 5 mm band is about a third of it in height
    ring = np.pi * (60 ** 2 - 45 ** 2) * 5
    assert banded.volume - plain == pytest.approx(ring * (1 - plain / dome.volume), rel=0.35)
