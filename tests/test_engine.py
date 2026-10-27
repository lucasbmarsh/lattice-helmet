import math

import numpy as np
import pytest
import trimesh

from lattice_helmet import LatticeSpec, build, preview
from lattice_helmet import expr
from lattice_helmet.fields import GRAPHS, lattice_kinds

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
