"""Command line: lattice-helmet shell.stl out.3mf [options]

Options override values from --recipe; --save-recipe writes the final spec.
"""

import argparse
import json

import trimesh

from .build import LatticeSpec, build, report
from .fields import lattice_kinds
from .maps import MAPS


def _parser():
    ap = argparse.ArgumentParser(description="Fill a closed shell mesh with a lattice.")
    ap.add_argument("input", help="closed shell mesh (STL/OBJ/3MF), mm")
    ap.add_argument("output", nargs="?", help="output mesh, .3mf or .stl")
    ap.add_argument("--recipe", help="JSON recipe to start from")
    ap.add_argument("--save-recipe", help="write the final recipe to this JSON file")
    o = ap.add_argument_group("lattice (each overrides the recipe)")
    o.add_argument("--lattice", choices=lattice_kinds())
    o.add_argument("--graph", help="JSON file of strut segments for --lattice custom")
    o.add_argument("--cell", type=float, help="unit cell size, mm")
    o.add_argument("--cell-map", choices=MAPS)
    o.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"), help="helmet centre")
    o.add_argument("--pole", choices=["x", "y", "z"], help="pole axis for spherical/conformal maps")
    o.add_argument("--layers", type=int, help="cells through the shell (conformal)")
    o.add_argument("--wall", dest="wall_min", type=float, help="wall/strut thickness, mm")
    o.add_argument("--wall-max", type=float, help="thickness where grading = 1 (default: --wall)")
    o.add_argument("--grading", choices=["uniform", "z", "depth", "point", "expr"])
    o.add_argument("--grade-point", type=float, nargs=3, metavar=("X", "Y", "Z"))
    o.add_argument("--grade-depth", type=float, help="mm, for --grading depth")
    o.add_argument("--grade-expr", help="expression giving 0..1, e.g. 'smoothstep(60, 0, dist(x,y,z,0,0,125))'")
    o.add_argument("--skin", type=float, help="solid skin thickness, mm (0 = none)")
    o.add_argument("--skin-surfaces", nargs="+", choices=["all", "outer", "inner", "rim"])
    o.add_argument("--hole", dest="holes", type=float, nargs=4, action="append",
                   metavar=("X", "Y", "Z", "R"), help="drain hole through the skin (repeatable)")
    o.add_argument("--fillet", type=float, help="lattice-to-skin blend radius, mm")
    o.add_argument("--pitch", type=float, help="voxel size, mm (smaller = finer, slower)")
    o.add_argument("--decimate", type=float, help="fraction of triangles to remove, e.g. 0.7")
    return ap


def main(argv=None):
    a = _parser().parse_args(argv)
    spec = {}
    if a.recipe:
        with open(a.recipe) as f:
            spec = json.load(f)
    overrides = {k: v for k, v in vars(a).items()
                 if v is not None and k not in ("input", "output", "recipe", "save_recipe", "graph")}
    if a.graph:
        with open(a.graph) as f:
            overrides["graph"] = json.load(f)
    if "wall_min" in overrides and "wall_max" not in overrides and "wall_max" not in spec:
        overrides["wall_max"] = overrides["wall_min"]
    spec.update(overrides)
    spec = LatticeSpec.from_dict(spec)

    if a.save_recipe:
        with open(a.save_recipe, "w") as f:
            f.write(spec.to_json())
    if not a.output:
        return
    shell = trimesh.load_mesh(a.input)
    part = build(shell, spec)
    part.export(a.output)
    print(json.dumps(report(shell, part), indent=2))


if __name__ == "__main__":
    main()
