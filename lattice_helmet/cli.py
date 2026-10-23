"""Command line: lattice-helmet shell.stl out.3mf [options]"""

import argparse
import json

import trimesh

from .build import LatticeSpec, build, report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fill a closed shell mesh with a lattice.")
    ap.add_argument("input", help="closed shell mesh (STL/OBJ/3MF), mm")
    ap.add_argument("output", help="output mesh, .3mf or .stl")
    ap.add_argument("--lattice", default="gyroid", choices=["gyroid", "diamond", "schwarz_p", "bcc"])
    ap.add_argument("--cell", type=float, default=10.0, help="unit cell size, mm")
    ap.add_argument("--wall", type=float, default=0.8, help="wall/strut thickness, mm")
    ap.add_argument("--wall-max", type=float, help="thickness where grading = 1 (default: --wall)")
    ap.add_argument("--grading", default="uniform", choices=["uniform", "z", "depth", "point"])
    ap.add_argument("--grade-point", type=float, nargs=3, default=(0, 0, 0), metavar=("X", "Y", "Z"))
    ap.add_argument("--grade-depth", type=float, default=10.0, help="mm, for --grading depth")
    ap.add_argument("--skin", type=float, default=1.2, help="solid skin thickness, mm (0 = none)")
    ap.add_argument("--fillet", type=float, default=0.5, help="lattice-to-skin blend radius, mm")
    ap.add_argument("--pitch", type=float, default=0.5, help="voxel size, mm (smaller = finer, slower)")
    ap.add_argument("--decimate", type=float, default=0.0, help="fraction of triangles to remove, e.g. 0.7")
    a = ap.parse_args(argv)

    shell = trimesh.load_mesh(a.input)
    spec = LatticeSpec(
        lattice=a.lattice, cell=a.cell,
        wall_min=a.wall, wall_max=a.wall if a.wall_max is None else a.wall_max,
        grading=a.grading, grade_point=tuple(a.grade_point), grade_depth=a.grade_depth,
        skin=a.skin, fillet=a.fillet, pitch=a.pitch, decimate=a.decimate,
    )
    part = build(shell, spec)
    part.export(a.output)
    print(json.dumps(report(shell, part), indent=2))


if __name__ == "__main__":
    main()
