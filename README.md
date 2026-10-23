# lattice-helmet

Phase 1 of an nTop-style lattice tool for Onshape: an implicit (signed distance
field) engine that fills a closed helmet shell with a lattice and writes a
printable 3MF/STL. The Onshape app will wrap this later.

## Install

```
pip install -e .            # numpy, scipy, scikit-image, trimesh, manifold3d
pip install -e .[decimate]  # optional mesh reduction
```

## Use

```
python examples/make_test_helmet.py out/test_helmet.stl
python -m lattice_helmet.cli out/test_helmet.stl out/helmet.3mf \
    --lattice gyroid --cell 12 --wall 1.6 --wall-max 0.8 \
    --grading point --grade-point 0 0 125 --pitch 0.7
python examples/preview.py out/helmet.3mf out/helmet_sections.png
```

Input is any watertight mesh in mm, e.g. a helmet liner exported from Onshape as STL.

| option | meaning |
|---|---|
| `--lattice` | `gyroid`, `diamond`, `schwarz_p` (sheet TPMS) or `bcc` (struts) |
| `--cell` | unit cell size, mm |
| `--wall`, `--wall-max` | wall/strut thickness where the grading field is 0 and 1 |
| `--grading` | `uniform`, `z` (bottom to top), `depth` (surface to core, `--grade-depth` mm), `point` (distance from `--grade-point`, e.g. a crown impact zone) |
| `--skin` | solid skin on every surface, mm (0 = open lattice) |
| `--fillet` | blend radius where lattice meets skin, mm |
| `--pitch` | voxel size, mm. Keep it at or below half the wall thickness |
| `--decimate` | fraction of triangles to remove after meshing |

The run prints a JSON report: triangle count, watertightness, number of surface
shells, volumes and relative density.

## How it works

1. `sdf.py` voxelises the shell with one parity ray per column, then builds a
   signed distance field from two Euclidean distance transforms.
2. `fields.py` evaluates the lattice as a distance field: TPMS use
   `|f| / |grad f|` with analytic gradients so wall thickness is in mm, struts
   use exact distance to the cell diagonals. A grading field in [0, 1]
   interpolates the thickness.
3. `build.py` combines them as `max(shell, smooth_min(skin, lattice))` and runs
   marching cubes in z-slabs that overlap by one slice so they stitch exactly.

## Known limits

- The shell SDF is accurate to about half a voxel at the surface; the lattice
  itself is analytic and smooth.
- Sheet TPMS walls come out about 5% thinner than requested (first-order
  distance estimate); BCC nodes are slightly under-filled.
- A fully skinned sheet TPMS seals its two labyrinths as internal voids
  (`surface_shells` > 1). Fine for FDM; resin and powder printing need drain
  holes, which are on the roadmap.
- `--decimate` can leave a few non-manifold edges where thin walls touch. The
  report flags it; slicers usually repair it.
- A 0.5 mm run on a full helmet is about 60 M voxels, 4 minutes and 11 M
  triangles on 4 CPU cores. GPU evaluation is on the roadmap.
