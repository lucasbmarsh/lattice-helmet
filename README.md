# lattice-helmet

An nTop-style lattice engine for helmets, to be wrapped by an Onshape app. It
fills a closed helmet shell with a field-driven lattice (signed distance
fields throughout) and writes a printable 3MF/STL. Phase 2 adds conformal
cell maps, more unit cells, expression fields, per-surface skins, drain holes
and JSON recipes.

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

Or from a recipe (any option on the command line overrides it):

```
python -m lattice_helmet.cli out/test_helmet.stl out/helmet.3mf --recipe recipes/conformal_gyroid_crown.json
```

Input is any watertight mesh in mm, e.g. a helmet liner exported from Onshape as STL.
The helmet is assumed to sit crown up (+z) with the rim at its lowest point; pass
`--center` if the head centre is elsewhere.

| option | meaning |
|---|---|
| `--lattice` | sheet TPMS `gyroid`, `diamond`, `schwarz_p`, `neovius`; solid TPMS `<name>_network`; struts `sc`, `bcc`, `fcc`, `octet`; `custom` with `--graph segments.json` |
| `--cell` | unit cell size, mm |
| `--cell-map` | `rectangular`, `cylindrical`, `spherical`, `conformal` (layers follow the shell, a whole number of cells through its thickness; `--layers` to force) |
| `--wall`, `--wall-max` | wall/strut thickness where the grading field is 0 and 1. For `_network` lattices it is the offset from the minimal surface (0 = 50% solid) |
| `--grading` | `uniform`, `z`, `depth`, `point`, or `expr` with `--grade-expr` |
| `--grade-expr` | expression giving 0 to 1. Variables `x y z` (mm), `depth` (mm from the nearest surface), `r` (mm from the centre), `shell` (0 outer to 1 inner). Functions `smoothstep lerp clip min max abs sqrt exp log sin cos tan atan2 dist where` |
| `--skin` | solid skin thickness, mm (0 = open lattice) |
| `--skin-surfaces` | `all`, or any of `outer inner rim`. Leaving the rim open lets the lattice drain |
| `--hole X Y Z R` | drain hole through the skin; repeatable |
| `--fillet` | blend radius where lattice meets skin, mm |
| `--pitch` | voxel size, mm. Keep it at or below half the sheet wall, a quarter of the strut diameter |
| `--decimate` | fraction of triangles to remove after meshing |
| `--recipe`, `--save-recipe` | load / write the full spec as JSON |

`custom` graphs are a JSON list of `[[x0, y0, z0], [x1, y1, z1]]` segments in
unit-cell coordinates (0 to 1).

The run prints a JSON report: triangle count, watertightness, number of surface
shells, volumes and relative density.

## How it works

1. `sdf.py` voxelises the shell with one parity ray per column, then builds a
   signed distance field from two Euclidean distance transforms.
2. `maps.py` turns world coordinates into lattice coordinates. The conformal
   map uses position through the shell (from distances to the outer and inner
   surfaces, `surfaces.py`) as one axis and an azimuthal equidistant projection
   around the crown for the other two.
3. `fields.py` evaluates the unit cell there. `build.py` converts it to a
   world-space distance with `R / |grad R|` (numerical gradient), so walls stay
   in mm under any map. Grading fields, including safe expressions
   (`expr.py`), interpolate the thickness.
4. `build.py` combines them as `max(shell, smooth_min(skin, lattice))` and runs
   marching cubes in z-slabs that overlap by one slice so they stitch exactly.

## Known limits

- The shell SDF is accurate to about half a voxel at the surface; the lattice
  itself is analytic and smooth.
- Sheet TPMS walls are calibrated to within about 2% of the requested
  thickness. Struts come out up to 15% under volume when the diameter is only
  4 voxels; use a finer pitch for thin struts.
- The conformal map compresses cells around the circumference toward the rim
  (about 0.64x at the equator), because it flattens the dome from the crown.
- With `--skin-surfaces all`, a sheet TPMS seals its two labyrinths as
  internal voids (`surface_shells` > 1). Fine for FDM; for resin or powder,
  leave the rim open or add `--hole`s.
- Strut lattices are slower than TPMS (octet: about 8 minutes at 0.5 mm on
  4 cores); GPU evaluation is on the roadmap.
- `--decimate` can leave a few non-manifold edges where thin walls touch. The
  report flags it; slicers usually repair it.
- A 0.5 mm run on a full helmet is about 60 M voxels, 4 minutes and 11 M
  triangles on 4 CPU cores. GPU evaluation is on the roadmap.

## Onshape app (Phase 3, in progress)

`app/` is the compute service and the panel that Onshape shows beside a Part
Studio.

```
pip install -e .[app]
export ONSHAPE_ACCESS_KEY=...  ONSHAPE_SECRET_KEY=...   # Onshape > My account > Developer > API keys
# or, when a proxy injects the Authorization header: export ONSHAPE_AUTH=proxy
uvicorn app.main:app --port 8000
```

Open http://localhost:8000/panel/ to use it with an uploaded mesh. Inside
Onshape the panel lists the Part Studio's parts, exports the chosen one as
STL, builds the lattice, and offers the 3MF for download or imports it back
into the document.

| endpoint | does |
|---|---|
| `GET /api/options` | lattice types, cell maps and the default recipe |
| `GET /api/parts?did&wvm&wvmid&eid` | parts in a Part Studio |
| `POST /api/jobs/onshape` | export a part from Onshape and start a build |
| `POST /api/jobs/upload` | start a build from an uploaded mesh |
| `GET /api/jobs/{id}` | status, log and report |
| `GET /api/jobs/{id}/result.3mf` | the printable result |
| `POST /api/jobs/{id}/import` | add the result to the source document |

To show the panel inside Onshape, the service needs a public HTTPS URL. Then,
in the Onshape Developer Portal, create an OAuth application and add a
right-panel extension in the Part Studio context whose action URL is

```
https://<your host>/panel/?documentId={$documentId}&workspaceId={$workspaceId}&elementId={$elementId}
```

The Onshape endpoint paths in `app/onshape.py` have not been run against
cad.onshape.com yet; the tests use a fake Onshape client. Jobs run one at a
time in-process, which suits a single user.
