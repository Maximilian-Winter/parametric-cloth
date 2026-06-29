# Parametric Cloth

A pipeline for generating 3D clothing from parametric 2D sewing patterns, using
automated cloth simulation and learned runtime deformation. Garments are defined
as code — 2D pattern geometry, construction rules, and fabric properties.

See [`parametric-cloth-pipeline-final.md`](parametric-cloth-pipeline-final.md)
for the full 10-module design.

## Status

| Module | Description | Status |
|--------|-------------|--------|
| 1. Pattern Definition | Parametric data model, fabric presets, templates | ✅ implemented |
| 2. Avatar System | SMPL-X parametric body + landmarks | ✅ implemented¹ |
| 3. Cloth Simulation | Headless Blender draping | ✅ implemented² |
| 4. Post-Processing & Export | UVs, normals, bone weights | ✅ implemented³ |
| 5. Variant System | Batch generation + PCA compression | ⬜ planned |
| 6. Learned Deformation | TailorNet-style pose-conditioned net | ⬜ planned |
| 7. Game Engine Integration | Customization, ONNX inference | ⬜ planned |
| 8–10. AI extensions | Pattern gen, fabric prediction, diff. fitting | ⬜ optional |

¹ Module 2's geometry (landmark lookup, waist segments, placement math) is
implemented and unit-tested with numpy. SMPL-X mesh generation needs the
`smplx`/`torch` packages plus model weights, and the placement *binding* needs
Blender (`bpy`) — both imported lazily. The landmark vertex indices are
**provisional** until confirmed with `scripts/verify_landmarks.py` (see
`LANDMARKS_VERIFIED`).

² Module 3's geometry pipeline — tessellation (ear clipping + subdivision),
seam-vertex correspondence, garment assembly, result validation, and the
fabric→solver mapping — is implemented and unit-tested with numpy. The headless
Blender driver (`simulation/blender_sim.py`, run via
`scripts/simulate_garment.py`) needs `bpy` and is exercised inside Blender. It is
built entirely on the tested pure modules: Blender just uploads the assembled
mesh, welds the precomputed seam pairs, runs the solver, and exports.

³ Module 4's UV-from-pattern atlas packing, layout reference (SVG), and package
metadata are pure numpy and unit-tested. Normal baking, bone-weight transfer, and
decimation are Blender-side (`postprocess/blender_post.py`, lazy `bpy`).

## Install

```bash
pip install -e .            # core (Module 1, pure standard library)
pip install -e ".[dev]"     # + pytest
pip install -e ".[avatar]"  # Module 2: numpy + smplx + torch + trimesh
```

## Quick start

```python
from parametric_cloth import create_tshirt, create_skirt, save_garment, FabricType

shirt = create_tshirt(fabric=FabricType.DENIM, ease=1.25)
assert shirt.is_valid()
save_garment(shirt, "tshirt.json")

skirt = create_skirt(panels=6, flare=1.8)
```

Player-facing customization maps friendly options to pattern parameters:

```python
from parametric_cloth import TShirtCustomization

garment = TShirtCustomization(
    fit="oversized", length="longline", sleeve_length="long", neckline="v_neck",
).build()
```

## CLI

```bash
create-garment --type tshirt --fit regular --length regular --output tshirt.json
create-garment --type skirt --panels 6 --flare 1.8 --output skirt.json
create-garment --type cape --fabric wool --output cape.json
```

The emitted JSON is the interchange format consumed by later pipeline stages
(e.g. the Blender simulation script in Module 3).

## Avatar & placement (Module 2)

```python
from parametric_cloth import create_skirt
from parametric_cloth.avatar import generate_smplx_avatar, place_garment

avatar = generate_smplx_avatar("athletic")     # needs smplx + weights
skirt = create_skirt(panels=6)
transforms = place_garment(skirt, avatar)       # piece_name -> world transform
```

`place_garment` resolves each piece's `PlacementHint`: landmark anchors come
from the body mesh, and skirt `waist_segment_*` anchors are auto-computed by
slicing the body at waist height. The result is a posed starting state for the
Module 3 simulation. The geometry runs on any `AvatarMesh` (vertices + faces),
so it is testable without SMPL-X.

## Cloth simulation (Module 3)

The geometry pipeline runs without Blender:

```python
from parametric_cloth import create_skirt
from parametric_cloth.avatar import generate_smplx_avatar, place_garment
from parametric_cloth.simulation import assemble_garment, SimulationConfig

skirt = create_skirt(panels=4)
avatar = generate_smplx_avatar("average")
transforms = place_garment(skirt, avatar)
assembled = assemble_garment(skirt, transforms, levels=2)
# -> assembled.vertices/faces (one welded cloth mesh) + assembled.seam_pairs
```

The actual draping runs headless in Blender:

```bash
blender --background --python scripts/simulate_garment.py -- \
    --garment skirt.json --avatar smplx_average.obj --output output/skirt.fbx
```

It places the pieces, assembles + welds them, runs the solver, validates the
result (detecting explosions/NaNs), and retries with progressively higher
damping on failure (`SimulationConfig.damping_schedule`).

## Post-processing & export (Module 4)

Because each mesh vertex carries its original 2D pattern coordinate, the pattern
*is* the UV map — no unwrapping. The atlas packing, layout reference image, and
metadata are pure:

```python
from parametric_cloth.postprocess import pack_uv_atlas, write_package

atlas = pack_uv_atlas(assembled)        # each panel -> a UV island, aspect-preserved
pkg = write_package("garments/skirt", skirt, assembled, atlas=atlas)
# -> garments/skirt/{metadata.json, uv_layout.svg}
```

The Blender step (`postprocess/blender_post.py`) adds normal-map baking,
bone-weight transfer (Surface Deform), decimation, and the final
`mesh.fbx` + `normal.png`. Pass `--package-dir` to `simulate_garment.py` to run
the whole draping→package flow at once. UVs are assigned from the pattern
*before* seam welding, so the merge produces correct panel-boundary UV seams.

## Data model

- `GarmentDefinition` — a complete garment: pieces, seams, simulation settings.
- `PatternPiece` — one flat 2D polygon (centimeters) with a fabric and a
  `PlacementHint` anchoring it to a body landmark.
- `Seam` — connects two `SeamEdge` ranges; pulled together during simulation.
- `FabricProperties` — KES-F-style physical parameters; `from_preset(...)` for
  the six built-in fabrics.

Every object exposes `.validate()` returning a list of human-readable problems
(`GarmentDefinition.is_valid()` for a boolean), used as geometry sanity checks
before simulation.

## Tests

```bash
python -m pytest
```
