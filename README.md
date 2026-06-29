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
| 2. Avatar System | SMPL-X parametric body + landmarks | ⬜ planned |
| 3. Cloth Simulation | Headless Blender draping | ⬜ planned |
| 4. Post-Processing & Export | UVs, normals, bone weights | ⬜ planned |
| 5. Variant System | Batch generation + PCA compression | ⬜ planned |
| 6. Learned Deformation | TailorNet-style pose-conditioned net | ⬜ planned |
| 7. Game Engine Integration | Customization, ONNX inference | ⬜ planned |
| 8–10. AI extensions | Pattern gen, fabric prediction, diff. fitting | ⬜ optional |

## Install

```bash
pip install -e .          # core (Module 1, pure standard library)
pip install -e ".[dev]"   # + pytest
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
