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
| 5. Variant System | Batch generation + PCA compression | ✅ implemented⁴ |
| 6. Learned Deformation | TailorNet-style pose-conditioned net | ✅ implemented⁵ |
| 7. Game Engine Integration | Customization, ONNX inference | ✅ implemented⁶ |
| 8. AI Pattern Generation | Photo/sketch/text → sewing pattern | ✅ implemented⁸ |
| 9. AI Fabric Prediction | Text description → simulation parameters | ✅ implemented⁷ |
| 10. Differentiable Fitting | Auto-refine patterns to match target | ✅ implemented⁹ |

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

⁴ Module 5 is pure numpy — PCA is computed via SVD, so **scikit-learn is not
required**. Only the optional blend-shape exporter touches Blender.

⁵ Module 6's dataset assembly, pose sampling, and a **pure-numpy inference
runtime** (`NumpyMLP`) are implemented and tested. The network and training/ONNX
export use PyTorch (lazy import); training exports both ONNX and a numpy weight
file, so inference runs with neither torch nor onnxruntime.

⁶ Module 7 is pure numpy and fully tested: two deformation paths (PCA blend
shapes / learned offsets), body masking + layering, texture customization, and
wardrobe management. Only `ONNXDeformer` needs an external runtime (lazy import).

⁷ Module 9's extended preset table and fuzzy matcher use only the Python stdlib
(`difflib`) — no ML dependency for the common case. `FabricPredictor` (lazy
`sentence-transformers` + `torch`) handles descriptions that don't match a preset.

⁸ Module 8's GarmentCode adapter and rule-based refinement are pure and tested.
The real SewFormer/ChatGarment backends need model weights not available here —
`PatternGenerator` takes `backend=` as an injectable callable, so the
orchestration (validation, GarmentCode conversion) is tested without them.

⁹ No production differentiable-physics/-rendering framework (Warp, Taichi,
DiffCloth, nvdiffrast, PyTorch3D) is available here, so Module 10 implements its
own differentiable mass-spring simulator and soft-splat silhouette renderer in
plain NumPy — gradients are hand-derived (not autodiff-generated) and verified
against central finite differences. See the section below for what that buys
and where it falls short of a production backend.

## Install

```bash
pip install -e .              # core (Module 1 is pure stdlib; everything else needs numpy, included)
pip install -e ".[dev]"       # + pytest
pip install -e ".[viz]"       # + matplotlib (pattern/drape/loss-curve plots)
pip install -e ".[avatar]"    # Module 2: smplx + torch + trimesh
pip install -e ".[learning]"  # Module 6: torch + onnx + onnxruntime
pip install -e ".[fabric-ai]" # Module 9: sentence-transformers
```

Run `parametric-cloth-doctor` any time to see which of these are installed and
exactly which modules/features that unlocks (plus whether Blender is on PATH).

See [`examples/`](examples/) for runnable scripts covering everything below
that needs nothing beyond the base install.

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

## Variant system (Module 5)

Compress a whole family of garment variants into one PCA basis plus a few
coefficients each — continuous variation at the storage cost of a handful of
shapes:

```python
from parametric_cloth.variants import latin_hypercube, build_variant_library

samples = latin_hypercube({"flare": (1.0, 2.2), "length": (40, 70)}, 20, seed=7)
library = build_variant_library(samples, simulate, n_components=8)  # simulate: params -> (V,3)
library.save("garments/skirt")     # pca_basis.npz + variants/*.json + metadata.json
mesh = library.reconstruct("variant_3")
```

`simulate` is injected (the real draping pipeline in production, a synthetic
deformer in tests). Offline you can also run `scripts/batch_simulate.py` (Blender)
to produce a `variants.npz`, then `scripts/build_pca_basis.py` to fit the basis.
PCA components can be exported as engine blend shapes via `blend_shape_targets` /
`export_pca_as_blend_shapes`. Variants must share topology, so vary continuous
parameters and keep structural ones (panel count, subdivision level) fixed.

## Learned deformation (Module 6)

Train a small pose-conditioned MLP on simulated drapes; at runtime it predicts
vertex offsets in well under a millisecond — no physics solver:

```python
from parametric_cloth.deformation import DeformationDataset, train_deformation_model, export_to_npz

ds = DeformationDataset.load("data/training/tshirt.npz")   # built by the Blender script
result = train_deformation_model(ds, n_epochs=200)          # needs torch
export_to_npz(result, "models/tshirt.npz")                  # for the numpy runtime
```

`scripts/generate_training_data.py` (Blender) drapes the garment across AMASS
poses and body shapes; `scripts/train_deformation.py` trains and exports ONNX +
npz. The `NumpyMLP` runtime loads the npz and runs inference with no torch.

## Runtime customization (Module 7)

```python
from parametric_cloth.engine import (PCADeformer, RuntimeGarment, DeformState,
                                      Wardrobe, resolve_visible_regions)

skirt = RuntimeGarment("skirt", PCADeformer(library.basis), regions={"hips", "legs"}, layer=1)
jacket = RuntimeGarment("jacket", PCADeformer(library.basis), regions={"torso", "arms"}, layer=3)
mesh = skirt.deform(DeformState(pca_coefficients=library.variants["variant_0"]))
visible = resolve_visible_regions([skirt.coverage(), jacket.coverage()])  # layering occlusion

w = Wardrobe(); w.equip("bottom", "skirt"); w.equip("outerwear", "jacket")
```

Two paths share one `deform(DeformState)` interface — `PCADeformer` (blend
shapes) or `LearnedDeformer`/`ONNXDeformer` (pose-conditioned offsets) — so the
engine can swap them. Plus body masking/layering, texture customization
(`apply_customization`), and `benchmark` for the per-garment budget.

## AI fabric prediction (Module 9)

```python
from parametric_cloth.fabric import FabricProperties

props = FabricProperties.from_description("heavy brushed cotton twill")
```

Fuzzy-matches free text against an extended preset table (20+ sub-variants of
the six base fabrics, e.g. `cotton_twill`, `silk_charmeuse`, `denim_heavyweight`)
using stdlib `difflib` — no ML dependency for the common case. Raises
`LookupError` if nothing matches closely; for genuinely novel descriptions,
train a `FabricPredictor` (needs `sentence-transformers` + `torch`) on
`fabric_ai.training_pairs_from_presets()` plus any KES-F-measured data, then
load its exported numpy weights — inference then needs neither dependency.

## AI pattern generation (Module 8)

Photo, sketch, and text all converge on the same `GarmentDefinition` used by
hand-authored templates, via a GarmentCode-shaped interchange format:

```python
from parametric_cloth.ai_pattern import PatternGenerator, garmentcode_to_definition

generator = PatternGenerator()
garment = generator.from_text("A-line midi skirt, 6 panels, high waist")  # needs a backend
garment = generator.refine(garment, "make the sleeves wider and shorten the hem by 5cm")
```

The real backends (SewFormer for photos, ChatGarment for text/sketches) need
model weights not available here, so calling `from_photo`/`from_sketch`/
`from_text` without one raises `ModuleNotFoundError`. Pass `backend=` to inject
a stand-in (for tests, or your own hosted endpoint) — the surrounding
validation and `GarmentDefinition` conversion run either way.
`generator.refine()` uses a deterministic rule-based fallback
(`<action> <target> [by Xcm]`, e.g. "shorten the hem by 5cm") that edits pattern
geometry directly; it recognizes simple numeric directives and leaves anything
else unchanged rather than guessing — wire up an LLM backend for open-ended
feedback the same way as the other `from_*` methods.

`garmentcode_to_definition`/`definition_to_garmentcode` convert between this
package's model and evaluated GarmentCode JSON (panels + vertices + stitches),
the format SewFormer/ChatGarment outputs and GarmentCodeData uses. Note: full
*programmatic* GarmentCode — panels defined by parametric edge generators —
needs `pygarment` to evaluate; this adapter operates on the already-evaluated
form.

## Differentiable physics fitting (Module 10)

Automatically refines a pattern piece's vertex positions to match a target
silhouette or 3D scan, by backpropagating through a draping simulation:

```python
from parametric_cloth.fitting import DifferentiableClothFitter
import numpy as np

fitter = DifferentiableClothFitter(n_sim_steps=12, stiffness=40.0, regularization=0.001)
fitted_garment = fitter.fit_to_3d_scan(garment, target_scan_vertices, piece_name="cape", n_iterations=200)
```

With no `torch`/Warp/Taichi/nvdiffrast/PyTorch3D available, this package
implements its own differentiable physics from scratch:

- **`DifferentiableMassSpring`** — a force-based mass-spring solver (semi-implicit
  Euler) with a hand-derived reverse-mode gradient (the same chain-rule math an
  autodiff framework would generate, written out by hand). Rest lengths are
  computed fresh from the initial positions each call, so gradients correctly
  flow through panel *scale*, not just post-drape position. Verified against
  central finite differences to ~1e-10 on random spring systems and a real
  tessellated panel.
- **`SoftSplatRenderer`** — a simplified differentiable silhouette renderer:
  orthographic projection + Gaussian point splatting, also with a hand-derived
  gradient. Not a full differentiable rasterizer (no triangle coverage,
  occlusion, or anti-aliasing) — it's a lightweight stand-in for
  nvdiffrast/PyTorch3D sufficient to fit a rough silhouette.
- **`chamfer_distance_and_grad`** — symmetric Chamfer distance with an analytic
  gradient, for fitting against a 3D scan/point cloud.
- A from-scratch **Adam** optimizer (trivial once gradients are analytic).

Pinned vertices (the pattern's attachment edge, e.g. a waistband or neckline —
configurable via `pin="min_y"`/`"max_y"`/explicit indices) are frozen both
during simulation *and* in the optimizer, so fitting reshapes the panel without
relocating its anchor.

**Honest characterization from testing this end-to-end:** the Chamfer/3D-scan
path converges robustly even from a very different starting shape (a real test
case went from 43cm to 102cm wide chasing a 104cm-wide target in 300
iterations, >95% loss reduction). The silhouette path is gradient-correct but
converges much more slowly when the start is far from the target — an expected
property of sparse point-splat rendering (weak gradient signal without dense
triangle coverage), not a bug; it works better given a closer initial guess or
a coarse-to-fine `sigma_px` schedule. Prefer `fit_to_3d_scan` when you have scan
data; reach for a production renderer (nvdiffrast/PyTorch3D) if you need robust
silhouette fitting from a poor initial guess.

**Scope:** fits one `PatternPiece` at a time, not a multi-panel garment with
avatar placement/contact — that would route through Modules 2/3's
(non-differentiable) Blender solver, which is exactly why this module needed
its own lightweight differentiable simulator. Fit a full garment by calling
this per piece.

## Tooling: drape preview, plots, and environment checks

A handful of additions aimed purely at making the rest of this easier to use —
none change any module's behavior, they just remove setup friction.

**Blender-free drape preview** (`parametric_cloth.preview`) reuses Module 10's
differentiable mass-spring solver in forward-only mode as a quick sanity check
— see how a single panel hangs with zero setup:

```python
from parametric_cloth.preview import preview_drape
from parametric_cloth.templates import create_cape

draped_vertices = preview_drape(create_cape().pieces[0])  # (V, 3) meters
```

It auto-orients the pinned attachment edge to the top regardless of which
y-convention the template uses (skirts pin `min_y` at the waist, T-shirts pin
`max_y` at the shoulder — see `templates.py`), and defaults to stiffer springs
than the fitter (which is tuned for optimizability, not looks) so the result
doesn't look stretchy. This drapes each piece **independently** — no seams, no
avatar — it's a sanity check for one panel, not a substitute for Module 3.

**Full garment on a body** (`preview_drape_garment_on_body`, also in
`parametric_cloth.preview`) is the closest thing to the pipeline's actual
promise — sewing pattern → 3D garment shape — runnable with nothing but numpy.
It places every piece of a garment on a body using the *real* Module 2
placement math, welds the seams with a pure-NumPy union-find weld
(`simulation.weld.weld_vertices` — the same operation Blender's
`remove_doubles` does after `weld_seams`), and settles the result under
gravity with simple body collision:

```python
from parametric_cloth.avatar.synthetic import make_simple_body
from parametric_cloth.preview import preview_drape_garment_on_body
from parametric_cloth.templates import create_skirt

body = make_simple_body()   # a simple torso+arms proxy -- no SMPL-X needed
skirt = create_skirt(panels=8, waist_half=18, hip_half=24, length=45, flare=1.5)
result = preview_drape_garment_on_body(skirt, body, pin="min_y", n_steps=150)
# result.vertices/faces: a welded, gravity-draped 3D garment mesh
```

`make_simple_body` (`parametric_cloth.avatar.synthetic`) builds a torso+arm-stub
`AvatarMesh` and resolves the anchors `create_tshirt`/`create_skirt` need
(`chest_front`, `chest_back`, `left_upper_arm`, `right_upper_arm`, plus a waist
height) directly against its own surface — it does **not** use SMPL-X vertex
indices, so it works with zero model weights. See `examples/07_garment_on_body.py`
for a full runnable example with rendering.

Getting this right surfaced two real placement bugs, now fixed: a raw
placement transform anchors a piece's *local origin*, but the piece is pinned
at its opposite edge during draping (a T-shirt's shoulder is 65cm from its
local origin at the hem) — left alone, the whole panel height was added on
top of the anchor instead of hanging below it. And `basis_from_normal` always
maps local +Y to world "up," which is backwards for skirts/capes (pinned at
low local-y, with the hem at high local-y that should hang *down*). Both are
fixed in `preview_drape_garment_on_body`.

**Known limitation:** there's no cloth-vs-cloth self-collision, only
garment-vs-body — a skirt (one continuous ring of panels) drapes cleanly, but
a T-shirt's separately-hanging front and back panels can sag into each other
since nothing stops one from passing through the other. Worth knowing before
you rely on the visual result for anything but a rough preview.

**Plotting helpers** (`parametric_cloth.viz`, needs `pip install -e ".[viz]"`)
turn results into pictures instead of numbers: `plot_pattern_piece`/
`plot_pattern_pieces` for 2D pattern outlines, `plot_draped_wireframe` for a 3D
view of a draped mesh (e.g. from `preview_drape`), `plot_garment_on_body` for a
`DrapedGarmentPreview` (colored per pattern piece, body drawn alongside),
`plot_loss_curve` for a `FitResult.losses` curve. `matplotlib` is imported
lazily, so the core package has no plotting dependency. Note: world
coordinates are y-up, but matplotlib's elev/azim treat *its* z-axis as
vertical — `plot_garment_on_body` remaps world y into the z slot it plots so
default-style viewing angles actually look right; a naive `ax.plot(x, y, z)`
here will look like a squashed blob at almost any angle.

**Environment check** (`parametric_cloth.envcheck`, console script
`parametric-cloth-doctor`) reports which optional dependencies are installed
and exactly what each one unlocks, plus whether `blender` is on `PATH` — the
fastest way to answer "what can I actually run on this machine right now."

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
