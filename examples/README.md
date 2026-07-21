# Examples

Runnable scripts covering everything that works with just `pip install -e .`
(no Blender, no GPU, no model weights). Run any of them directly:

```bash
python examples/01_pattern_basics.py
```

| Script | Module(s) | What it shows |
|--------|-----------|----------------|
| `01_pattern_basics.py` | 1 | Creating, validating, and JSON-serializing garments |
| `02_drape_preview.py` | 10 (repurposed) | Blender-free drape preview of a single panel |
| `03_pca_variants.py` | 5 | Compressing garment variants into a PCA basis |
| `04_runtime_customization.py` | 7 | PCA-driven deformation, wardrobe layering, perf budget |
| `05_differentiable_fitting.py` | 10 | Fitting a pattern's shape to a target via gradient descent |
| `06_fabric_and_pattern_ai.py` | 8, 9 | Fuzzy fabric matching and rule-based pattern refinement |
| `07_garment_on_body.py` | 1, 2, 3 (synthetic) | **The actual pipeline promise**: place a garment on a body, weld its seams, and drape it under gravity — a real 3D garment shape, not just numbers |

Three scripts (`02`, `05`, `07`) render a figure if `matplotlib` is installed
(`pip install -e ".[viz]"`); they print results either way. `07` is the one
worth looking at first if you want to see an actual garment shape rather than
console output — it builds a simple body proxy (no SMPL-X needed), places a
skirt and a T-shirt on it using the real Module 2 placement math, welds the
seams with a pure-NumPy union-find weld (`simulation.weld`), and settles the
result under gravity with simple body collision. The skirt result is clean;
the T-shirt shows a known limitation (front and back panels can sag into each
other — there's no cloth-vs-cloth self-collision, only garment-vs-body).

Not covered here because they need Blender, SMPL-X weights, or torch/ONNX:
Modules 2 (SMPL-X generation), 3, 4 (Blender simulation/export), 6 (training).
Run `parametric-cloth-doctor` to see exactly what's installed and what each
missing piece unlocks.
