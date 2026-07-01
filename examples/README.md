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

Two scripts (`02`, `05`) plot a figure if `matplotlib` is installed
(`pip install -e ".[viz]"`); they print results either way.

Not covered here because they need Blender, SMPL-X weights, or torch/ONNX:
Modules 2 (SMPL-X generation), 3, 4 (Blender simulation/export), 6 (training).
Run `parametric-cloth-doctor` to see exactly what's installed and what each
missing piece unlocks.
