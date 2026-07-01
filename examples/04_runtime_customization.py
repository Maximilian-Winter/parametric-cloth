#!/usr/bin/env python3
"""Module 7: deform garments at runtime and layer a wardrobe.

Needs: nothing beyond the base install.
Run: python examples/04_runtime_customization.py
"""

from __future__ import annotations

import numpy as np

from parametric_cloth.engine import (
    DeformState,
    PCADeformer,
    RuntimeGarment,
    Wardrobe,
    benchmark,
    resolve_visible_regions,
)
from parametric_cloth.simulation import assemble_garment
from parametric_cloth.templates import create_skirt
from parametric_cloth.variants import build_variant_library, latin_hypercube


def simulate(params: dict) -> np.ndarray:
    garment = create_skirt(panels=6, flare=params["flare"], length=params["length"])
    return assemble_garment(garment, levels=2).vertices


def main() -> None:
    samples = latin_hypercube({"flare": (1.0, 2.2), "length": (40, 70)}, n_samples=12, seed=1)
    library = build_variant_library(samples, simulate, n_components=6)

    skirt = RuntimeGarment("skirt", PCADeformer(library.basis), regions={"hips", "legs"}, layer=1)
    shirt = RuntimeGarment("shirt", PCADeformer(library.basis), regions={"torso"}, layer=2)
    jacket = RuntimeGarment("jacket", PCADeformer(library.basis), regions={"torso", "arms"}, layer=3)

    wardrobe = Wardrobe()
    wardrobe.equip("bottom", "skirt")
    wardrobe.equip("top", "shirt")
    wardrobe.equip("outerwear", "jacket")
    print("equip order (inner -> outer):", [gid for _, gid, _ in wardrobe.ordered_garments()])

    visible = resolve_visible_regions([skirt.coverage(), shirt.coverage(), jacket.coverage()])
    print(f"shirt visible regions:  {visible['shirt'] or '(fully hidden under jacket)'}")
    print(f"jacket visible regions: {visible['jacket']}")

    coeffs = next(iter(library.variants.values()))
    state = DeformState(pca_coefficients=coeffs)
    mesh = skirt.deform(state)
    print(f"\ndeformed skirt mesh shape: {mesh.shape}")

    result = benchmark(lambda: skirt.deform(state), n_iterations=200)
    print(f"PCA deform cost: mean={result.mean_ms:.4f}ms  "
          f"within 5ms/frame budget: {result.within_budget}")


if __name__ == "__main__":
    main()
