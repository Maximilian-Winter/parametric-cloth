#!/usr/bin/env python3
"""Module 5: compress a family of garment variants into a PCA basis.

Needs: nothing beyond the base install -- PCA is pure NumPy (SVD), no
scikit-learn required. Uses the flat assembled mesh as a stand-in "simulator"
(swap in the real Blender draping pipeline -- scripts/batch_simulate.py -- for
production use).
Run: python examples/03_pca_variants.py
"""

from __future__ import annotations

import numpy as np

from parametric_cloth.simulation import assemble_garment
from parametric_cloth.templates import create_skirt
from parametric_cloth.variants import build_variant_library, latin_hypercube


def simulate(params: dict) -> np.ndarray:
    garment = create_skirt(panels=6, flare=params["flare"], length=params["length"])
    return assemble_garment(garment, levels=2).vertices


def main() -> None:
    samples = latin_hypercube({"flare": (1.0, 2.2), "length": (40, 70)}, n_samples=20, seed=7)
    library = build_variant_library(samples, simulate, n_components=8)
    basis = library.basis

    print(f"variants={len(library.variants)}  vertices={basis.n_vertices}  "
          f"components={basis.n_components}")
    print(f"cumulative explained variance: {np.cumsum(basis.explained_variance_ratio).round(4)}")

    errors = [
        basis.reconstruction_error(simulate(library.parameters[name]))
        for name in library.variants
    ]
    print(f"reconstruction error: mean={np.mean(errors):.2e}m  max={np.max(errors):.2e}m")

    full_size = len(library.variants) * basis.n_vertices * 3 * 4
    basis_size = (basis.n_vertices * 3 + basis.n_components * basis.n_vertices * 3) * 4
    coeff_size = len(library.variants) * basis.n_components * 4
    print(f"storage: every mesh stored = {full_size / 1024:.1f}KB   "
          f"vs.  basis + coefficients = {(basis_size + coeff_size) / 1024:.1f}KB")


if __name__ == "__main__":
    main()
