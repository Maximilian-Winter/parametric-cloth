#!/usr/bin/env python3
"""Build a PCA variant library from simulated variant meshes (no Blender needed).

Reads a ``.npz`` produced by ``batch_simulate.py`` containing:
  * ``vertices``  : float array (N, V, 3) -- one row per variant, shared topology
  * ``names``     : (optional) array of N variant names
  * ``parameters``: (optional) JSON string of a list of N parameter dicts

Writes a ``VariantLibrary`` (pca_basis.npz + variants/*.json + metadata.json).

    python scripts/build_pca_basis.py --input variants.npz --components 10 \
        --output garments/tshirt
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from parametric_cloth.variants.library import VariantLibrary
from parametric_cloth.variants.pca import build_pca_basis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="variants .npz path")
    parser.add_argument("--output", required=True, help="output library directory")
    parser.add_argument("--components", type=int, default=10)
    args = parser.parse_args(argv)

    data = np.load(args.input, allow_pickle=False)
    vertices = data["vertices"]
    n = len(vertices)
    if n < 2:
        print("error: need at least 2 variants", file=sys.stderr)
        return 1

    names = (
        [str(x) for x in data["names"]] if "names" in data
        else [f"variant_{i}" for i in range(n)]
    )
    params = (
        json.loads(str(data["parameters"])) if "parameters" in data
        else [{} for _ in range(n)]
    )

    basis = build_pca_basis(list(vertices), n_components=args.components)
    library = VariantLibrary(
        basis=basis,
        variants={names[i]: basis.encode(vertices[i]) for i in range(n)},
        parameters={names[i]: params[i] for i in range(n)},
    )
    library.save(args.output)

    evr = basis.explained_variance_ratio
    print(f"built basis: {basis.n_components} components, "
          f"{basis.n_vertices} vertices, {n} variants")
    print(f"cumulative explained variance: {float(np.sum(evr)):.4f}")
    print(f"library written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
