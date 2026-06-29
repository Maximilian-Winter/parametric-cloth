"""A compact, on-disk garment variant library.

Stores one PCA basis plus a coefficient vector per named variant, so a whole
family of garments costs ~one basis + a few floats each. Layout mirrors the
design::

    garments/<garment>/
      pca_basis.npz          # mean shape + components
      variants/<name>.json   # PCA coefficients + source parameters
      metadata.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .pca import PCABasis, build_pca_basis

BASIS_FILENAME = "pca_basis.npz"
METADATA_FILENAME = "metadata.json"
VARIANTS_DIR = "variants"

# A simulator maps a parameter dict to a deformed mesh (V, 3), or None on failure.
Simulator = Callable[[dict], Optional[np.ndarray]]


@dataclass
class VariantLibrary:
    """A PCA basis plus named variant coefficients and their source parameters."""

    basis: PCABasis
    variants: dict[str, np.ndarray] = field(default_factory=dict)
    parameters: dict[str, dict] = field(default_factory=dict)

    def reconstruct(self, name: str) -> np.ndarray:
        """Reconstruct a stored variant's mesh (V, 3)."""
        if name not in self.variants:
            raise KeyError(f"unknown variant '{name}'")
        return self.basis.decode(self.variants[name])

    def save(self, out_dir: str) -> str:
        os.makedirs(os.path.join(out_dir, VARIANTS_DIR), exist_ok=True)
        self.basis.save(os.path.join(out_dir, BASIS_FILENAME))

        for name, coeffs in self.variants.items():
            payload = {
                "name": name,
                "coefficients": [float(c) for c in coeffs],
                "parameters": self.parameters.get(name, {}),
            }
            with open(os.path.join(out_dir, VARIANTS_DIR, f"{name}.json"), "w") as fh:
                json.dump(payload, fh, indent=2)

        metadata = {
            "n_variants": len(self.variants),
            "n_components": self.basis.n_components,
            "n_vertices": self.basis.n_vertices,
            "explained_variance_ratio": [
                float(x) for x in self.basis.explained_variance_ratio
            ],
            "variants": sorted(self.variants),
        }
        with open(os.path.join(out_dir, METADATA_FILENAME), "w") as fh:
            json.dump(metadata, fh, indent=2)
        return out_dir

    @classmethod
    def load(cls, out_dir: str) -> "VariantLibrary":
        basis = PCABasis.load(os.path.join(out_dir, BASIS_FILENAME))
        variants: dict[str, np.ndarray] = {}
        parameters: dict[str, dict] = {}
        vdir = os.path.join(out_dir, VARIANTS_DIR)
        for fname in sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []:
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(vdir, fname)) as fh:
                payload = json.load(fh)
            name = payload["name"]
            variants[name] = np.asarray(payload["coefficients"], dtype=float)
            parameters[name] = payload.get("parameters", {})
        return cls(basis=basis, variants=variants, parameters=parameters)


def build_variant_library(
    samples: list[dict],
    simulate: Simulator,
    *,
    n_components: int = 10,
    name_fn: Optional[Callable[[int, dict], str]] = None,
) -> VariantLibrary:
    """Simulate each sample, fit a PCA basis, and encode every variant.

    ``simulate`` is injected so this orchestration is testable without Blender:
    pass the real draping pipeline in production, or a synthetic deformer in tests.
    Samples whose simulation returns ``None`` are skipped.
    """
    name_fn = name_fn or (lambda i, s: f"variant_{i}")
    records: list[tuple[str, dict, np.ndarray]] = []
    for i, sample in enumerate(samples):
        mesh = simulate(sample)
        if mesh is None:
            continue
        records.append((name_fn(i, sample), sample, np.asarray(mesh, dtype=float)))

    if len(records) < 2:
        raise ValueError(
            f"need at least 2 successful simulations to build a basis "
            f"(got {len(records)})"
        )

    basis = build_pca_basis([r[2] for r in records], n_components=n_components)
    variants = {name: basis.encode(mesh) for name, _, mesh in records}
    parameters = {name: params for name, params, _ in records}
    return VariantLibrary(basis=basis, variants=variants, parameters=parameters)
