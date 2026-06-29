"""PCA compression of garment variant meshes.

Instead of storing every parameter combination as a full mesh, we simulate a
chosen subset, compute a PCA basis over their vertex positions, and represent any
variant as a handful of coefficients. Reconstruction is ``mean + coeffs @
components``.

Implemented with a plain numpy SVD (no scikit-learn dependency): for centered
data ``Xc``, ``Xc = U S Vt`` and the rows of ``Vt`` are the principal directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCABasis:
    """A PCA basis over garment meshes of a fixed topology."""

    mean_shape: np.ndarray            # (V, 3)
    components: np.ndarray            # (C, V, 3), orthonormal directions
    explained_variance_ratio: np.ndarray  # (C,)

    @property
    def n_vertices(self) -> int:
        return int(self.mean_shape.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def _components_flat(self) -> np.ndarray:
        return self.components.reshape(self.n_components, -1)   # (C, D)

    def encode(self, vertices: np.ndarray) -> np.ndarray:
        """Project a mesh (V, 3) onto the basis -> coefficients (C,)."""
        flat = np.asarray(vertices, dtype=float).reshape(-1) - self.mean_shape.reshape(-1)
        return self._components_flat() @ flat

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct a mesh (V, 3) from coefficients."""
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if coeffs.shape[0] != self.n_components:
            raise ValueError(
                f"expected {self.n_components} coefficients, got {coeffs.shape[0]}"
            )
        flat = self.mean_shape.reshape(-1) + coeffs @ self._components_flat()
        return flat.reshape(-1, 3)

    def reconstruction_error(self, vertices: np.ndarray) -> float:
        """Mean per-vertex reconstruction distance for a mesh (meters)."""
        recon = self.decode(self.encode(vertices))
        return float(np.linalg.norm(recon - np.asarray(vertices, float), axis=1).mean())

    def save(self, path: str) -> str:
        np.savez(
            path,
            mean_shape=self.mean_shape,
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
        )
        return path

    @classmethod
    def load(cls, path: str) -> "PCABasis":
        data = np.load(path)
        return cls(
            mean_shape=data["mean_shape"],
            components=data["components"],
            explained_variance_ratio=data["explained_variance_ratio"],
        )


def build_pca_basis(variant_meshes: list, n_components: int = 10) -> PCABasis:
    """Compute a PCA basis from simulated variant meshes.

    Args:
        variant_meshes: list of vertex arrays, each (V, 3) with identical topology.
        n_components: number of components to retain (clamped to the data rank).
    """
    if len(variant_meshes) < 2:
        raise ValueError("need at least 2 variants to build a PCA basis")

    meshes = [np.asarray(m, dtype=float) for m in variant_meshes]
    v_shape = meshes[0].shape
    if any(m.shape != v_shape for m in meshes):
        raise ValueError("all variant meshes must share the same topology/shape")

    n_vertices = v_shape[0]
    stacked = np.stack([m.reshape(-1) for m in meshes])     # (N, D)
    mean = stacked.mean(axis=0)
    centered = stacked - mean

    # Centered data has rank <= N-1; never ask SVD for more than is meaningful.
    max_rank = min(centered.shape)
    k = min(n_components, max_rank)

    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k]                                       # (k, D), orthonormal

    variance = singular ** 2 / max(len(meshes) - 1, 1)
    total = variance.sum()
    evr = (variance[:k] / total) if total > 0 else np.zeros(k)

    return PCABasis(
        mean_shape=mean.reshape(n_vertices, 3),
        components=components.reshape(k, n_vertices, 3),
        explained_variance_ratio=evr,
    )
