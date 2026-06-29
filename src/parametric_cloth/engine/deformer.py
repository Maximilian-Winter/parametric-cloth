"""Runtime garment deformers (the two integration paths from the design).

* :class:`PCADeformer` -- Path A: garment shape varies via PCA coefficients
  (blend shapes); pose handled by skinning. Pure numpy.
* :class:`LearnedDeformer` -- Path B: a pose-conditioned MLP predicts offsets
  from the rest shape (full cloth behavior). Pure numpy via ``NumpyMLP``.
* :class:`ONNXDeformer` -- Path B backed by ONNX Runtime (lazy import) for
  deployment parity with a game engine.

All share :meth:`deform(DeformState) -> (V, 3)` so the engine can swap paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..deformation.runtime import NumpyMLP
from ..variants.pca import PCABasis


@dataclass
class DeformState:
    """Everything a deformer might need for one frame; each uses what applies."""

    pose: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None
    garment_params: Optional[np.ndarray] = None
    pca_coefficients: Optional[np.ndarray] = None


class PCADeformer:
    """Path A: reconstruct the garment from PCA coefficients (blend shapes)."""

    def __init__(self, basis: PCABasis):
        self.basis = basis

    @property
    def n_vertices(self) -> int:
        return self.basis.n_vertices

    def deform(self, state: DeformState) -> np.ndarray:
        coeffs = state.pca_coefficients
        if coeffs is None:
            coeffs = np.zeros(self.basis.n_components)   # -> mean shape
        return self.basis.decode(coeffs)


class LearnedDeformer:
    """Path B: rest shape + MLP-predicted, pose-conditioned offsets."""

    def __init__(self, mlp: NumpyMLP, rest_vertices: np.ndarray):
        self.mlp = mlp
        self.rest_vertices = np.asarray(rest_vertices, dtype=float)
        if self.rest_vertices.shape[0] != mlp.n_vertices:
            raise ValueError("rest vertices count does not match the model")

    @property
    def n_vertices(self) -> int:
        return int(self.rest_vertices.shape[0])

    def deform(self, state: DeformState) -> np.ndarray:
        provided = (state.garment_params, state.shape, state.pose)
        if all(p is None for p in provided):
            return self.rest_vertices.copy()             # rest pose, no inputs
        if any(p is None for p in provided):
            raise ValueError(
                "LearnedDeformer needs garment_params, shape, and pose together"
            )
        offsets = self.mlp.predict_offsets(*provided)
        return self.rest_vertices + offsets


class ONNXDeformer:
    """Path B via ONNX Runtime (for engine deployment parity)."""

    def __init__(self, model_path: str, rest_vertices: np.ndarray):
        import onnxruntime  # lazy
        self.session = onnxruntime.InferenceSession(model_path)
        self.rest_vertices = np.asarray(rest_vertices, dtype=float)

    @property
    def n_vertices(self) -> int:
        return int(self.rest_vertices.shape[0])

    def deform(self, state: DeformState) -> np.ndarray:
        def row(a):
            return np.asarray(a, dtype=np.float32).reshape(1, -1)

        out = self.session.run(
            ["vertex_offsets"],
            {
                "garment_params": row(state.garment_params),
                "shape_params": row(state.shape),
                "pose_params": row(state.pose),
            },
        )[0]
        return self.rest_vertices + out.reshape(-1, 3)
