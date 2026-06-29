"""Pure-numpy inference for the deformation MLP.

The trained network is small (a few dense ReLU layers), so its forward pass is a
handful of matmuls. Implementing it in numpy gives a dependency-free reference
runtime: it needs neither torch nor onnxruntime, so the learned-deformation path
is testable here and usable as a fallback in the engine (Module 7).

Weight files are produced by ``train.export_to_npz`` (from a trained torch
model) and consumed by :meth:`NumpyMLP.load`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class NumpyMLP:
    """A ReLU MLP that predicts per-vertex offsets from the rest shape."""

    weights: list[np.ndarray]         # per layer (in_dim, out_dim)
    biases: list[np.ndarray]          # per layer (out_dim,)
    n_vertices: int
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None

    @property
    def input_dim(self) -> int:
        return int(self.weights[0].shape[0])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Predict offsets for inputs ``x`` (D,) or (N, D) -> (N, V, 3)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.input_mean is not None and self.input_std is not None:
            x = (x - self.input_mean) / self.input_std
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:        # ReLU on all but the last layer
                x = _relu(x)
        return x.reshape(x.shape[0], self.n_vertices, 3)

    def predict_offsets(self, garment_params, shape, pose) -> np.ndarray:
        """Convenience: concatenate the three input blocks and run forward (V, 3)."""
        x = np.concatenate([
            np.asarray(garment_params, float).reshape(-1),
            np.asarray(shape, float).reshape(-1),
            np.asarray(pose, float).reshape(-1),
        ])
        return self.forward(x)[0]

    def save(self, path: str) -> str:
        arrays: dict[str, np.ndarray] = {
            "n_layers": np.array(len(self.weights)),
            "n_vertices": np.array(self.n_vertices),
        }
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        if self.input_mean is not None:
            arrays["input_mean"] = self.input_mean
            arrays["input_std"] = self.input_std
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> "NumpyMLP":
        d = np.load(path)
        n_layers = int(d["n_layers"])
        weights = [d[f"W{i}"] for i in range(n_layers)]
        biases = [d[f"b{i}"] for i in range(n_layers)]
        mean = d["input_mean"] if "input_mean" in d else None
        std = d["input_std"] if "input_std" in d else None
        return cls(
            weights=weights, biases=biases, n_vertices=int(d["n_vertices"]),
            input_mean=mean, input_std=std,
        )
