"""Training data for the learned deformation model.

A sample pairs an input ``(garment_params, body_shape, body_pose)`` with the
draped garment vertices produced by the offline simulation pipeline (Modules
1-4). The network learns ``rest_garment + offset = deformed_garment``, so we
train on per-vertex offsets from a rest shape.

Pure numpy -- assembling, normalizing, and (de)serializing the dataset needs no
torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DeformationDataset:
    """Offsets-from-rest training data for one garment."""

    rest_vertices: np.ndarray         # (V, 3)
    garment_params: np.ndarray        # (N, G)
    shapes: np.ndarray                # (N, S)
    poses: np.ndarray                 # (N, P)
    vertices: np.ndarray              # (N, V, 3) draped results

    def __post_init__(self) -> None:
        self.rest_vertices = np.asarray(self.rest_vertices, dtype=float)
        self.garment_params = np.atleast_2d(np.asarray(self.garment_params, dtype=float))
        self.shapes = np.atleast_2d(np.asarray(self.shapes, dtype=float))
        self.poses = np.atleast_2d(np.asarray(self.poses, dtype=float))
        self.vertices = np.asarray(self.vertices, dtype=float)
        n = self.n_samples
        if not (len(self.garment_params) == len(self.shapes) == len(self.poses) == n):
            raise ValueError("inconsistent sample counts across inputs and vertices")
        if self.vertices.shape[1:] != self.rest_vertices.shape:
            raise ValueError("vertices topology does not match rest_vertices")

    @property
    def n_samples(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.rest_vertices.shape[0])

    @property
    def input_dim(self) -> int:
        return self.garment_params.shape[1] + self.shapes.shape[1] + self.poses.shape[1]

    @property
    def output_dim(self) -> int:
        return self.n_vertices * 3

    def inputs(self) -> np.ndarray:
        """Concatenated input matrix (N, G+S+P)."""
        return np.concatenate([self.garment_params, self.shapes, self.poses], axis=1)

    def offsets(self) -> np.ndarray:
        """Per-vertex offsets from the rest shape, flattened (N, V*3)."""
        return (self.vertices - self.rest_vertices[None]).reshape(self.n_samples, -1)

    def normalization(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature input mean and std (std floored to avoid divide-by-zero)."""
        x = self.inputs()
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return mean, std

    def save(self, path: str) -> str:
        np.savez(
            path,
            rest_vertices=self.rest_vertices,
            garment_params=self.garment_params,
            shapes=self.shapes,
            poses=self.poses,
            vertices=self.vertices,
        )
        return path

    @classmethod
    def load(cls, path: str) -> "DeformationDataset":
        d = np.load(path)
        return cls(
            rest_vertices=d["rest_vertices"],
            garment_params=d["garment_params"],
            shapes=d["shapes"],
            poses=d["poses"],
            vertices=d["vertices"],
        )

    @classmethod
    def from_samples(cls, rest_vertices: np.ndarray, samples: list[dict]) -> "DeformationDataset":
        """Build from a list of ``{garment_params, body_shape, body_pose, vertices}``."""
        return cls(
            rest_vertices=rest_vertices,
            garment_params=np.array([s["garment_params"] for s in samples], dtype=float),
            shapes=np.array([s["body_shape"] for s in samples], dtype=float),
            poses=np.array([s["body_pose"] for s in samples], dtype=float),
            vertices=np.array([s["vertices"] for s in samples], dtype=float),
        )
