"""Avatar mesh abstraction and small vector helpers.

``AvatarMesh`` is a deliberately thin wrapper around ``(vertices, faces)`` numpy
arrays. Decoupling the geometry from SMPL-X and Blender means landmark lookup,
waist-segment computation, and placement math all operate on plain numpy and are
unit-testable with a synthetic mesh -- no model weights or ``bpy`` required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .landmarks import SMPLX_LANDMARKS


def normalize(v: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return ``v`` scaled to unit length; zero-length input is returned as-is."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals (V, 3).

    Each face contributes its (un-normalized) cross product, which is
    proportional to face area, so larger faces dominate -- the standard
    smooth-shading normal estimate.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices)

    tris = vertices[faces]                       # (F, 3, 3)
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])

    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths


@dataclass
class AvatarMesh:
    """A body mesh: vertices in meters, triangle faces, optional metadata.

    SMPL-X meshes use a fixed topology, so a vertex index always refers to the
    same anatomical location regardless of body-shape parameters -- which is
    what makes landmark-based placement deterministic.
    """

    vertices: np.ndarray              # (V, 3) float
    faces: np.ndarray                 # (F, 3) int
    name: str = "avatar"
    _vertex_normals: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V, 3), got {self.vertices.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"faces must be (F, 3), got {self.faces.shape}")

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def vertex_normals(self) -> np.ndarray:
        if self._vertex_normals is None:
            self._vertex_normals = compute_vertex_normals(self.vertices, self.faces)
        return self._vertex_normals

    @property
    def center(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def landmark_position(self, name: str) -> np.ndarray:
        """World position of a named landmark (see ``SMPLX_LANDMARKS``)."""
        idx = self._landmark_index(name)
        return self.vertices[idx].copy()

    def landmark_normal(self, name: str) -> np.ndarray:
        """Outward surface normal at a named landmark."""
        idx = self._landmark_index(name)
        return self.vertex_normals[idx].copy()

    def _landmark_index(self, name: str) -> int:
        try:
            idx = SMPLX_LANDMARKS[name]
        except KeyError:
            raise KeyError(
                f"unknown landmark '{name}'; known: {sorted(SMPLX_LANDMARKS)}"
            ) from None
        if not 0 <= idx < self.n_vertices:
            raise IndexError(
                f"landmark '{name}' index {idx} out of range for mesh with "
                f"{self.n_vertices} vertices"
            )
        return idx
