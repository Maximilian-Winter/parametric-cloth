"""A simple non-anatomical body proxy for previewing placement/drape without
SMPL-X -- no model weights, no ``smplx``/``torch`` dependency.

Not derived from or compatible with the SMPL-X topology or vertex indices;
built purely from primitive cylinders, with each named anchor resolved to the
nearest point on its own surface. Good enough to place a skirt around a waist
or a T-shirt across chest/shoulders/arms for a quick, dependency-free preview
of the whole pipeline -- not a substitute for a real body model, and not
intended for production placement quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .mesh import AvatarMesh


def _cylinder(
    radius_top: float, radius_bottom: float, height: float, *,
    n_theta: int = 24, n_levels: int = 8, y0: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """A (possibly tapered) open cylinder: vertices (N,3), triangle faces (F,3)."""
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ys = np.linspace(y0, y0 + height, n_levels)
    verts = []
    for li, y in enumerate(ys):
        t = li / max(n_levels - 1, 1)
        r = radius_bottom + (radius_top - radius_bottom) * t
        for th in thetas:
            verts.append((center[0] + r * math.cos(th), y, center[1] + r * math.sin(th)))

    faces = []
    for li in range(n_levels - 1):
        for ti in range(n_theta):
            a = li * n_theta + ti
            b = li * n_theta + (ti + 1) % n_theta
            c = (li + 1) * n_theta + (ti + 1) % n_theta
            d = (li + 1) * n_theta + ti
            # Winding chosen so compute_vertex_normals() points *outward*
            # (verified against a plain cylinder) -- placement offsets a piece
            # along this normal, so an inward-facing normal would tuck it
            # inside the body instead of out past the surface.
            faces.append((a, c, b))
            faces.append((a, d, c))
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)


@dataclass
class SyntheticBody:
    """A torso + arm-stub body proxy, plus the anchors the built-in templates need."""

    mesh: AvatarMesh
    hip_radius: float
    chest_radius: float
    hip_height: float
    chest_height: float
    shoulder_height: float
    anchors: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def radius_at_height(self, y: np.ndarray) -> np.ndarray:
        """Approximate torso radius at a world height (for simple collision)."""
        y = np.asarray(y, dtype=float)
        t = np.clip((y - self.hip_height) / max(self.chest_height - self.hip_height, 1e-9), 0.0, 1.0)
        return self.hip_radius + (self.chest_radius - self.hip_radius) * t


def make_simple_body(
    *,
    hip_radius: float = 0.16,
    chest_radius: float = 0.15,
    hip_height: float = 0.85,
    chest_height: float = 1.30,
    shoulder_height: float = 1.40,
    shoulder_offset: float = 0.19,
    arm_radius: float = 0.045,
    arm_length: float = 0.28,
    n_theta: int = 28,
    n_levels: int = 10,
) -> SyntheticBody:
    """Build a torso+arm-stub body proxy and its garment-placement anchors.

    Provides exactly the anchors ``create_tshirt``'s pieces reference
    (``chest_front``, ``chest_back``, ``left_upper_arm``, ``right_upper_arm``)
    plus ``hip_height`` for ``compute_waist_segments`` (skirts don't use a
    named anchor -- they need an explicit ``waist_height`` on a non-SMPL-X
    body, since ``waist_front``/``waist_back`` are SMPL-X landmark indices).
    """
    torso_v, torso_f = _cylinder(
        chest_radius, hip_radius, chest_height - hip_height,
        n_theta=n_theta, n_levels=n_levels, y0=hip_height,
    )
    neck_v, neck_f = _cylinder(
        chest_radius * 0.85, chest_radius, shoulder_height - chest_height,
        n_theta=n_theta, n_levels=4, y0=chest_height,
    )
    left_v, left_f = _cylinder(
        arm_radius, arm_radius, arm_length, n_theta=12, n_levels=5,
        y0=shoulder_height - arm_length, center=(-shoulder_offset, 0.0),
    )
    right_v, right_f = _cylinder(
        arm_radius, arm_radius, arm_length, n_theta=12, n_levels=5,
        y0=shoulder_height - arm_length, center=(shoulder_offset, 0.0),
    )

    offsets = np.cumsum([0, len(torso_v), len(neck_v), len(left_v)])
    vertices = np.vstack([torso_v, neck_v, left_v, right_v])
    faces = np.vstack([torso_f, neck_f + offsets[1], left_f + offsets[2], right_f + offsets[3]])
    mesh = AvatarMesh(vertices=vertices, faces=faces, name="synthetic_body")

    def nearest_anchor(target: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        idx = int(np.argmin(np.linalg.norm(mesh.vertices - np.array(target), axis=1)))
        return mesh.vertices[idx].copy(), mesh.vertex_normals[idx].copy()

    anchors = {
        "chest_front": nearest_anchor((0.0, chest_height, chest_radius)),
        "chest_back": nearest_anchor((0.0, chest_height, -chest_radius)),
        "left_upper_arm": nearest_anchor((-shoulder_offset - arm_radius, shoulder_height - arm_length * 0.5, 0.0)),
        "right_upper_arm": nearest_anchor((shoulder_offset + arm_radius, shoulder_height - arm_length * 0.5, 0.0)),
    }

    return SyntheticBody(
        mesh=mesh, hip_radius=hip_radius, chest_radius=chest_radius,
        hip_height=hip_height, chest_height=chest_height, shoulder_height=shoulder_height,
        anchors=anchors,
    )
