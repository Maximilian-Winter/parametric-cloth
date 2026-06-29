"""Pattern-piece placement on the avatar body.

The geometry -- turning a landmark position + surface normal + offset into a
world transform -- is pure numpy (:func:`compute_placement_transform`,
:class:`PlacementTransform`) and fully unit-tested. The Blender binding
(:func:`position_piece_on_avatar`) is a thin wrapper that imports ``bpy`` lazily
and applies that transform to a ``bpy`` object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..pattern import GarmentDefinition, PlacementHint
from .landmarks import SMPLX_LANDMARKS
from .mesh import AvatarMesh, normalize
from .segments import compute_waist_segments

if TYPE_CHECKING:  # pragma: no cover
    AnchorTable = dict[str, tuple[np.ndarray, np.ndarray]]


def axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotation matrix for ``angle_rad`` about ``axis`` (Rodrigues' formula)."""
    x, y, z = normalize(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def basis_from_normal(normal: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Orthonormal basis (columns x, y, z) whose +z aligns with ``normal``.

    Mirrors Blender's ``Vector.to_track_quat('Z', 'Y')``: local +Z points along
    the surface normal, local +Y points as close to world up as possible.
    """
    z = normalize(normal)
    up_v = np.asarray(up, dtype=float)
    if abs(float(np.dot(z, up_v))) > 0.999:   # normal ~parallel to up; pick another
        up_v = np.array([1.0, 0.0, 0.0])
    x = normalize(np.cross(up_v, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


@dataclass
class PlacementTransform:
    """Result of placing a piece: a world location and an orientation basis."""

    location: np.ndarray              # (3,) meters
    rotation: np.ndarray             # (3, 3) orthonormal, columns = x, y, z axes

    def euler_xyz(self) -> tuple[float, float, float]:
        """Convert the rotation matrix to intrinsic XYZ Euler angles (radians)."""
        m = self.rotation
        sy = math.hypot(m[0, 0], m[1, 0])
        if sy > 1e-6:
            x = math.atan2(m[2, 1], m[2, 2])
            y = math.atan2(-m[2, 0], sy)
            z = math.atan2(m[1, 0], m[0, 0])
        else:  # gimbal lock
            x = math.atan2(-m[1, 2], m[1, 1])
            y = math.atan2(-m[2, 0], sy)
            z = 0.0
        return (x, y, z)


def compute_placement_transform(
    position: np.ndarray,
    normal: np.ndarray,
    placement: PlacementHint,
) -> PlacementTransform:
    """Place a piece relative to a body point.

    The piece is offset ``placement.offset_normal`` centimeters along the
    surface normal and oriented so its +Z faces outward, then spun by
    ``placement.rotation`` degrees about that normal.
    """
    position = np.asarray(position, dtype=float)
    z = normalize(normal)

    offset_m = placement.offset_normal / 100.0          # cm -> m
    location = position + z * offset_m

    rotation = basis_from_normal(z)
    if placement.rotation:
        rotation = axis_angle_matrix(z, math.radians(placement.rotation)) @ rotation

    return PlacementTransform(location=location, rotation=rotation)


class AnchorResolver:
    """Resolves a ``PlacementHint.anchor`` name to a (position, normal) pair.

    Static landmarks come from the mesh; dynamic anchors (e.g. the
    ``waist_segment_*`` points from :func:`compute_waist_segments`) are supplied
    explicitly and take precedence.
    """

    def __init__(
        self,
        mesh: AvatarMesh,
        dynamic: Optional["AnchorTable"] = None,
    ) -> None:
        self.mesh = mesh
        self.dynamic = dict(dynamic) if dynamic else {}

    def resolve(self, anchor: str) -> tuple[np.ndarray, np.ndarray]:
        if anchor in self.dynamic:
            pos, nrm = self.dynamic[anchor]
            return np.asarray(pos, float), np.asarray(nrm, float)
        if anchor in SMPLX_LANDMARKS:
            return self.mesh.landmark_position(anchor), self.mesh.landmark_normal(anchor)
        raise KeyError(
            f"cannot resolve anchor '{anchor}': not a known landmark and not "
            f"provided as a dynamic anchor"
        )

    def transform_for(self, placement: PlacementHint) -> PlacementTransform:
        position, normal = self.resolve(placement.anchor)
        return compute_placement_transform(position, normal, placement)


def build_resolver_for_garment(
    garment: GarmentDefinition, mesh: AvatarMesh
) -> AnchorResolver:
    """Build an :class:`AnchorResolver` for a garment on a body.

    Static landmark anchors resolve against ``mesh`` directly. Dynamic
    ``waist_segment_*`` anchors (used by skirts) are auto-computed from the body
    at the panel count the garment actually requires.
    """
    n_waist = sum(
        1 for p in garment.pieces
        if p.placement and p.placement.anchor.startswith("waist_segment_")
    )
    dynamic: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if n_waist:
        dynamic.update(compute_waist_segments(mesh, n_waist))
    return AnchorResolver(mesh, dynamic=dynamic)


def place_garment(
    garment: GarmentDefinition, mesh: AvatarMesh
) -> dict[str, PlacementTransform]:
    """Compute a world placement transform for every placed piece in a garment.

    Returns ``{piece_name: PlacementTransform}``. Pieces without a
    ``PlacementHint`` are skipped. This is the bridge from Module 1 geometry to a
    posed starting state for the Module 3 simulation.
    """
    resolver = build_resolver_for_garment(garment, mesh)
    transforms: dict[str, PlacementTransform] = {}
    for piece in garment.pieces:
        if piece.placement is None:
            continue
        transforms[piece.name] = resolver.transform_for(piece.placement)
    return transforms


# --- Blender binding (requires bpy; not exercised in unit tests) ------------

def position_piece_on_avatar(pattern_obj, avatar_obj, placement: PlacementHint,
                             resolver: Optional[AnchorResolver] = None) -> None:
    """Position a 2D pattern-piece object near the avatar body surface.

    Uses SMPL-X vertex positions and normals for deterministic placement that
    adapts to any body-shape parameterization. ``resolver`` lets callers inject
    dynamic anchors (e.g. waist segments); if omitted, one is built from the
    avatar object's mesh.
    """
    import bpy  # noqa: F401  (lazy: only needed inside Blender)
    from mathutils import Matrix

    if resolver is None:
        resolver = AnchorResolver(avatar_mesh_from_bpy(avatar_obj))

    transform = resolver.transform_for(placement)

    pattern_obj.location = tuple(float(v) for v in transform.location)
    rot = Matrix([[float(c) for c in row] for row in transform.rotation])
    pattern_obj.rotation_euler = rot.to_euler()


def avatar_mesh_from_bpy(avatar_obj) -> AvatarMesh:
    """Build an :class:`AvatarMesh` from a Blender mesh object (world space)."""
    mesh = avatar_obj.data
    mw = avatar_obj.matrix_world
    vertices = np.array([list(mw @ v.co) for v in mesh.vertices], dtype=float)

    faces = []
    for poly in mesh.polygons:
        idx = list(poly.vertices)
        # Fan-triangulate n-gons so faces are uniformly triangles.
        for k in range(1, len(idx) - 1):
            faces.append((idx[0], idx[k], idx[k + 1]))
    return AvatarMesh(vertices=vertices, faces=np.array(faces, dtype=np.int64),
                      name=avatar_obj.name)
