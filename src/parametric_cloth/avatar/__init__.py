"""Module 2: Avatar System.

A standardized SMPL-X parametric body with deterministic landmark positions,
plus the geometry to place Module 1 pattern pieces on it.

Note: importing this package requires ``numpy``. The SMPL-X generation/export
helpers in :mod:`parametric_cloth.avatar.body` additionally need ``smplx`` /
``torch`` / ``trimesh`` (and Blender for FBX), imported lazily on use.
"""

from __future__ import annotations

from .body import SMPLXConfig, export_avatar, generate_smplx_avatar
from .landmarks import (
    LANDMARKS_VERIFIED,
    LEFT_RIGHT_PAIRS,
    SMPLX_LANDMARKS,
    SMPLX_NUM_VERTICES,
    verify_landmark_indices,
)
from .mesh import AvatarMesh, compute_vertex_normals, normalize
from .placement import (
    AnchorResolver,
    PlacementTransform,
    axis_angle_matrix,
    basis_from_normal,
    build_resolver_for_garment,
    compute_placement_transform,
    place_garment,
    position_piece_on_avatar,
)
from .segments import compute_waist_segments
from .shapes import BODY_SHAPE_SAMPLES, N_BETAS, get_body_shape

__all__ = [
    "AvatarMesh",
    "compute_vertex_normals",
    "normalize",
    "SMPLX_LANDMARKS",
    "SMPLX_NUM_VERTICES",
    "LANDMARKS_VERIFIED",
    "LEFT_RIGHT_PAIRS",
    "verify_landmark_indices",
    "BODY_SHAPE_SAMPLES",
    "N_BETAS",
    "get_body_shape",
    "compute_waist_segments",
    "AnchorResolver",
    "PlacementTransform",
    "compute_placement_transform",
    "basis_from_normal",
    "axis_angle_matrix",
    "build_resolver_for_garment",
    "place_garment",
    "position_piece_on_avatar",
    "SMPLXConfig",
    "generate_smplx_avatar",
    "export_avatar",
]
