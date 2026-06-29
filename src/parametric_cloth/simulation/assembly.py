"""Assemble tessellated pattern pieces into one world-space cloth mesh.

This is the bridge from Module 1 geometry + Module 2 placement to a single mesh
ready for the Blender solver. It is pure numpy: the Blender script just uploads
``AssembledGarment.vertices/faces`` and welds ``seam_pairs``. Doing the assembly
and seam correspondence here (not in ``bpy``) is what makes it testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..pattern import GarmentDefinition
from .seams import boundary_loop, closest_point_pairs, seam_vertex_chain
from .tessellate import PieceMesh, tessellate_piece

if TYPE_CHECKING:  # pragma: no cover
    from ..avatar.placement import PlacementTransform

CM_TO_M = 0.01


@dataclass
class _PlacedPiece:
    mesh: PieceMesh
    offset: int                       # global index of this piece's vertex 0
    loop: list[int]                   # boundary loop in *local* indices


@dataclass
class AssembledGarment:
    """A single combined cloth mesh plus the seam welds to apply."""

    name: str
    vertices: np.ndarray              # (V, 3) meters, world space
    faces: np.ndarray                 # (F, 3) int, global indices
    seam_pairs: list[tuple[int, int]]  # global vertex index pairs to weld
    piece_offsets: dict[str, tuple[int, int]]  # name -> (start, end) global range

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    def seam_gap(self) -> float:
        """Mean distance between welded pairs (meters); ~0 once merged."""
        if not self.seam_pairs:
            return 0.0
        a = self.vertices[[i for i, _ in self.seam_pairs]]
        b = self.vertices[[j for _, j in self.seam_pairs]]
        return float(np.linalg.norm(a - b, axis=1).mean())


def _identity_transform():
    return np.zeros(3), np.eye(3)


def assemble_garment(
    garment: GarmentDefinition,
    transforms: Optional[dict[str, "PlacementTransform"]] = None,
    *,
    levels: int = 2,
) -> AssembledGarment:
    """Tessellate, place, and connect every piece of a garment.

    Args:
        garment: the garment definition (Module 1).
        transforms: per-piece world placement (from Module 2 ``place_garment``).
            Missing pieces are placed at the origin with identity rotation, so a
            garment can be assembled flat without an avatar.
        levels: midpoint-subdivision levels passed to the tessellator.
    """
    transforms = transforms or {}
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    placed: dict[str, _PlacedPiece] = {}
    piece_offsets: dict[str, tuple[int, int]] = {}
    cursor = 0

    for piece in garment.pieces:
        mesh = tessellate_piece(piece, levels=levels)

        if piece.name in transforms:
            t = transforms[piece.name]
            location, rotation = np.asarray(t.location), np.asarray(t.rotation)
        else:
            location, rotation = _identity_transform()

        # Local 2D (cm) -> 3D plane (m) -> rotate -> translate into world.
        local = np.zeros((mesh.n_vertices, 3))
        local[:, :2] = mesh.vertices * CM_TO_M
        world = local @ rotation.T + location

        all_vertices.append(world)
        all_faces.append(mesh.faces + cursor)
        placed[piece.name] = _PlacedPiece(
            mesh=mesh, offset=cursor, loop=boundary_loop(mesh.faces)
        )
        piece_offsets[piece.name] = (cursor, cursor + mesh.n_vertices)
        cursor += mesh.n_vertices

    vertices = np.vstack(all_vertices) if all_vertices else np.zeros((0, 3))
    faces = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=np.int64)

    seam_pairs = _resolve_seam_pairs(garment, placed, vertices)

    return AssembledGarment(
        name=garment.name,
        vertices=vertices,
        faces=faces,
        seam_pairs=seam_pairs,
        piece_offsets=piece_offsets,
    )


def _resolve_seam_pairs(garment, placed, vertices) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for seam in garment.seams:
        a = _global_chain(placed, seam.edge_a)
        b = _global_chain(placed, seam.edge_b)
        if not a or not b:
            continue
        local_pairs = closest_point_pairs(vertices[a], vertices[b])
        pairs.extend((a[i], b[j]) for i, j in local_pairs)
    return pairs


def _global_chain(placed: dict[str, _PlacedPiece], edge) -> list[int]:
    pp = placed.get(edge.piece_name)
    if pp is None:
        return []
    local_chain = seam_vertex_chain(
        pp.loop, pp.mesh.n_corners,
        edge.vertex_start_index, edge.vertex_end_index,
    )
    return [pp.offset + v for v in local_chain]
