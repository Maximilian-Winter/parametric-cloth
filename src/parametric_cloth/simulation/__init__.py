"""Module 3: Cloth Simulation.

Headless Blender draping of Module 1 garments on Module 2 avatars. The geometry
pipeline -- tessellation, seam correspondence, assembly, validation, and the
fabric -> solver mapping -- is pure numpy and unit-tested. The Blender driver in
:mod:`parametric_cloth.simulation.blender_sim` imports ``bpy``/``bmesh`` lazily
and is exercised inside Blender (see ``scripts/simulate_garment.py``).
"""

from __future__ import annotations

from .assembly import AssembledGarment, assemble_garment
from .config import (
    ClothSettings,
    SeamStrategy,
    SimulationConfig,
    cloth_settings_from_fabric,
)
from .seams import boundary_loop, closest_point_pairs, seam_vertex_chain
from .tessellate import PieceMesh, ear_clip, midpoint_subdivide, tessellate_piece
from .validate import ValidationResult, settle_delta, validate_simulation_result

__all__ = [
    "tessellate_piece",
    "PieceMesh",
    "ear_clip",
    "midpoint_subdivide",
    "boundary_loop",
    "seam_vertex_chain",
    "closest_point_pairs",
    "assemble_garment",
    "AssembledGarment",
    "SimulationConfig",
    "SeamStrategy",
    "ClothSettings",
    "cloth_settings_from_fabric",
    "validate_simulation_result",
    "ValidationResult",
    "settle_delta",
]
