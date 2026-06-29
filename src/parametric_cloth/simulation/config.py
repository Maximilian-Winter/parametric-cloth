"""Simulation configuration and fabric -> cloth-solver parameter mapping.

Pure data and arithmetic -- no Blender -- so the mapping from a Module 1
``FabricProperties`` to concrete Blender cloth settings is testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..fabric import FabricProperties


class SeamStrategy(Enum):
    """How separate pattern pieces are joined before/during simulation."""

    PRE_MERGE = "pre_merge"     # weld matching edge vertices (Strategy A)
    TWO_PASS = "two_pass"       # shrinkwrap then merge (Strategy B) -- planned
    SOLVER = "solver"           # dedicated sewing-spring solver (Strategy C) -- planned


@dataclass
class SimulationConfig:
    """Top-level knobs for a draping run."""

    frames: int = 250
    quality: int = 15                 # solver substeps per frame
    collision_quality: int = 5
    seam_strategy: SeamStrategy = SeamStrategy.PRE_MERGE
    subdivide_levels: int = 2         # midpoint subdivisions of each pattern piece
    max_retries: int = 3
    explosion_distance: float = 2.0   # meters from avatar center -> "exploded"
    self_collision: bool = True

    # Collision thickness on the avatar (meters).
    collision_thickness_outer: float = 0.002
    collision_thickness_inner: float = 0.001

    def damping_schedule(self) -> list[float]:
        """Damping multipliers tried on successive retries.

        Each retry settles more aggressively, trading realism for stability.
        """
        return [2.0 ** i for i in range(self.max_retries)]  # 1, 2, 4, ...


@dataclass
class ClothSettings:
    """Concrete Blender cloth-modifier settings derived from a fabric."""

    mass: float
    tension_stiffness: float
    compression_stiffness: float
    bending_stiffness: float
    tension_damping: float
    air_damping: float
    friction: float
    quality: int = 15

    def to_blender_dict(self) -> dict[str, float]:
        """Settings keyed by ``cloth.settings`` attribute name."""
        return {
            "mass": self.mass,
            "tension_stiffness": self.tension_stiffness,
            "compression_stiffness": self.compression_stiffness,
            "bending_stiffness": self.bending_stiffness,
            "tension_damping": self.tension_damping,
            "air_damping": self.air_damping,
            "quality": self.quality,
        }


def cloth_settings_from_fabric(
    fabric: FabricProperties,
    *,
    quality: int = 15,
    damping_multiplier: float = 1.0,
) -> ClothSettings:
    """Map physical fabric properties onto Blender cloth-solver settings."""
    return ClothSettings(
        mass=fabric.mass_per_area,
        tension_stiffness=fabric.stiffness,
        compression_stiffness=fabric.stiffness,
        bending_stiffness=fabric.bending,
        tension_damping=fabric.damping * damping_multiplier,
        air_damping=1.0,
        friction=fabric.friction,
        quality=quality,
    )
