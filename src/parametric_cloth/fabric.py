"""Fabric types and physical properties.

Physical parameters are sourced from KES-F (Kawabata Evaluation System for
Fabrics) style textile measurements and tuned for cloth simulation. Each
property maps onto a setting in the downstream Blender cloth solver (Module 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FabricType(Enum):
    COTTON = "cotton"
    DENIM = "denim"
    SILK = "silk"
    LEATHER = "leather"
    WOOL = "wool"
    LINEN = "linen"


# Physical properties from KES-F textile measurements.
FABRIC_PRESETS: dict[FabricType, dict[str, float]] = {
    FabricType.COTTON:  {"mass_per_area": 0.20, "stiffness": 15.0, "bending": 0.8,  "damping": 5.0,  "friction": 0.5, "stretch_limit": 1.05},
    FabricType.DENIM:   {"mass_per_area": 0.45, "stiffness": 30.0, "bending": 4.5,  "damping": 8.0,  "friction": 0.6, "stretch_limit": 1.02},
    FabricType.SILK:    {"mass_per_area": 0.10, "stiffness": 8.0,  "bending": 0.1,  "damping": 2.0,  "friction": 0.3, "stretch_limit": 1.08},
    FabricType.LEATHER: {"mass_per_area": 0.80, "stiffness": 50.0, "bending": 12.0, "damping": 15.0, "friction": 0.7, "stretch_limit": 1.01},
    FabricType.WOOL:    {"mass_per_area": 0.30, "stiffness": 20.0, "bending": 1.5,  "damping": 7.0,  "friction": 0.5, "stretch_limit": 1.04},
    FabricType.LINEN:   {"mass_per_area": 0.18, "stiffness": 12.0, "bending": 0.6,  "damping": 4.0,  "friction": 0.4, "stretch_limit": 1.06},
}


@dataclass
class FabricProperties:
    """Physical properties controlling simulation behavior."""

    type: FabricType = FabricType.COTTON
    mass_per_area: float = 0.20       # kg/m^2
    stiffness: float = 15.0           # structural stiffness
    bending: float = 0.8              # bending resistance
    damping: float = 5.0              # motion settling speed
    friction: float = 0.5             # surface friction
    stretch_limit: float = 1.05       # max stretch ratio (1.0 = no stretch)

    @classmethod
    def from_preset(cls, fabric_type: FabricType) -> "FabricProperties":
        return cls(type=fabric_type, **FABRIC_PRESETS[fabric_type])

    @classmethod
    def from_description(cls, description: str, **overrides) -> "FabricProperties":
        """Resolve a free-text fabric description (Module 9: AI Fabric Prediction).

        Tries the extended preset table with fuzzy matching first (no ML
        dependency); raises ``LookupError`` if nothing matches closely enough.
        """
        from .fabric_ai.presets import fabric_from_description  # lazy: avoid a
        # module-1 -> module-9 import at package load time.
        return fabric_from_description(description, **overrides)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems, empty if valid."""
        issues: list[str] = []
        if self.mass_per_area <= 0:
            issues.append(f"mass_per_area must be > 0 (got {self.mass_per_area})")
        if self.stiffness <= 0:
            issues.append(f"stiffness must be > 0 (got {self.stiffness})")
        if self.bending < 0:
            issues.append(f"bending must be >= 0 (got {self.bending})")
        if self.damping < 0:
            issues.append(f"damping must be >= 0 (got {self.damping})")
        if not 0.0 <= self.friction <= 1.0:
            issues.append(f"friction must be in [0, 1] (got {self.friction})")
        if self.stretch_limit < 1.0:
            issues.append(f"stretch_limit must be >= 1.0 (got {self.stretch_limit})")
        return issues
