"""Parametric Cloth -- from sewing patterns to game-ready garments.

Module 1: Pattern Definition. Garments are described as parametric 2D geometry,
fabric properties, and seam rules; everything downstream consumes this model.
"""

from __future__ import annotations

from .customization import TShirtCustomization
from .fabric import FABRIC_PRESETS, FabricProperties, FabricType
from .pattern import (
    GarmentDefinition,
    PatternPiece,
    PatternVertex,
    PlacementHint,
    Seam,
    SeamEdge,
)
from .serialization import (
    garment_from_dict,
    garment_from_json,
    garment_to_dict,
    garment_to_json,
    load_garment,
    save_garment,
)
from .templates import create_cape, create_skirt, create_tshirt

__version__ = "0.1.0"

__all__ = [
    "FabricType",
    "FabricProperties",
    "FABRIC_PRESETS",
    "PatternVertex",
    "SeamEdge",
    "Seam",
    "PlacementHint",
    "PatternPiece",
    "GarmentDefinition",
    "create_skirt",
    "create_tshirt",
    "create_cape",
    "TShirtCustomization",
    "garment_to_dict",
    "garment_from_dict",
    "garment_to_json",
    "garment_from_json",
    "save_garment",
    "load_garment",
    "__version__",
]
