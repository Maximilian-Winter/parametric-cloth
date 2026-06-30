"""Module 9: AI Fabric Parameter Prediction.

Maps natural-language fabric descriptions to simulation parameters. The
extended preset table and fuzzy matcher (stdlib ``difflib``, no ML dependency)
handle the common case; :class:`FabricPredictor` (lazy ``sentence-transformers``
+ ``torch``) handles genuinely novel descriptions.
"""

from __future__ import annotations

from .predictor import FabricPredictor, PARAM_NAMES, training_pairs_from_presets
from .presets import (
    FABRIC_PRESETS_EXTENDED,
    fabric_from_description,
    fabric_properties_for,
    find_closest_preset,
)

__all__ = [
    "FABRIC_PRESETS_EXTENDED",
    "fabric_properties_for",
    "find_closest_preset",
    "fabric_from_description",
    "FabricPredictor",
    "PARAM_NAMES",
    "training_pairs_from_presets",
]
