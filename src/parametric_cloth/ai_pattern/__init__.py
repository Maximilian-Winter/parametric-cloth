"""Module 8: AI Pattern Generation.

Alternative entry points for garment authoring -- photo, sketch, or text -- all
converging on the same :class:`~parametric_cloth.pattern.GarmentDefinition` used
by hand-authored templates (Module 1). The GarmentCode adapter and rule-based
refinement are pure and tested; the actual SewFormer/ChatGarment backends need
model weights not available here (lazy import, dependency-injectable for tests).
"""

from __future__ import annotations

from .garmentcode import definition_to_garmentcode, garmentcode_to_definition
from .generator import PatternGenerator
from .refine import RefineDirective, apply_directive, parse_feedback, rule_based_refine

__all__ = [
    "garmentcode_to_definition",
    "definition_to_garmentcode",
    "PatternGenerator",
    "RefineDirective",
    "parse_feedback",
    "apply_directive",
    "rule_based_refine",
]
