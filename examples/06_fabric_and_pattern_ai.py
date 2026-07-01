#!/usr/bin/env python3
"""Modules 8/9: fuzzy fabric matching and rule-based pattern refinement.

Needs: nothing beyond the base install -- both use only the Python stdlib.
Run: python examples/06_fabric_and_pattern_ai.py
"""

from __future__ import annotations

from parametric_cloth.ai_pattern import parse_feedback, rule_based_refine
from parametric_cloth.fabric import FabricProperties
from parametric_cloth.fabric_ai import find_closest_preset
from parametric_cloth.templates import create_tshirt


def main() -> None:
    print("Module 9: fuzzy fabric matching (no ML dependency)")
    descriptions = [
        "heavy brushed cotton twill",
        "lightweight silk charmeuse",
        "raw denim",
        "medieval chain mail",
        "workwear",
    ]
    for description in descriptions:
        match = find_closest_preset(description)
        props = FabricProperties.from_description(description)
        print(f"  {description!r:32} -> {match:18} (mass={props.mass_per_area:.2f} kg/m^2)")

    print("\nModule 8: rule-based pattern refinement from text feedback")
    shirt = create_tshirt()
    feedback = "Make the sleeves wider and shorten the hem by 5cm"
    directives = parse_feedback(feedback)
    print(f"  feedback: {feedback!r}")
    print(f"  parsed directives: {[(d.action, d.target, d.amount_cm) for d in directives]}")

    refined = rule_based_refine(shirt, feedback)
    print(f"  sleeve area: {shirt.piece('left_sleeve').area():.1f} -> "
          f"{refined.piece('left_sleeve').area():.1f} cm^2")
    print(f"  front area:  {shirt.piece('front').area():.1f} -> "
          f"{refined.piece('front').area():.1f} cm^2")
    print(f"  refined garment valid: {refined.is_valid()}")


if __name__ == "__main__":
    main()
