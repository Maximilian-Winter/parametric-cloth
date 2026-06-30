#!/usr/bin/env python3
"""Module 1: create garments, validate them, and round-trip through JSON.

Needs: nothing beyond the base install (`pip install -e .`).
Run: python examples/01_pattern_basics.py
"""

from __future__ import annotations

from parametric_cloth import (
    FabricType,
    TShirtCustomization,
    create_cape,
    create_skirt,
    create_tshirt,
    garment_from_json,
    garment_to_json,
)


def main() -> None:
    skirt = create_skirt(panels=6, flare=1.8, fabric=FabricType.SILK)
    shirt = TShirtCustomization(fit="oversized", sleeve_length="long", neckline="v_neck").build()
    cape = create_cape(neck_half=20.0, length=95.0, fabric=FabricType.WOOL)

    for garment in (skirt, shirt, cape):
        issues = garment.validate()
        total_area = sum(p.area() for p in garment.pieces)
        print(f"{garment.name:16} pieces={len(garment.pieces):2} seams={len(garment.seams):2} "
              f"area={total_area:7.1f}cm^2  valid={not issues}")
        if issues:
            for issue in issues:
                print(f"    - {issue}")

    # JSON is the interchange format the rest of the pipeline consumes.
    text = garment_to_json(skirt)
    restored = garment_from_json(text)
    assert restored.name == skirt.name and len(restored.pieces) == len(skirt.pieces)
    print(f"\nround-tripped '{skirt.name}' through JSON ({len(text)} bytes) successfully")


if __name__ == "__main__":
    main()
