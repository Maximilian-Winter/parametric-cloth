#!/usr/bin/env python3
"""The actual promise of this framework, runnable with just numpy:
sewing pattern -> placed on a body -> seams welded -> draped under gravity.

Needs: nothing beyond the base install for the simulation; `pip install -e
".[viz]"` to render the PNGs.
Run: python examples/07_garment_on_body.py
"""

from __future__ import annotations

from parametric_cloth.avatar.synthetic import make_simple_body
from parametric_cloth.preview import preview_drape_garment_on_body
from parametric_cloth.templates import create_skirt, create_tshirt


def main() -> None:
    body = make_simple_body()

    skirt = create_skirt(panels=8, waist_half=18, hip_half=24, length=45, flare=1.5)
    skirt_result = preview_drape_garment_on_body(skirt, body, pin="min_y", n_steps=150)

    shirt = create_tshirt()
    shirt_result = preview_drape_garment_on_body(shirt, body, pin="max_y", n_steps=150)

    for name, result in [("skirt", skirt_result), ("tshirt", shirt_result)]:
        print(f"{name:8} vertices={result.vertices.shape[0]:4d} faces={result.faces.shape[0]:4d}")

    try:
        from parametric_cloth import viz
        viz.plot_garment_on_body(skirt_result, title="skirt on body", show=False).figure.savefig(
            "skirt_on_body.png", dpi=150)
        viz.plot_garment_on_body(shirt_result, title="tshirt on body", show=False).figure.savefig(
            "tshirt_on_body.png", dpi=150)
        print("\nwrote skirt_on_body.png and tshirt_on_body.png")
    except ModuleNotFoundError:
        print("\n(install matplotlib via `pip install -e \".[viz]\"` to render images)")


if __name__ == "__main__":
    main()
