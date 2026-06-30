#!/usr/bin/env python3
"""Module 10 (repurposed): preview how a pattern piece hangs -- no Blender needed.

Needs: nothing beyond the base install. Optionally `pip install -e ".[viz]"`
for the wireframe plot.
Run: python examples/02_drape_preview.py
"""

from __future__ import annotations

from parametric_cloth.preview import preview_drape
from parametric_cloth.simulation.tessellate import tessellate_piece
from parametric_cloth.templates import create_cape, create_skirt, create_tshirt


def main() -> None:
    cases = [
        ("cape", create_cape().pieces[0], "min_y"),
        ("skirt panel", create_skirt(panels=6).pieces[0], "min_y"),
        ("tshirt front", create_tshirt().piece("front"), "max_y"),
    ]

    print(f"{'piece':14}{'flat span (cm)':>16}{'drape drop (m)':>16}")
    draped_results = {}
    for name, piece, pin in cases:
        flat_span = max(v.y for v in piece.vertices) - min(v.y for v in piece.vertices)
        vertices = preview_drape(piece, pin=pin)
        drop = vertices[:, 1].max() - vertices[:, 1].min()
        draped_results[name] = (piece, vertices)
        print(f"{name:14}{flat_span:16.1f}{drop:16.3f}")

    try:
        from parametric_cloth import viz
        name, (piece, vertices) = next(iter(draped_results.items()))
        mesh = tessellate_piece(piece, levels=1)
        viz.plot_draped_wireframe(vertices, mesh.faces, title=f"{name} (draped preview)")
    except ModuleNotFoundError:
        print("\n(install matplotlib via `pip install -e \".[viz]\"` to see a wireframe plot)")


if __name__ == "__main__":
    main()
