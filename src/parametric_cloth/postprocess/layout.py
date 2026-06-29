"""Render a UV/pattern layout reference image as SVG (pure, dependency-free).

A designer paints textures onto this flat layout; because UVs come from the
sewing pattern, the result wraps correctly on the 3D garment. SVG keeps it
dependency-free and testable; the Blender step can additionally rasterize a PNG.
"""

from __future__ import annotations

import numpy as np

from ..simulation.assembly import AssembledGarment
from ..simulation.seams import boundary_loop
from .uv import AtlasLayout, pack_uv_atlas

_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def _panel_boundary_global(assembled: AssembledGarment, name: str) -> list[int]:
    lo, hi = assembled.piece_offsets[name]
    sel = [f for f in assembled.faces if lo <= f[0] < hi and lo <= f[1] < hi and lo <= f[2] < hi]
    if not sel:
        return []
    local = np.array(sel) - lo
    return [v + lo for v in boundary_loop(local)]


def render_uv_layout_svg(
    assembled: AssembledGarment,
    path: str,
    *,
    atlas: AtlasLayout | None = None,
    size: int = 1024,
) -> str:
    """Write an SVG of the packed UV layout (one filled outline per panel)."""
    atlas = atlas or pack_uv_atlas(assembled)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#1a1a1a"/>',
    ]
    for i, name in enumerate(assembled.piece_offsets):
        loop = _panel_boundary_global(assembled, name)
        if not loop:
            continue
        # UV origin is bottom-left; SVG y grows downward, so flip v.
        pts = " ".join(
            f"{atlas.uv[v, 0] * size:.2f},{(1.0 - atlas.uv[v, 1]) * size:.2f}"
            for v in loop
        )
        color = _PALETTE[i % len(_PALETTE)]
        parts.append(
            f'<polygon points="{pts}" fill="{color}" fill-opacity="0.35" '
            f'stroke="{color}" stroke-width="2"/>'
        )
    parts.append("</svg>")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return path
