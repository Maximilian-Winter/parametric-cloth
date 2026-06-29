"""Module 4: Post-Processing & Export.

UV mapping from patterns, normal baking, bone-weight transfer, decimation, and a
standardized game-ready package. The UV/atlas/layout/metadata logic is pure numpy
and unit-tested; the Blender steps live in
:mod:`parametric_cloth.postprocess.blender_post` behind a lazy ``bpy`` import.
"""

from __future__ import annotations

from .layout import render_uv_layout_svg
from .metadata import (
    GarmentPackage,
    build_metadata,
    write_package,
)
from .uv import AtlasLayout, pack_uv_atlas

__all__ = [
    "pack_uv_atlas",
    "AtlasLayout",
    "render_uv_layout_svg",
    "build_metadata",
    "write_package",
    "GarmentPackage",
]
