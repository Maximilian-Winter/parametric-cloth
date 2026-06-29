"""Standardized per-garment export package: metadata + UV layout.

Mirrors the design's package layout::

    garments/<name>/
      mesh.fbx          # geometry with bone weights   (written by Blender step)
      normal.png        # baked wrinkle detail          (written by Blender step)
      uv_layout.svg     # reference image of the UV/pattern layout
      metadata.json     # pattern parameters, fabric, poly count

This module writes the dependency-free artifacts (``metadata.json``,
``uv_layout.svg``). The mesh and normal map are produced by the Blender
post-process (:mod:`parametric_cloth.postprocess.blender_post`).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

from ..pattern import GarmentDefinition
from ..simulation.assembly import AssembledGarment
from .layout import render_uv_layout_svg
from .uv import AtlasLayout, pack_uv_atlas

MESH_FILENAME = "mesh.fbx"
NORMAL_FILENAME = "normal.png"
UV_LAYOUT_FILENAME = "uv_layout.svg"
METADATA_FILENAME = "metadata.json"


@dataclass
class GarmentPackage:
    """Paths that make up an exported garment package."""

    directory: str
    metadata_path: str
    uv_layout_path: str
    mesh_filename: str = MESH_FILENAME
    normal_filename: str = NORMAL_FILENAME


def build_metadata(
    garment: GarmentDefinition,
    assembled: AssembledGarment,
    *,
    atlas: Optional[AtlasLayout] = None,
    parameters: Optional[dict[str, Any]] = None,
    poly_count: Optional[int] = None,
) -> dict[str, Any]:
    """Assemble the metadata dict describing an exported garment."""
    n_faces = int(len(assembled.faces))
    fabrics = sorted({p.fabric.type.value for p in garment.pieces})
    meta: dict[str, Any] = {
        "name": garment.name,
        "n_pieces": len(garment.pieces),
        "n_seams": len(garment.seams),
        "n_vertices": assembled.n_vertices,
        "n_faces": n_faces,
        "poly_count": int(poly_count) if poly_count is not None else n_faces,
        "fabrics": fabrics,
        "parameters": parameters or {},
        "files": {
            "mesh": MESH_FILENAME,
            "normal": NORMAL_FILENAME,
            "uv_layout": UV_LAYOUT_FILENAME,
            "metadata": METADATA_FILENAME,
        },
    }
    if atlas is not None:
        meta["uv_atlas"] = {"cols": atlas.cols, "rows": atlas.rows, "margin": atlas.margin}
    return meta


def write_package(
    out_dir: str,
    garment: GarmentDefinition,
    assembled: AssembledGarment,
    *,
    atlas: Optional[AtlasLayout] = None,
    parameters: Optional[dict[str, Any]] = None,
    poly_count: Optional[int] = None,
) -> GarmentPackage:
    """Write the dependency-free package artifacts and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    atlas = atlas or pack_uv_atlas(assembled)

    uv_layout_path = os.path.join(out_dir, UV_LAYOUT_FILENAME)
    render_uv_layout_svg(assembled, uv_layout_path, atlas=atlas)

    metadata = build_metadata(
        garment, assembled, atlas=atlas, parameters=parameters, poly_count=poly_count
    )
    metadata_path = os.path.join(out_dir, METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return GarmentPackage(
        directory=out_dir,
        metadata_path=metadata_path,
        uv_layout_path=uv_layout_path,
    )
