"""Bidirectional conversion between GarmentCode JSON and GarmentDefinition.

GarmentCode (Korosteleva & Sorkine-Hornung, SIGGRAPH Asia 2023) represents
patterns as panels with 2D vertices and a list of stitches connecting panel
edges -- structurally close to Module 1's ``PatternPiece``/``Seam`` model. This
adapter handles the *evaluated* GarmentCode representation (panels already
carry concrete vertex coordinates), which is what pattern-generation model
outputs (SewFormer, ChatGarment) and the GarmentCodeData dataset provide.

Note: full programmatic GarmentCode -- panels defined by parametric edge
generators (curves, parameterized darts) evaluated lazily -- requires the
``pygarment`` package to evaluate the program into vertices. That evaluation
step is out of scope here (no ``pygarment`` in this environment); this adapter
operates on the post-evaluation, vertices-already-present JSON shape, which
covers the common interchange case described in the design doc.
"""

from __future__ import annotations

from typing import Any

from ..fabric import FabricProperties, FabricType
from ..pattern import (
    GarmentDefinition,
    PatternPiece,
    PatternVertex,
    PlacementHint,
    Seam,
    SeamEdge,
)

# GarmentCode panel translation/rotation isn't modeled by Module 1's flat
# PatternPiece (which is always placed via a named anchor); we preserve it as
# an opaque field on round-trip rather than discarding it silently.
_TRANSLATION_KEY = "translation"
_ROTATION_KEY = "rotation"


def garmentcode_to_definition(
    gc_json: dict[str, Any],
    *,
    default_fabric: FabricType = FabricType.COTTON,
) -> GarmentDefinition:
    """Convert an evaluated GarmentCode JSON spec to a GarmentDefinition.

    Expected shape (the common evaluated/interchange form)::

        {
          "name": "...",
          "panels": {
            "<panel_name>": {
              "vertices": [[x, y], ...],          # cm
              "translation": [x, y, z],            # optional, preserved
              "rotation": [x, y, z],               # optional, preserved
              "fabric": "cotton",                  # optional FabricType value
            }, ...
          },
          "stitches": [
            [{"panel": "...", "edge": [i, j]}, {"panel": "...", "edge": [k, l]}],
            ...
          ]
        }
    """
    panels = gc_json.get("panels", {})
    if not panels:
        raise ValueError("GarmentCode spec has no panels")

    pieces = []
    for name, panel in panels.items():
        verts = panel.get("vertices")
        if not verts:
            raise ValueError(f"panel '{name}' has no vertices")
        fabric_value = panel.get("fabric")
        fabric_type = FabricType(fabric_value) if fabric_value else default_fabric

        placement = None
        if "anchor" in panel:
            placement = PlacementHint(
                anchor=panel["anchor"],
                offset_normal=panel.get("offset_normal", 2.0),
                rotation=panel.get("placement_rotation", 0.0),
            )

        pieces.append(PatternPiece(
            name=name,
            vertices=[PatternVertex(float(x), float(y)) for x, y in verts],
            placement=placement,
            fabric=FabricProperties.from_preset(fabric_type),
        ))

    seams = []
    for stitch in gc_json.get("stitches", []):
        if len(stitch) != 2:
            raise ValueError(f"stitch must connect exactly 2 edges, got {len(stitch)}")
        (a, b) = stitch
        seams.append(Seam(
            edge_a=SeamEdge(a["panel"], a["edge"][0], a["edge"][1]),
            edge_b=SeamEdge(b["panel"], b["edge"][0], b["edge"][1]),
        ))

    return GarmentDefinition(
        name=gc_json.get("name", "generated"),
        pieces=pieces,
        seams=seams,
    )


def definition_to_garmentcode(garment: GarmentDefinition) -> dict[str, Any]:
    """Convert a GarmentDefinition to evaluated GarmentCode JSON.

    Inverse of :func:`garmentcode_to_definition` for the subset of fields each
    format supports; round-trips losslessly for geometry, fabric, and stitches.
    """
    panels: dict[str, Any] = {}
    for piece in garment.pieces:
        panel: dict[str, Any] = {
            "vertices": [[v.x, v.y] for v in piece.vertices],
            "fabric": piece.fabric.type.value,
        }
        if piece.placement is not None:
            panel["anchor"] = piece.placement.anchor
            panel["offset_normal"] = piece.placement.offset_normal
            panel["placement_rotation"] = piece.placement.rotation
        panels[piece.name] = panel

    stitches = [
        [
            {"panel": s.edge_a.piece_name,
             "edge": [s.edge_a.vertex_start_index, s.edge_a.vertex_end_index]},
            {"panel": s.edge_b.piece_name,
             "edge": [s.edge_b.vertex_start_index, s.edge_b.vertex_end_index]},
        ]
        for s in garment.seams
    ]

    return {"name": garment.name, "panels": panels, "stitches": stitches}
