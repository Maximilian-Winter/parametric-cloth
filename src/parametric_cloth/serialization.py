"""JSON serialization for the garment data model.

Provides a stable on-disk interchange format used to hand garments between
pipeline stages (e.g. ``create_garment.py`` -> Blender ``simulate.py``).

The format is explicit rather than relying on :func:`dataclasses.asdict`, so
that enums round-trip cleanly and the schema is versioned.
"""

from __future__ import annotations

import json
from typing import Any

from .fabric import FabricProperties, FabricType
from .pattern import (
    GarmentDefinition,
    PatternPiece,
    PatternVertex,
    PlacementHint,
    Seam,
    SeamEdge,
)

SCHEMA_VERSION = 1


# --- to dict ---------------------------------------------------------------

def _fabric_to_dict(f: FabricProperties) -> dict[str, Any]:
    return {
        "type": f.type.value,
        "mass_per_area": f.mass_per_area,
        "stiffness": f.stiffness,
        "bending": f.bending,
        "damping": f.damping,
        "friction": f.friction,
        "stretch_limit": f.stretch_limit,
    }


def _placement_to_dict(p: PlacementHint) -> dict[str, Any]:
    return {"anchor": p.anchor, "offset_normal": p.offset_normal, "rotation": p.rotation}


def _piece_to_dict(p: PatternPiece) -> dict[str, Any]:
    return {
        "name": p.name,
        "vertices": [{"x": v.x, "y": v.y} for v in p.vertices],
        "subdivisions": p.subdivisions,
        "placement": _placement_to_dict(p.placement) if p.placement else None,
        "fabric": _fabric_to_dict(p.fabric),
        "seam_allowance": p.seam_allowance,
    }


def _seam_edge_to_dict(e: SeamEdge) -> dict[str, Any]:
    return {
        "piece_name": e.piece_name,
        "vertex_start_index": e.vertex_start_index,
        "vertex_end_index": e.vertex_end_index,
    }


def _seam_to_dict(s: Seam) -> dict[str, Any]:
    return {
        "edge_a": _seam_edge_to_dict(s.edge_a),
        "edge_b": _seam_edge_to_dict(s.edge_b),
        "stiffness": s.stiffness,
    }


def garment_to_dict(g: GarmentDefinition) -> dict[str, Any]:
    """Convert a garment definition to a JSON-serializable dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "name": g.name,
        "pieces": [_piece_to_dict(p) for p in g.pieces],
        "seams": [_seam_to_dict(s) for s in g.seams],
        "simulation_frames": g.simulation_frames,
        "simulation_substeps": g.simulation_substeps,
        "gravity": g.gravity,
    }


# --- from dict -------------------------------------------------------------

def _fabric_from_dict(d: dict[str, Any]) -> FabricProperties:
    return FabricProperties(
        type=FabricType(d["type"]),
        mass_per_area=d["mass_per_area"],
        stiffness=d["stiffness"],
        bending=d["bending"],
        damping=d["damping"],
        friction=d["friction"],
        stretch_limit=d["stretch_limit"],
    )


def _placement_from_dict(d: dict[str, Any] | None) -> PlacementHint | None:
    if d is None:
        return None
    return PlacementHint(
        anchor=d["anchor"],
        offset_normal=d["offset_normal"],
        rotation=d.get("rotation", 0.0),
    )


def _piece_from_dict(d: dict[str, Any]) -> PatternPiece:
    return PatternPiece(
        name=d["name"],
        vertices=[PatternVertex(v["x"], v["y"]) for v in d["vertices"]],
        subdivisions=d.get("subdivisions", 10),
        placement=_placement_from_dict(d.get("placement")),
        fabric=_fabric_from_dict(d["fabric"]) if d.get("fabric") else FabricProperties(),
        seam_allowance=d.get("seam_allowance", 1.0),
    )


def _seam_edge_from_dict(d: dict[str, Any]) -> SeamEdge:
    return SeamEdge(
        piece_name=d["piece_name"],
        vertex_start_index=d["vertex_start_index"],
        vertex_end_index=d["vertex_end_index"],
    )


def _seam_from_dict(d: dict[str, Any]) -> Seam:
    return Seam(
        edge_a=_seam_edge_from_dict(d["edge_a"]),
        edge_b=_seam_edge_from_dict(d["edge_b"]),
        stiffness=d.get("stiffness", 1.0),
    )


def garment_from_dict(d: dict[str, Any]) -> GarmentDefinition:
    """Reconstruct a garment definition from a dict produced by ``garment_to_dict``."""
    version = d.get("schema_version", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"garment schema version {version} is newer than supported "
            f"version {SCHEMA_VERSION}"
        )
    return GarmentDefinition(
        name=d["name"],
        pieces=[_piece_from_dict(p) for p in d["pieces"]],
        seams=[_seam_from_dict(s) for s in d.get("seams", [])],
        simulation_frames=d.get("simulation_frames", 250),
        simulation_substeps=d.get("simulation_substeps", 15),
        gravity=d.get("gravity", -9.81),
    )


# --- JSON helpers ----------------------------------------------------------

def garment_to_json(g: GarmentDefinition, *, indent: int | None = 2) -> str:
    return json.dumps(garment_to_dict(g), indent=indent)


def garment_from_json(text: str) -> GarmentDefinition:
    return garment_from_dict(json.loads(text))


def save_garment(g: GarmentDefinition, path: str, *, indent: int | None = 2) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(garment_to_json(g, indent=indent))


def load_garment(path: str) -> GarmentDefinition:
    with open(path, encoding="utf-8") as fh:
        return garment_from_json(fh.read())
