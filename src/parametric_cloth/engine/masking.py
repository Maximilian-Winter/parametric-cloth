"""Body masking and garment layering.

The standard shipped-game approach (no inter-garment physics): each garment
covers a set of body regions. Equipping it hides the body faces underneath; when
two garments overlap a region, only the outer (higher-layer) one shows there.

Pure numpy / set logic -- fully testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GarmentCoverage:
    """Which body regions a garment covers, and its layering order."""

    garment_id: str
    regions: set[str]
    layer: int = 0                    # higher = outer


def visible_body_faces(face_regions: np.ndarray, covered_regions: set[str]) -> np.ndarray:
    """Boolean mask (per body face) of faces NOT hidden by any garment.

    Args:
        face_regions: region name per body face, shape (F,).
        covered_regions: regions covered by at least one equipped garment.
    """
    face_regions = np.asarray(face_regions)
    return np.array([r not in covered_regions for r in face_regions], dtype=bool)


def covered_regions(coverages: list[GarmentCoverage]) -> set[str]:
    """Union of all regions covered by the equipped garments."""
    out: set[str] = set()
    for c in coverages:
        out |= c.regions
    return out


def resolve_visible_regions(coverages: list[GarmentCoverage]) -> dict[str, set[str]]:
    """For each garment, the regions where it is the outermost layer (so visible).

    In a region covered by several garments, only the highest ``layer`` shows;
    ties favor all tied garments (e.g. non-overlapping panels at the same layer).
    """
    top_layer: dict[str, int] = {}
    for c in coverages:
        for r in c.regions:
            top_layer[r] = max(top_layer.get(r, c.layer), c.layer)

    visible: dict[str, set[str]] = {}
    for c in coverages:
        visible[c.garment_id] = {r for r in c.regions if c.layer >= top_layer[r]}
    return visible


def hidden_garment_regions(coverages: list[GarmentCoverage]) -> dict[str, set[str]]:
    """Inverse of :func:`resolve_visible_regions`: regions hidden under an outer layer."""
    visible = resolve_visible_regions(coverages)
    return {c.garment_id: (c.regions - visible[c.garment_id]) for c in coverages}
