"""Core data model: garments described as parametric 2D geometry.

A garment is a set of flat ``PatternPiece`` polygons (in centimeters) plus a
list of ``Seam`` connections describing how their edges are sewn together.
Everything downstream in the pipeline consumes this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .fabric import FabricProperties


def _orient(p, q, r) -> float:
    """Signed area of triangle pqr; sign gives turn direction."""
    return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)


def _segments_properly_cross(p1, p2, p3, p4, *, eps: float = 1e-12) -> bool:
    """True if segment p1p2 and p3p4 cross at an interior point of both."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    return (
        ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps))
        and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps))
    )


@dataclass
class PatternVertex:
    """A point on a 2D pattern piece, in centimeters."""

    x: float
    y: float


@dataclass
class SeamEdge:
    """One side of a seam, referencing a range of vertices on a pattern piece.

    The edge spans the polygon vertices from ``vertex_start_index`` to
    ``vertex_end_index`` inclusive (following polygon winding order).
    """

    piece_name: str
    vertex_start_index: int
    vertex_end_index: int


@dataclass
class Seam:
    """A connection between two edges on pattern pieces.

    During simulation these edges are pulled together like sewing.
    """

    edge_a: SeamEdge
    edge_b: SeamEdge
    stiffness: float = 1.0


@dataclass
class PlacementHint:
    """Where to position a pattern piece relative to the avatar.

    Uses body landmark names resolved by the avatar system (Module 2).
    """

    anchor: str               # e.g. "chest_front", "left_upper_arm", "waist"
    offset_normal: float      # distance from body surface in cm
    rotation: float = 0.0     # rotation around the surface normal in degrees


@dataclass
class PatternPiece:
    """A single 2D pattern piece -- one component of a garment."""

    name: str
    vertices: List[PatternVertex]
    subdivisions: int = 10
    placement: Optional[PlacementHint] = None
    fabric: FabricProperties = field(default_factory=FabricProperties)
    seam_allowance: float = 1.0       # cm, excluded from simulation

    def signed_area(self) -> float:
        """Signed polygon area in cm^2 via the shoelace formula.

        Positive for counter-clockwise winding, negative for clockwise.
        """
        verts = self.vertices
        n = len(verts)
        acc = 0.0
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            acc += a.x * b.y - b.x * a.y
        return acc / 2.0

    def area(self) -> float:
        """Absolute polygon area in cm^2."""
        return abs(self.signed_area())

    def validate(self) -> list[str]:
        """Return a list of geometry problems with this piece, empty if valid."""
        issues: list[str] = []
        n = len(self.vertices)
        if n < 3:
            issues.append(f"piece '{self.name}' has {n} vertices, needs at least 3")
            return issues

        if self.area() <= 1e-9:
            issues.append(f"piece '{self.name}' has degenerate (zero) area")

        # Flag coincident consecutive vertices, which create zero-length edges.
        for i in range(n):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % n]
            if abs(a.x - b.x) < 1e-9 and abs(a.y - b.y) < 1e-9:
                issues.append(
                    f"piece '{self.name}' has coincident vertices at index {i}"
                )

        if self.subdivisions < 0:
            issues.append(f"piece '{self.name}' has negative subdivisions")

        if self._is_self_intersecting():
            issues.append(
                f"piece '{self.name}' is self-intersecting (edges cross); "
                f"check vertex ordering"
            )

        issues.extend(self.fabric.validate())
        return issues

    def _is_self_intersecting(self) -> bool:
        """True if any pair of non-adjacent polygon edges properly cross."""
        verts = self.vertices
        n = len(verts)
        for i in range(n):
            a1, a2 = verts[i], verts[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i or j == (i + 1) % n or (j + 1) % n == i:
                    continue  # adjacent edges share an endpoint
                b1, b2 = verts[j], verts[(j + 1) % n]
                if _segments_properly_cross(a1, a2, b1, b2):
                    return True
        return False


@dataclass
class GarmentDefinition:
    """Complete definition of a garment -- patterns, construction, simulation."""

    name: str
    pieces: List[PatternPiece]
    seams: List[Seam]
    simulation_frames: int = 250
    simulation_substeps: int = 15
    gravity: float = -9.81

    def piece(self, name: str) -> Optional[PatternPiece]:
        """Return the piece with the given name, or None."""
        for p in self.pieces:
            if p.name == name:
                return p
        return None

    def validate(self) -> list[str]:
        """Return all geometry and construction problems, empty if valid."""
        issues: list[str] = []

        names = [p.name for p in self.pieces]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            issues.append(f"duplicate piece names: {', '.join(dupes)}")

        for p in self.pieces:
            issues.extend(p.validate())

        for i, seam in enumerate(self.seams):
            issues.extend(self._validate_seam(i, seam))

        if self.simulation_frames <= 0:
            issues.append("simulation_frames must be > 0")
        if self.simulation_substeps <= 0:
            issues.append("simulation_substeps must be > 0")

        return issues

    def _validate_seam(self, index: int, seam: Seam) -> list[str]:
        issues: list[str] = []
        for label, edge in (("edge_a", seam.edge_a), ("edge_b", seam.edge_b)):
            piece = self.piece(edge.piece_name)
            if piece is None:
                issues.append(
                    f"seam {index} {label} references unknown piece "
                    f"'{edge.piece_name}'"
                )
                continue
            n = len(piece.vertices)
            for attr in ("vertex_start_index", "vertex_end_index"):
                idx = getattr(edge, attr)
                if not 0 <= idx < n:
                    issues.append(
                        f"seam {index} {label} {attr}={idx} out of range "
                        f"for piece '{edge.piece_name}' ({n} vertices)"
                    )
        if seam.stiffness <= 0:
            issues.append(f"seam {index} stiffness must be > 0")
        return issues

    def is_valid(self) -> bool:
        return not self.validate()
