"""Parametric garment template functions.

Each function takes measurements and returns a :class:`GarmentDefinition`.
This is where the parametric power lives -- the same function produces
different garments for different parameters.
"""

from __future__ import annotations

from .fabric import FabricProperties, FabricType
from .pattern import (
    GarmentDefinition,
    PatternPiece,
    PatternVertex,
    PlacementHint,
    Seam,
    SeamEdge,
)


def create_skirt(
    waist_half: float = 18.0,       # cm, half the waist circumference
    hip_half: float = 24.0,         # cm, half the hip circumference
    length: float = 50.0,           # cm, waist to hem
    panels: int = 4,                # number of panels (2, 4, 6, 8)
    flare: float = 1.3,             # hem-to-hip ratio (1.0 = pencil, 2.0 = full)
    fabric: FabricType = FabricType.COTTON,
) -> GarmentDefinition:
    """Generate a parametric skirt from panel count and measurements."""
    if panels < 2:
        raise ValueError(f"a skirt needs at least 2 panels (got {panels})")

    panel_waist = waist_half * 2 / panels
    panel_hip = hip_half * 2 / panels
    panel_hem = panel_hip * flare
    hip_drop = 20.0  # cm from waist to hip line

    pieces = []
    seams = []

    for i in range(panels):
        piece = PatternPiece(
            name=f"panel_{i}",
            vertices=[
                PatternVertex(0, 0),                          # 0: waist left
                PatternVertex(panel_waist, 0),                # 1: waist right
                PatternVertex(panel_hip + (panel_hem - panel_hip) * 0.5, hip_drop),  # 2: hip right
                PatternVertex(panel_hem, length),             # 3: hem right
                PatternVertex(0 - (panel_hem - panel_hip) * 0.5 + (panel_hem - panel_waist) * 0.5, length),  # 4: hem left
            ],
            placement=PlacementHint(
                anchor=f"waist_segment_{i}",
                offset_normal=2.0,
            ),
            fabric=FabricProperties.from_preset(fabric),
            subdivisions=12,
        )
        pieces.append(piece)

        # Connect each panel to the next (wrapping around).
        next_i = (i + 1) % panels
        seams.append(Seam(
            edge_a=SeamEdge(f"panel_{i}", 1, 2),       # right edge of this panel
            edge_b=SeamEdge(f"panel_{next_i}", 0, 4),  # left edge of next panel
        ))

    return GarmentDefinition(
        name=f"skirt_{panels}panel",
        pieces=pieces,
        seams=seams,
    )


def create_tshirt(
    chest_half_width: float = 25.0,
    length: float = 65.0,
    shoulder_width: float = 20.0,
    neck_width: float = 8.0,
    neck_depth_front: float = 6.0,
    neck_depth_back: float = 2.0,
    sleeve_length: float = 20.0,
    sleeve_upper_width: float = 18.0,
    sleeve_lower_width: float = 16.0,
    ease: float = 1.1,
    fabric: FabricType = FabricType.COTTON,
) -> GarmentDefinition:
    """Generate a parametric T-shirt."""
    w = chest_half_width * ease
    fab = FabricProperties.from_preset(fabric)

    # Vertices wind around the boundary so the polygon stays simple (no
    # self-intersection): bottom hem -> right side -> right shoulder -> neckline
    # (right to left) -> left shoulder -> left side.
    front = PatternPiece(
        name="front",
        vertices=[
            PatternVertex(0, 0),                                              # 0 hem left
            PatternVertex(w, 0),                                             # 1 hem right
            PatternVertex(w, length),                                        # 2 top right
            PatternVertex(shoulder_width, length),                          # 3 right shoulder
            PatternVertex(w / 2 + neck_width / 2, length - neck_depth_front * 0.3),  # 4 neck right
            PatternVertex(w / 2, length - neck_depth_front),                 # 5 neck bottom
            PatternVertex(w / 2 - neck_width / 2, length - neck_depth_front * 0.3),  # 6 neck left
            PatternVertex(w - shoulder_width, length),                      # 7 left shoulder
            PatternVertex(0, length),                                        # 8 top left
        ],
        placement=PlacementHint(anchor="chest_front", offset_normal=3.0),
        fabric=fab,
        subdivisions=12,
    )

    back = PatternPiece(
        name="back",
        vertices=[
            PatternVertex(0, 0),                                             # 0 hem left
            PatternVertex(w, 0),                                            # 1 hem right
            PatternVertex(w, length),                                       # 2 top right
            PatternVertex(shoulder_width, length),                         # 3 right shoulder
            PatternVertex(w / 2 + neck_width / 2, length - neck_depth_back * 0.3),  # 4 neck right
            PatternVertex(w / 2, length - neck_depth_back),                 # 5 neck bottom
            PatternVertex(w / 2 - neck_width / 2, length - neck_depth_back * 0.3),  # 6 neck left
            PatternVertex(w - shoulder_width, length),                     # 7 left shoulder
            PatternVertex(0, length),                                       # 8 top left
        ],
        placement=PlacementHint(anchor="chest_back", offset_normal=3.0),
        fabric=fab,
        subdivisions=12,
    )

    left_sleeve = PatternPiece(
        name="left_sleeve",
        vertices=[
            PatternVertex(0, 0),
            PatternVertex(sleeve_lower_width, 0),
            PatternVertex(sleeve_upper_width, sleeve_length),
            PatternVertex(0, sleeve_length),
        ],
        placement=PlacementHint(anchor="left_upper_arm", offset_normal=2.0),
        fabric=fab,
        subdivisions=8,
    )

    right_sleeve = PatternPiece(
        name="right_sleeve",
        vertices=[
            PatternVertex(0, 0),
            PatternVertex(sleeve_lower_width, 0),
            PatternVertex(sleeve_upper_width, sleeve_length),
            PatternVertex(0, sleeve_length),
        ],
        placement=PlacementHint(anchor="right_upper_arm", offset_normal=2.0),
        fabric=fab,
        subdivisions=8,
    )

    seams = [
        # Body: two side seams and two shoulder seams join front to back.
        Seam(edge_a=SeamEdge("front", 0, 8), edge_b=SeamEdge("back", 0, 8)),  # left side
        Seam(edge_a=SeamEdge("front", 1, 2), edge_b=SeamEdge("back", 1, 2)),  # right side
        Seam(edge_a=SeamEdge("front", 2, 3), edge_b=SeamEdge("back", 2, 3)),  # right shoulder
        Seam(edge_a=SeamEdge("front", 7, 8), edge_b=SeamEdge("back", 7, 8)),  # left shoulder
        # Sleeve caps attach at the shoulders. The simplified rectangular sleeve
        # has no dedicated armhole edge, so this attachment is approximate.
        Seam(edge_a=SeamEdge("left_sleeve", 2, 3), edge_b=SeamEdge("front", 7, 8)),
        Seam(edge_a=SeamEdge("right_sleeve", 2, 3), edge_b=SeamEdge("back", 2, 3)),
    ]

    return GarmentDefinition(
        name="tshirt",
        pieces=[front, back, left_sleeve, right_sleeve],
        seams=seams,
    )


def create_cape(
    neck_half: float = 18.0,    # cm, half the neck/shoulder span at the top
    length: float = 90.0,       # cm, neck to hem
    flare: float = 2.5,         # hem-to-neck width ratio
    fabric: FabricType = FabricType.WOOL,
) -> GarmentDefinition:
    """Generate a single-panel cape -- the simplest garment.

    A trapezoidal panel that hangs from the shoulders; gravity does the work.
    """
    top = neck_half * 2
    hem = top * flare
    overhang = (hem - top) / 2

    panel = PatternPiece(
        name="cape",
        vertices=[
            PatternVertex(0, 0),            # 0: neck left
            PatternVertex(top, 0),          # 1: neck right
            PatternVertex(top + overhang, length),   # 2: hem right
            PatternVertex(-overhang, length),        # 3: hem left
        ],
        placement=PlacementHint(anchor="chest_back", offset_normal=3.0),
        fabric=FabricProperties.from_preset(fabric),
        subdivisions=16,
    )

    return GarmentDefinition(name="cape", pieces=[panel], seams=[])
