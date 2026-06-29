import math

import pytest

from parametric_cloth.fabric import FabricType
from parametric_cloth.pattern import PatternPiece, PatternVertex
from parametric_cloth.templates import create_cape, create_skirt, create_tshirt


# --- skirt -----------------------------------------------------------------

@pytest.mark.parametrize("panels", [2, 4, 6, 8])
def test_skirt_has_one_seam_and_piece_per_panel(panels):
    skirt = create_skirt(panels=panels)
    assert len(skirt.pieces) == panels
    assert len(skirt.seams) == panels  # wraps around -> one seam per panel
    assert skirt.is_valid(), skirt.validate()


def test_skirt_rejects_too_few_panels():
    with pytest.raises(ValueError):
        create_skirt(panels=1)


def test_skirt_flare_widens_the_hem():
    narrow = create_skirt(panels=4, flare=1.0)
    wide = create_skirt(panels=4, flare=2.0)
    assert wide.pieces[0].area() > narrow.pieces[0].area()


def test_skirt_seams_reference_existing_pieces():
    skirt = create_skirt(panels=4)
    names = {p.name for p in skirt.pieces}
    for seam in skirt.seams:
        assert seam.edge_a.piece_name in names
        assert seam.edge_b.piece_name in names


# --- t-shirt ---------------------------------------------------------------

def test_tshirt_structure():
    shirt = create_tshirt()
    names = {p.name for p in shirt.pieces}
    assert names == {"front", "back", "left_sleeve", "right_sleeve"}
    assert shirt.is_valid(), shirt.validate()


def test_tshirt_ease_increases_width():
    slim = create_tshirt(ease=1.0)
    loose = create_tshirt(ease=1.4)
    assert loose.piece("front").area() > slim.piece("front").area()


def test_tshirt_fabric_propagates_to_pieces():
    shirt = create_tshirt(fabric=FabricType.DENIM)
    for piece in shirt.pieces:
        assert piece.fabric.type is FabricType.DENIM


# --- cape ------------------------------------------------------------------

def test_cape_is_single_panel_no_seams():
    cape = create_cape()
    assert len(cape.pieces) == 1
    assert cape.seams == []
    assert cape.is_valid(), cape.validate()


# --- geometry sanity -------------------------------------------------------

def test_degenerate_piece_is_invalid():
    piece = PatternPiece(
        name="line",
        vertices=[PatternVertex(0, 0), PatternVertex(1, 0), PatternVertex(2, 0)],
    )
    assert piece.area() == pytest.approx(0.0)
    assert piece.validate()


def test_square_area_is_correct():
    square = PatternPiece(
        name="square",
        vertices=[
            PatternVertex(0, 0), PatternVertex(10, 0),
            PatternVertex(10, 10), PatternVertex(0, 10),
        ],
    )
    assert square.area() == pytest.approx(100.0)
    assert not square.validate()


def test_self_intersecting_piece_flagged():
    # A "bowtie" quad whose edges cross.
    piece = PatternPiece(
        name="bowtie",
        vertices=[
            PatternVertex(0, 0), PatternVertex(10, 10),
            PatternVertex(10, 0), PatternVertex(0, 10),
        ],
    )
    assert any("self-intersecting" in i for i in piece.validate())


def test_tshirt_panels_are_simple_polygons():
    shirt = create_tshirt()
    for piece in shirt.pieces:
        assert not piece._is_self_intersecting(), piece.name


def test_coincident_vertices_flagged():
    piece = PatternPiece(
        name="dupe",
        vertices=[
            PatternVertex(0, 0), PatternVertex(0, 0),
            PatternVertex(10, 0), PatternVertex(10, 10),
        ],
    )
    assert any("coincident" in i for i in piece.validate())
