import numpy as np
import pytest

from parametric_cloth.preview import preview_drape, preview_drape_all
from parametric_cloth.templates import create_cape, create_skirt, create_tshirt


def test_preview_drape_returns_finite_result():
    cape = create_cape()
    draped = preview_drape(cape.pieces[0])
    assert draped.shape[1] == 3
    assert np.all(np.isfinite(draped))


def test_pinned_edge_ends_up_at_the_top():
    cape = create_cape()
    draped = preview_drape(cape.pieces[0], pin="min_y")
    # Pin edge stays at world y=0 (by construction); everything else hangs below it.
    assert draped[:, 1].max() == pytest.approx(0.0, abs=1e-9)
    assert draped[:, 1].min() < 0.0


@pytest.mark.parametrize("factory,piece_name,pin", [
    (lambda: create_cape(), "cape", "min_y"),
    (lambda: create_skirt(panels=4), "panel_0", "min_y"),
    (lambda: create_tshirt(), "front", "max_y"),
])
def test_drop_is_bounded_not_exploding(factory, piece_name, pin):
    garment = factory()
    piece = garment.piece(piece_name)
    flat_span_cm = max(v.y for v in piece.vertices) - min(v.y for v in piece.vertices)
    draped = preview_drape(piece, pin=pin)
    drop_m = draped[:, 1].max() - draped[:, 1].min()
    # Sanity ceiling: a stable drape shouldn't stretch wildly past its own
    # flat length (this is exactly the failure mode of an under-damped or
    # too-soft mass-spring system -- catches a regression to bad defaults).
    assert 0.0 < drop_m < (flat_span_cm / 100.0) * 2.0


def test_preview_drape_all_covers_every_piece():
    shirt = create_tshirt()
    results = preview_drape_all(shirt, pin="max_y")
    assert set(results) == {"front", "back", "left_sleeve", "right_sleeve"}
    for vertices in results.values():
        assert np.all(np.isfinite(vertices))


def test_explicit_pin_mask_does_not_reorient():
    cape = create_cape()
    piece = cape.pieces[0]
    n = len(piece.vertices)
    # levels=0 keeps the simulated mesh exactly the polygon corners, so an
    # explicit per-corner pin mask lines up with it.
    # Pin everything -> nothing should move at all, regardless of orientation logic.
    draped = preview_drape(piece, pin=np.ones(n, dtype=bool), levels=0)
    flat = np.zeros((n, 3))
    flat[:, 0] = [v.x / 100.0 for v in piece.vertices]
    flat[:, 1] = [v.y / 100.0 for v in piece.vertices]
    assert np.allclose(draped, flat)
