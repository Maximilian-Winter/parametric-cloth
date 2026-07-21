import numpy as np
import pytest

from parametric_cloth.avatar.synthetic import make_simple_body
from parametric_cloth.preview import preview_drape_garment_on_body
from parametric_cloth.templates import create_cape, create_skirt, create_tshirt


def test_skirt_wraps_the_waist_and_hangs_below_it():
    body = make_simple_body()
    skirt = create_skirt(panels=8, waist_half=18, hip_half=24, length=45, flare=1.5)
    result = preview_drape_garment_on_body(skirt, body, pin="min_y", n_steps=150)

    assert np.all(np.isfinite(result.vertices))
    # Hangs below the waist, doesn't fly up past it.
    assert result.vertices[:, 1].max() <= body.hip_height + 0.05
    assert result.vertices[:, 1].min() < body.hip_height - 0.1


def test_skirt_wraps_full_circumference():
    body = make_simple_body()
    skirt = create_skirt(panels=8, waist_half=18, hip_half=24, length=45, flare=1.5)
    result = preview_drape_garment_on_body(skirt, body, pin="min_y", n_steps=150)

    angles = np.degrees(np.arctan2(result.vertices[:, 2], result.vertices[:, 0])) % 360
    hist, _ = np.histogram(angles, bins=8, range=(0, 360))
    assert np.all(hist > 0)          # every angular sector has coverage, no gap


def test_collision_keeps_vertices_outside_the_body():
    body = make_simple_body()
    skirt = create_skirt(panels=6, waist_half=18, hip_half=24, length=45, flare=1.3)
    result = preview_drape_garment_on_body(skirt, body, pin="min_y", n_steps=150, collide=True)

    radial = np.linalg.norm(result.vertices[:, [0, 2]], axis=1)
    limit = body.radius_at_height(result.vertices[:, 1])
    assert np.all(radial >= limit - 1e-6)


def test_collision_disabled_can_penetrate():
    body = make_simple_body()
    skirt = create_skirt(panels=6, waist_half=10, hip_half=10, length=45, flare=0.5)
    result = preview_drape_garment_on_body(skirt, body, pin="min_y", n_steps=150, collide=False)
    radial = np.linalg.norm(result.vertices[:, [0, 2]], axis=1)
    limit = body.radius_at_height(result.vertices[:, 1])
    # A narrow, non-flared skirt with collision off should sink inside somewhere.
    assert np.any(radial < limit - 0.005)


def test_tshirt_shoulder_lands_near_the_chest_anchor_not_far_above_it():
    body = make_simple_body()
    shirt = create_tshirt()
    result = preview_drape_garment_on_body(shirt, body, pin="max_y", n_steps=0, collide=False)
    anchor_y = body.anchors["chest_front"][0][1]
    # Before the fix, the panel's top ended up anchor_y + full panel length above
    # the anchor; after it, the pinned edge should land at/near the anchor itself.
    assert result.vertices[:, 1].max() < anchor_y + 0.1


def test_tshirt_produces_finite_result_for_all_panels():
    body = make_simple_body()
    shirt = create_tshirt()
    result = preview_drape_garment_on_body(shirt, body, pin="max_y", n_steps=100)
    assert np.all(np.isfinite(result.vertices))
    assert set(np.unique(result.vertex_panel)) == {0, 1, 2, 3}


def test_single_piece_garment_cape():
    body = make_simple_body()
    cape = create_cape()
    result = preview_drape_garment_on_body(cape, body, pin="min_y", n_steps=100)
    assert np.all(np.isfinite(result.vertices))
    assert result.vertices[:, 1].max() <= body.chest_height + 0.05


def test_zero_steps_is_just_placement_and_weld():
    body = make_simple_body()
    skirt = create_skirt(panels=4)
    result = preview_drape_garment_on_body(skirt, body, pin="min_y", n_steps=0, collide=False)
    assert np.all(np.isfinite(result.vertices))
    assert result.faces.shape[1] == 3
