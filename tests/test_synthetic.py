import numpy as np
import pytest

from parametric_cloth.avatar.synthetic import make_simple_body


def test_body_has_the_anchors_templates_need():
    body = make_simple_body()
    assert set(body.anchors) == {"chest_front", "chest_back", "left_upper_arm", "right_upper_arm"}


def test_anchor_normals_point_outward_from_body_axis():
    body = make_simple_body()
    for name, (pos, normal) in body.anchors.items():
        assert np.linalg.norm(normal) == pytest.approx(1.0)
        radial = np.array([pos[0], 0.0, pos[2]])
        if np.linalg.norm(radial) < 1e-6:
            continue
        # Outward: the normal should point away from the central axis, not into it.
        assert np.dot(normal, radial) > 0, f"{name} normal points inward"


def test_front_and_back_anchors_are_on_opposite_sides():
    body = make_simple_body()
    front_pos, _ = body.anchors["chest_front"]
    back_pos, _ = body.anchors["chest_back"]
    assert front_pos[2] > 0 and back_pos[2] < 0


def test_left_and_right_arm_anchors_are_mirrored():
    body = make_simple_body()
    left_pos, _ = body.anchors["left_upper_arm"]
    right_pos, _ = body.anchors["right_upper_arm"]
    assert left_pos[0] < 0 and right_pos[0] > 0
    assert left_pos[0] == pytest.approx(-right_pos[0])


def test_radius_at_height_matches_endpoints():
    body = make_simple_body(hip_radius=0.16, chest_radius=0.15)
    assert body.radius_at_height(body.hip_height) == pytest.approx(0.16)
    assert body.radius_at_height(body.chest_height) == pytest.approx(0.15)


def test_mesh_is_finite_and_nonempty():
    body = make_simple_body()
    assert body.mesh.n_vertices > 0
    assert np.all(np.isfinite(body.mesh.vertices))
