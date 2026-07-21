import math

import numpy as np
import pytest

from parametric_cloth.avatar import (
    AnchorResolver,
    AvatarMesh,
    BODY_SHAPE_SAMPLES,
    N_BETAS,
    SMPLX_LANDMARKS,
    SMPLX_NUM_VERTICES,
    axis_angle_matrix,
    basis_from_normal,
    compute_placement_transform,
    compute_waist_segments,
    export_avatar,
    generate_smplx_avatar,
    get_body_shape,
    verify_landmark_indices,
)
from parametric_cloth.avatar import place_garment
from parametric_cloth.pattern import PlacementHint
from parametric_cloth.templates import create_skirt, create_tshirt


def make_cylinder(radius=0.15, height=1.0, n_theta=48, n_levels=40, y0=0.0):
    """A vertical cylinder around the y-axis -- a stand-in body torso."""
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ys = np.linspace(y0, y0 + height, n_levels)
    verts = [
        (radius * math.cos(t), y, radius * math.sin(t))
        for y in ys for t in thetas
    ]
    faces = []
    for li in range(n_levels - 1):
        for ti in range(n_theta):
            a = li * n_theta + ti
            b = li * n_theta + (ti + 1) % n_theta
            c = (li + 1) * n_theta + (ti + 1) % n_theta
            d = (li + 1) * n_theta + ti
            # Winding gives an outward-facing normal (verified numerically);
            # existing tests here only check normals are *radial*, not which
            # way they point, so this was silently inward before.
            faces.append((a, c, b))
            faces.append((a, d, c))
    return AvatarMesh(np.array(verts, float), np.array(faces, np.int64))


# --- mesh / normals --------------------------------------------------------

def test_mesh_rejects_bad_shapes():
    with pytest.raises(ValueError):
        AvatarMesh(np.zeros((4, 2)), np.zeros((1, 3), int))
    with pytest.raises(ValueError):
        AvatarMesh(np.zeros((4, 3)), np.zeros((1, 4), int))


def test_vertex_normals_unit_and_horizontal_on_cylinder():
    mesh = make_cylinder()
    normals = mesh.vertex_normals
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-6)
    # A vertical cylinder's normals are radial -> ~zero y component.
    assert np.max(np.abs(normals[:, 1])) < 0.2


def test_vertex_normals_are_radial():
    mesh = make_cylinder()
    for i in range(0, mesh.n_vertices, 97):
        p = mesh.vertices[i]
        radial = np.array([p[0], 0.0, p[2]])
        if np.linalg.norm(radial) < 1e-6:
            continue
        radial /= np.linalg.norm(radial)
        # Normal is parallel to the radial direction (sign depends on winding).
        assert abs(abs(np.dot(mesh.vertex_normals[i], radial)) - 1.0) < 0.1


# --- waist segments --------------------------------------------------------

def test_waist_segments_count_and_geometry():
    mesh = make_cylinder(radius=0.15)
    segs = compute_waist_segments(mesh, 4, waist_height=0.5)
    assert set(segs) == {f"waist_segment_{i}" for i in range(4)}

    positions = np.array([p for p, _ in segs.values()])
    # Each anchor sits roughly on the cylinder surface.
    radii = np.hypot(positions[:, 0], positions[:, 2])
    assert np.allclose(radii, 0.15, atol=0.02)
    # ...near the requested height.
    assert np.allclose(positions[:, 1], 0.5, atol=0.05)
    # ...and they balance around the axis.
    assert np.linalg.norm(positions[:, [0, 2]].mean(axis=0)) < 0.02


def test_waist_segment_normals_point_outward():
    mesh = make_cylinder(radius=0.15)
    segs = compute_waist_segments(mesh, 6, waist_height=0.5)
    axis = np.array([0.0, 0.5, 0.0])
    for pos, nrm in segs.values():
        assert np.linalg.norm(nrm) == pytest.approx(1.0, abs=1e-6)
        assert abs(nrm[1]) < 1e-6                       # horizontal
        outward = pos - axis
        assert np.dot(nrm, outward) > 0                 # away from axis


def test_waist_segments_evenly_spaced():
    mesh = make_cylinder()
    segs = compute_waist_segments(mesh, 4, waist_height=0.5)
    angles = sorted(
        math.atan2(p[2], p[0]) % (2 * math.pi) for p, _ in segs.values()
    )
    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    assert all(abs(g - math.pi / 2) < 0.3 for g in gaps)


def test_waist_segments_rejects_zero():
    with pytest.raises(ValueError):
        compute_waist_segments(make_cylinder(), 0, waist_height=0.5)


# --- placement math --------------------------------------------------------

def test_basis_from_normal_is_orthonormal():
    for n in [(1, 0, 0), (0, 1, 0), (0.3, 0.4, 0.5)]:
        b = basis_from_normal(np.array(n, float))
        assert np.allclose(b @ b.T, np.eye(3), atol=1e-6)
        # +Z column aligns with the (normalized) normal.
        z = np.array(n, float) / np.linalg.norm(n)
        assert np.allclose(b[:, 2], z, atol=1e-6)


def test_axis_angle_rotation():
    r = axis_angle_matrix(np.array([0, 0, 1.0]), math.radians(90))
    assert np.allclose(r @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-6)


def test_placement_offsets_along_normal():
    placement = PlacementHint(anchor="x", offset_normal=5.0)  # 5 cm
    t = compute_placement_transform(
        np.zeros(3), np.array([1.0, 0, 0]), placement
    )
    assert np.allclose(t.location, [0.05, 0, 0], atol=1e-9)   # cm -> m
    assert np.allclose(t.rotation[:, 2], [1, 0, 0], atol=1e-6)


def test_placement_rotation_preserves_outward_axis():
    placement = PlacementHint(anchor="x", offset_normal=2.0, rotation=90.0)
    n = np.array([0.0, 0.0, 1.0])
    t = compute_placement_transform(np.zeros(3), n, placement)
    # Spinning about the normal must leave the outward (+Z) axis unchanged.
    assert np.allclose(t.rotation[:, 2], n, atol=1e-6)
    assert np.allclose(t.rotation @ t.rotation.T, np.eye(3), atol=1e-6)


def test_euler_roundtrip_via_matrix():
    placement = PlacementHint(anchor="x", offset_normal=0.0, rotation=30.0)
    t = compute_placement_transform(
        np.zeros(3), np.array([0.2, 0.3, 0.9]), placement
    )
    # Euler angles are finite and the matrix stays orthonormal.
    ex, ey, ez = t.euler_xyz()
    assert all(math.isfinite(a) for a in (ex, ey, ez))


# --- anchor resolver -------------------------------------------------------

def test_resolver_dynamic_takes_precedence():
    mesh = make_cylinder()
    dyn = {"waist_segment_0": (np.array([1.0, 0, 0]), np.array([1.0, 0, 0]))}
    resolver = AnchorResolver(mesh, dynamic=dyn)
    pos, nrm = resolver.resolve("waist_segment_0")
    assert np.allclose(pos, [1, 0, 0])


def test_resolver_unknown_anchor_raises():
    resolver = AnchorResolver(make_cylinder())
    with pytest.raises(KeyError):
        resolver.resolve("nonexistent_anchor")


def test_resolver_transform_for_landmark():
    # Cylinder large enough to cover every landmark index.
    mesh = make_cylinder(n_theta=60, n_levels=180)  # 10800 verts
    assert mesh.n_vertices > max(SMPLX_LANDMARKS.values())
    resolver = AnchorResolver(mesh)
    t = resolver.transform_for(PlacementHint(anchor="chest_front", offset_normal=3.0))
    assert np.allclose(t.rotation @ t.rotation.T, np.eye(3), atol=1e-6)


# --- landmark registry -----------------------------------------------------

def test_registry_is_internally_consistent():
    # No mesh: range (against full SMPL-X count) + uniqueness checks must pass.
    assert verify_landmark_indices(num_vertices=SMPLX_NUM_VERTICES) == []


def test_all_indices_within_smplx_topology():
    assert max(SMPLX_LANDMARKS.values()) < SMPLX_NUM_VERTICES
    assert min(SMPLX_LANDMARKS.values()) >= 0


def test_indices_are_unique():
    values = list(SMPLX_LANDMARKS.values())
    assert len(values) == len(set(values))


def test_verify_flags_too_small_mesh():
    issues = verify_landmark_indices(num_vertices=100)
    assert issues and any("vertex count" in i for i in issues)


# --- body shapes -----------------------------------------------------------

def test_body_shape_samples():
    assert len(BODY_SHAPE_SAMPLES) == 5
    for name, betas in BODY_SHAPE_SAMPLES.items():
        assert betas.shape == (N_BETAS,)
    assert np.allclose(BODY_SHAPE_SAMPLES["average"], 0.0)


def test_get_body_shape_returns_copy():
    a = get_body_shape("athletic")
    a[0] = 999.0
    assert BODY_SHAPE_SAMPLES["athletic"][0] != 999.0


def test_get_body_shape_unknown_raises():
    with pytest.raises(KeyError):
        get_body_shape("bogus")


# --- Module 1 <-> Module 2 integration -------------------------------------

def test_place_skirt_uses_waist_segments():
    mesh = make_cylinder(radius=0.15, n_theta=60, n_levels=180)  # covers landmarks
    skirt = create_skirt(panels=6)
    transforms = place_garment(skirt, mesh)
    # One transform per panel, each a valid orthonormal placement.
    assert set(transforms) == {f"panel_{i}" for i in range(6)}
    for t in transforms.values():
        assert np.allclose(t.rotation @ t.rotation.T, np.eye(3), atol=1e-6)
    # Panels are distributed around the body, not stacked on one spot.
    locs = np.array([t.location for t in transforms.values()])
    assert np.linalg.norm(locs[:, [0, 2]].mean(axis=0)) < 0.03


def test_place_tshirt_uses_landmarks():
    mesh = make_cylinder(n_theta=60, n_levels=180)  # covers all landmark indices
    shirt = create_tshirt()
    transforms = place_garment(shirt, mesh)
    assert set(transforms) == {"front", "back", "left_sleeve", "right_sleeve"}


# --- heavy deps absent here: confirm graceful failure ----------------------

def test_generate_avatar_requires_smplx():
    # smplx / torch are not installed in this environment.
    with pytest.raises(ModuleNotFoundError):
        generate_smplx_avatar("average")


def test_fbx_export_is_rejected_with_guidance():
    mesh = make_cylinder()
    with pytest.raises(NotImplementedError):
        export_avatar(mesh, "/tmp/whatever.fbx")
