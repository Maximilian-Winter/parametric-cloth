import importlib

import numpy as np
import pytest

from parametric_cloth.avatar import place_garment
from parametric_cloth.avatar.placement import PlacementTransform
from parametric_cloth.fabric import FabricProperties, FabricType
from parametric_cloth.pattern import (
    GarmentDefinition,
    PatternPiece,
    PatternVertex,
    Seam,
    SeamEdge,
)
from parametric_cloth.simulation import (
    AssembledGarment,
    SimulationConfig,
    assemble_garment,
    boundary_loop,
    closest_point_pairs,
    cloth_settings_from_fabric,
    ear_clip,
    midpoint_subdivide,
    seam_vertex_chain,
    settle_delta,
    tessellate_piece,
    validate_simulation_result,
)


def _square(name="sq", size=10.0):
    return PatternPiece(
        name=name,
        vertices=[
            PatternVertex(0, 0), PatternVertex(size, 0),
            PatternVertex(size, size), PatternVertex(0, size),
        ],
    )


def _tri_area_2d(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    ab, ac = b - a, c - a
    cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    return 0.5 * np.abs(cross).sum()


# --- tessellation ----------------------------------------------------------

def test_ear_clip_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    tris = ear_clip(sq)
    assert len(tris) == 2


def test_ear_clip_nonconvex_tshirt_front():
    # The T-shirt front has a concave neckline -> exercises reflex handling.
    from parametric_cloth.templates import create_tshirt
    front = create_tshirt().piece("front")
    pts = np.array([[v.x, v.y] for v in front.vertices])
    tris = ear_clip(pts)
    assert len(tris) == len(pts) - 2          # a simple polygon triangulates to n-2


def test_tessellation_preserves_area():
    piece = create_skirt_panel()
    mesh = tessellate_piece(piece, levels=2)
    assert _tri_area_2d(mesh.vertices, mesh.faces) == pytest.approx(piece.area(), rel=1e-6)


def create_skirt_panel():
    from parametric_cloth.templates import create_skirt
    return create_skirt(panels=4).pieces[0]


def test_subdivision_keeps_original_vertices_first():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    faces = np.array(ear_clip(sq))
    v, f = midpoint_subdivide(sq, faces, levels=1)
    # Original 4 corners unchanged and still at the front.
    assert np.allclose(v[:4], sq)
    assert len(f) == len(faces) * 4


def test_tessellate_levels_increase_resolution():
    piece = _square()
    low = tessellate_piece(piece, levels=1)
    high = tessellate_piece(piece, levels=3)
    assert high.n_vertices > low.n_vertices
    assert high.n_corners == low.n_corners == 4


# --- seams -----------------------------------------------------------------

def test_boundary_loop_of_subdivided_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    faces = np.array(ear_clip(sq))
    _, f = midpoint_subdivide(sq, faces, levels=1)
    loop = boundary_loop(f)
    # 4 corners + 1 midpoint per edge = 8 boundary vertices.
    assert len(loop) == 8
    assert set(range(4)).issubset(set(loop))


def test_seam_vertex_chain_isolates_one_edge():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    faces = np.array(ear_clip(sq))
    _, f = midpoint_subdivide(sq, faces, levels=1)
    loop = boundary_loop(f)
    chain = seam_vertex_chain(loop, n_corners=4, start_corner=0, end_corner=1)
    assert chain[0] == 0 and chain[-1] == 1
    assert len(chain) == 3                       # corner, midpoint, corner
    assert chain[1] >= 4                          # interior point is a midpoint


def test_closest_point_pairs():
    a = np.array([[0, 0, 0], [0, 0, 1.0]])
    b = np.array([[0, 0, 1.0], [0, 0, 0]])
    pairs = closest_point_pairs(a, b)
    assert pairs == [(0, 1), (1, 0)]


# --- assembly --------------------------------------------------------------

def test_assemble_combines_pieces():
    from parametric_cloth.templates import create_skirt
    skirt = create_skirt(panels=4)
    asm = assemble_garment(skirt, levels=1)
    assert isinstance(asm, AssembledGarment)
    expected_v = sum(
        tessellate_piece(p, levels=1).n_vertices for p in skirt.pieces
    )
    assert asm.n_vertices == expected_v
    assert asm.seam_pairs                          # skirt has wrap-around seams


def test_assembly_seams_cross_between_pieces():
    from parametric_cloth.templates import create_skirt
    skirt = create_skirt(panels=4)
    asm = assemble_garment(skirt, levels=1)
    # Every welded pair must join two *different* pieces.
    def piece_of(global_idx):
        for name, (lo, hi) in asm.piece_offsets.items():
            if lo <= global_idx < hi:
                return name
        return None
    assert all(piece_of(i) != piece_of(j) for i, j in asm.seam_pairs)


def test_two_squares_seam_welds_to_zero_gap():
    # Two unit (10 cm) squares; B shifted +10 cm in x so A's right edge meets
    # B's left edge exactly -> the seam gap should collapse to ~0.
    a = _square("A")
    b = _square("B")
    garment = GarmentDefinition(
        name="two_squares",
        pieces=[a, b],
        seams=[Seam(edge_a=SeamEdge("A", 1, 2), edge_b=SeamEdge("B", 0, 3))],
    )
    transforms = {
        "A": PlacementTransform(location=np.zeros(3), rotation=np.eye(3)),
        "B": PlacementTransform(location=np.array([0.10, 0, 0]), rotation=np.eye(3)),
    }
    asm = assemble_garment(garment, transforms, levels=1)
    assert asm.seam_gap() == pytest.approx(0.0, abs=1e-9)


def test_place_then_assemble_skirt_on_body():
    # End-to-end (sans Blender): place a skirt on a synthetic body and assemble.
    from tests.test_avatar import make_cylinder
    from parametric_cloth.templates import create_skirt

    mesh = make_cylinder(radius=0.15, n_theta=60, n_levels=180)
    skirt = create_skirt(panels=6)
    transforms = place_garment(skirt, mesh)
    asm = assemble_garment(skirt, transforms, levels=2)
    assert asm.n_vertices > 0
    assert np.all(np.isfinite(asm.vertices))
    assert asm.seam_gap() < 1.0                    # finite, not exploded


# --- validation ------------------------------------------------------------

def test_validate_accepts_settled_mesh():
    verts = np.random.RandomState(0).uniform(-0.3, 0.3, size=(100, 3))
    res = validate_simulation_result(verts, np.zeros(3), max_distance=2.0)
    assert res.ok and not res.issues


def test_validate_detects_explosion():
    verts = np.zeros((10, 3))
    verts[0] = [100.0, 0, 0]
    res = validate_simulation_result(verts, np.zeros(3), max_distance=2.0)
    assert not res.ok and "exploded" in res.issues[0]


def test_validate_detects_nan():
    verts = np.zeros((10, 3))
    verts[0, 0] = np.nan
    res = validate_simulation_result(verts, np.zeros(3))
    assert not res.ok and "non-finite" in res.issues[0]


def test_settle_delta():
    a = np.zeros((5, 3))
    b = np.zeros((5, 3))
    b[:, 1] = 0.01
    assert settle_delta(a, b) == pytest.approx(0.01)
    assert settle_delta(a, np.zeros((4, 3))) == float("inf")


# --- config / fabric mapping ----------------------------------------------

def test_cloth_settings_track_fabric_stiffness():
    silk = cloth_settings_from_fabric(FabricProperties.from_preset(FabricType.SILK))
    denim = cloth_settings_from_fabric(FabricProperties.from_preset(FabricType.DENIM))
    assert denim.tension_stiffness > silk.tension_stiffness
    assert denim.bending_stiffness > silk.bending_stiffness
    assert denim.mass > silk.mass


def test_damping_schedule_grows():
    sched = SimulationConfig(max_retries=3).damping_schedule()
    assert sched == [1.0, 2.0, 4.0]


def test_damping_multiplier_applied():
    base = cloth_settings_from_fabric(FabricProperties.from_preset(FabricType.COTTON))
    damped = cloth_settings_from_fabric(
        FabricProperties.from_preset(FabricType.COTTON), damping_multiplier=2.0
    )
    assert damped.tension_damping == pytest.approx(base.tension_damping * 2.0)


# --- Blender driver stays import-safe without bpy ---------------------------

def test_blender_sim_imports_without_bpy():
    mod = importlib.import_module("parametric_cloth.simulation.blender_sim")
    assert hasattr(mod, "simulate_garment")


def test_blender_calls_require_bpy():
    mod = importlib.import_module("parametric_cloth.simulation.blender_sim")
    with pytest.raises(ModuleNotFoundError):
        mod.clear_scene()
