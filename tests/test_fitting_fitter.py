import numpy as np
import pytest

from parametric_cloth.fitting import DifferentiableClothFitter
from parametric_cloth.fitting.fitter import _replace_piece, _select_piece, resolve_pin_mask
from parametric_cloth.fitting.rendering import OrthographicCamera, SoftSplatRenderer
from parametric_cloth.pattern import PatternPiece, PatternVertex
from parametric_cloth.simulation.tessellate import tessellate_piece
from parametric_cloth.templates import create_cape, create_skirt


def _square_piece(name="sq", size=20.0):
    return PatternPiece(
        name=name,
        vertices=[
            PatternVertex(0, 0), PatternVertex(size, 0),
            PatternVertex(size, size), PatternVertex(0, size),
        ],
    )


# --- pin resolution ---------------------------------------------------

def test_resolve_pin_mask_min_y():
    verts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    mask = resolve_pin_mask(verts, "min_y")
    assert list(mask) == [True, True, False, False]


def test_resolve_pin_mask_max_y():
    verts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    mask = resolve_pin_mask(verts, "max_y")
    assert list(mask) == [False, False, True, True]


def test_resolve_pin_mask_explicit_indices():
    verts = np.zeros((5, 2))
    mask = resolve_pin_mask(verts, [0, 2])
    assert list(mask) == [True, False, True, False, False]


def test_resolve_pin_mask_explicit_bool():
    verts = np.zeros((3, 2))
    bool_mask = np.array([True, False, True])
    assert np.array_equal(resolve_pin_mask(verts, bool_mask), bool_mask)


def test_resolve_pin_mask_unknown_strategy_raises():
    with pytest.raises(ValueError):
        resolve_pin_mask(np.zeros((3, 2)), "middle")


# --- piece selection / replacement -----------------------------------

def test_select_piece_single_piece_garment():
    cape = create_cape()
    assert _select_piece(cape, None).name == "cape"


def test_select_piece_requires_name_for_multi_piece():
    skirt = create_skirt(panels=4)
    with pytest.raises(ValueError):
        _select_piece(skirt, None)
    assert _select_piece(skirt, "panel_2").name == "panel_2"


def test_select_piece_unknown_name_raises():
    cape = create_cape()
    with pytest.raises(KeyError):
        _select_piece(cape, "nonexistent")


def test_replace_piece_only_touches_named_piece():
    skirt = create_skirt(panels=4)
    new_panel0 = PatternPiece(name="panel_0", vertices=[PatternVertex(0, 0)] * 3)
    out = _replace_piece(skirt, new_panel0)
    assert out.piece("panel_0") is new_panel0
    assert out.piece("panel_1") is skirt.piece("panel_1")


# --- end-to-end: fit drives loss down -----------------------------------

def test_fit_to_3d_scan_reduces_loss():
    piece = _square_piece(size=20.0)
    fitter = DifferentiableClothFitter(n_sim_steps=10, stiffness=40.0, regularization=0.0)

    # Build a target by draping a *larger* square -> the fitter should grow
    # the panel toward it, decreasing chamfer loss over iterations.
    bigger = _square_piece(size=26.0)
    mesh = tessellate_piece(bigger, levels=0)
    target = mesh.vertices.copy()
    target_3d = np.zeros((target.shape[0], 3))
    target_3d[:, :2] = target * 0.01

    result = fitter.fit_piece_to_3d_scan(piece, target_3d, n_iterations=60, lr=0.5)
    assert result.losses[-1] < result.losses[0]


def test_fit_to_silhouette_reduces_loss():
    piece = _square_piece(size=15.0)
    fitter = DifferentiableClothFitter(n_sim_steps=8, stiffness=40.0, regularization=0.0)
    camera = OrthographicCamera(resolution=(32, 32), extent=0.5)
    renderer = SoftSplatRenderer(camera)

    bigger = _square_piece(size=20.0)
    mesh = tessellate_piece(bigger, levels=0)
    x0 = np.zeros((mesh.n_vertices, 3))
    x0[:, :2] = mesh.vertices * 0.01
    target_silhouette = renderer.render(x0)

    result = fitter.fit_piece_to_silhouette(
        piece, target_silhouette, camera=camera, n_iterations=60, lr=0.5,
    )
    assert result.losses[-1] < result.losses[0]


def test_fit_result_piece_has_same_corner_count():
    piece = _square_piece()
    fitter = DifferentiableClothFitter(n_sim_steps=5, regularization=0.0)
    target = np.zeros((4, 3))
    target[:, :2] = np.array([[0, 0], [25, 0], [25, 25], [0, 25]]) * 0.01
    result = fitter.fit_piece_to_3d_scan(piece, target, n_iterations=5, lr=0.1)
    assert len(result.piece.vertices) == len(piece.vertices)


def test_regularization_pulls_back_toward_initial():
    piece = _square_piece(size=20.0)
    far_target = np.zeros((4, 3))
    far_target[:, :2] = np.array([[0, 0], [200, 0], [200, 200], [0, 200]]) * 0.01

    loose = DifferentiableClothFitter(n_sim_steps=5, regularization=0.0)
    strict = DifferentiableClothFitter(n_sim_steps=5, regularization=50.0)

    r_loose = loose.fit_piece_to_3d_scan(piece, far_target, n_iterations=30, lr=0.5)
    r_strict = strict.fit_piece_to_3d_scan(piece, far_target, n_iterations=30, lr=0.5)

    def total_span(p):
        xs = [v.x for v in p.vertices]
        return max(xs) - min(xs)

    # Heavy regularization should keep the panel closer to its original size
    # than a fit with no regularization, given the same far-away target.
    assert total_span(r_strict.piece) < total_span(r_loose.piece)


# --- GarmentDefinition-level wrappers -----------------------------------

def test_fit_to_3d_scan_on_garment_cape():
    cape = create_cape()
    fitter = DifferentiableClothFitter(n_sim_steps=5, regularization=0.01)
    mesh = tessellate_piece(cape.pieces[0], levels=0)
    target = np.zeros((mesh.n_vertices, 3))
    target[:, :2] = mesh.vertices * 0.01

    out = fitter.fit_to_3d_scan(cape, target, n_iterations=5, lr=0.1)
    assert out.name == cape.name
    assert out.piece("cape") is not None
    assert out.is_valid(), out.validate()


def test_fit_to_silhouette_requires_piece_name_for_multi_piece_garment():
    skirt = create_skirt(panels=4)
    fitter = DifferentiableClothFitter(n_sim_steps=3)
    target = np.zeros((10, 10))
    with pytest.raises(ValueError):
        fitter.fit_to_silhouette(skirt, target, n_iterations=1)
