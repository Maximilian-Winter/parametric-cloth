import numpy as np
import pytest

from parametric_cloth.deformation import NumpyMLP
from parametric_cloth.deformation.poses import POSE_DIM
from parametric_cloth.engine import (
    DeformState,
    GarmentCoverage,
    LearnedDeformer,
    PCADeformer,
    RuntimeGarment,
    TextureCustomization,
    Wardrobe,
    apply_customization,
    benchmark,
    covered_regions,
    hidden_garment_regions,
    resolve_visible_regions,
    visible_body_faces,
)
from parametric_cloth.engine.deformer import ONNXDeformer
from parametric_cloth.variants.pca import build_pca_basis


def _basis(V=12, rank=3, n=6, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(V, 3))
    modes = rng.normal(size=(rank, V, 3))
    variants = [base + np.tensordot(rng.normal(size=rank), modes, axes=1) for _ in range(n)]
    return build_pca_basis(variants, n_components=3)


# --- PCA deformer (Path A) -------------------------------------------------

def test_pca_deformer_decodes_coefficients():
    basis = _basis()
    deformer = PCADeformer(basis)
    coeffs = np.array([0.5, -0.3, 0.1])
    out = deformer.deform(DeformState(pca_coefficients=coeffs))
    assert np.allclose(out, basis.decode(coeffs))


def test_pca_deformer_defaults_to_mean():
    basis = _basis()
    out = PCADeformer(basis).deform(DeformState())
    assert np.allclose(out, basis.mean_shape)


# --- learned deformer (Path B) ---------------------------------------------

def _mlp(V=5, G=2, S=10, P=POSE_DIM, seed=1):
    rng = np.random.default_rng(seed)
    D = G + S + P
    return NumpyMLP(weights=[rng.normal(size=(D, V * 3)) * 0.01],
                    biases=[np.zeros(V * 3)], n_vertices=V)


def test_learned_deformer_adds_offsets_to_rest():
    mlp = _mlp(V=5)
    rest = np.random.default_rng(2).normal(size=(5, 3))
    deformer = LearnedDeformer(mlp, rest)
    state = DeformState(garment_params=np.zeros(2), shape=np.zeros(10),
                        pose=np.zeros(POSE_DIM))
    out = deformer.deform(state)
    expected = rest + mlp.predict_offsets(state.garment_params, state.shape, state.pose)
    assert np.allclose(out, expected)


def test_learned_deformer_rest_when_no_inputs():
    mlp = _mlp(V=5)
    rest = np.ones((5, 3))
    assert np.allclose(LearnedDeformer(mlp, rest).deform(DeformState()), rest)


def test_learned_deformer_partial_inputs_raise():
    mlp = _mlp(V=5)
    deformer = LearnedDeformer(mlp, np.zeros((5, 3)))
    with pytest.raises(ValueError):
        deformer.deform(DeformState(pose=np.zeros(POSE_DIM)))   # missing shape/params


def test_learned_deformer_rejects_rest_mismatch():
    with pytest.raises(ValueError):
        LearnedDeformer(_mlp(V=5), np.zeros((6, 3)))


def test_onnx_deformer_requires_runtime():
    with pytest.raises(ModuleNotFoundError):
        ONNXDeformer("model.onnx", np.zeros((5, 3)))


# --- masking / layering ----------------------------------------------------

def test_visible_body_faces():
    face_regions = np.array(["torso", "torso", "left_arm", "leg"])
    mask = visible_body_faces(face_regions, {"torso"})
    assert list(mask) == [False, False, True, True]


def test_covered_regions_union():
    covs = [
        GarmentCoverage("shirt", {"torso", "left_arm"}),
        GarmentCoverage("skirt", {"hips", "legs"}),
    ]
    assert covered_regions(covs) == {"torso", "left_arm", "hips", "legs"}


def test_layering_outer_hides_inner():
    covs = [
        GarmentCoverage("shirt", {"torso"}, layer=2),
        GarmentCoverage("jacket", {"torso", "arms"}, layer=3),
    ]
    visible = resolve_visible_regions(covs)
    assert visible["jacket"] == {"torso", "arms"}
    assert visible["shirt"] == set()                   # fully under the jacket
    hidden = hidden_garment_regions(covs)
    assert hidden["shirt"] == {"torso"}


def test_layering_same_layer_both_visible():
    covs = [
        GarmentCoverage("a", {"left"}, layer=1),
        GarmentCoverage("b", {"right"}, layer=1),
    ]
    visible = resolve_visible_regions(covs)
    assert visible["a"] == {"left"} and visible["b"] == {"right"}


# --- texture ---------------------------------------------------------------

def test_apply_color_multiply():
    base = np.ones((2, 2, 3))
    out = apply_customization(base, TextureCustomization(base_color=(1.0, 0.5, 0.0)))
    assert np.allclose(out[..., 0], 1.0)
    assert np.allclose(out[..., 1], 0.5)
    assert np.allclose(out[..., 2], 0.0)


def test_apply_pattern_overlay():
    base = np.zeros((2, 2, 3))
    pattern = np.ones((2, 2, 3))
    custom = TextureCustomization(pattern=pattern, pattern_opacity=0.5)
    out = apply_customization(base, custom)
    assert np.allclose(out, 0.5)


def test_texture_pattern_shape_mismatch_raises():
    base = np.zeros((2, 2, 3))
    custom = TextureCustomization(pattern=np.ones((3, 3, 3)), pattern_opacity=0.5)
    with pytest.raises(ValueError):
        apply_customization(base, custom)


# --- wardrobe --------------------------------------------------------------

def test_wardrobe_equip_and_layer_order():
    w = Wardrobe()
    w.equip("outerwear", "jacket")
    w.equip("top", "shirt")
    w.equip("bottom", "jeans")
    order = [gid for _, gid, _ in w.ordered_garments()]
    assert order == ["jeans", "shirt", "jacket"]       # inner -> outer


def test_wardrobe_unequip_and_unknown_slot():
    w = Wardrobe()
    w.equip("top", "shirt")
    assert w.unequip("top") == "shirt"
    assert w.unequip("top") is None
    with pytest.raises(ValueError):
        w.equip("hat", "fedora")


def test_wardrobe_loadout_save_load(tmp_path):
    w = Wardrobe()
    w.equip("top", "shirt")
    w.equip("bottom", "skirt")
    w.save(str(tmp_path / "loadout.json"))
    restored = Wardrobe.load(str(tmp_path / "loadout.json"))
    assert restored.loadout() == w.loadout()


# --- runtime garment + profiling -------------------------------------------

def test_runtime_garment_deforms_and_reports_coverage():
    basis = _basis()
    g = RuntimeGarment("dress", PCADeformer(basis), regions={"torso", "legs"}, layer=2)
    out = g.deform(DeformState(pca_coefficients=np.zeros(basis.n_components)))
    assert out.shape == (basis.n_vertices, 3)
    cov = g.coverage()
    assert cov.regions == {"torso", "legs"} and cov.layer == 2


def test_benchmark_returns_positive_timings():
    basis = _basis()
    deformer = PCADeformer(basis)
    state = DeformState(pca_coefficients=np.zeros(basis.n_components))
    result = benchmark(lambda: deformer.deform(state), n_iterations=20, warmup=2)
    assert result.n_iterations == 20
    assert result.mean_ms >= 0.0
    assert result.min_ms <= result.mean_ms <= result.max_ms
