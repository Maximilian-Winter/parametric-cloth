import importlib
import json

import numpy as np
import pytest

from parametric_cloth.variants import (
    PCABasis,
    blend_shape_targets,
    build_pca_basis,
    build_variant_library,
    coefficients_to_weights,
    generate_sample_points,
    latin_hypercube,
)
from parametric_cloth.variants.library import VariantLibrary


def _low_rank_variants(n=8, V=60, rank=3, seed=0):
    """Variants lying in a `rank`-dim affine subspace -> PCA recovers them exactly."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(V, 3))
    modes = rng.normal(size=(rank, V, 3))
    out = []
    for _ in range(n):
        coeffs = rng.normal(size=rank)
        out.append(base + np.tensordot(coeffs, modes, axes=1))
    return out, base, modes


# --- PCA -------------------------------------------------------------------

def test_pca_roundtrip_is_exact_for_low_rank_data():
    variants, _, _ = _low_rank_variants(rank=3)
    basis = build_pca_basis(variants, n_components=3)
    for v in variants:
        assert basis.reconstruction_error(v) < 1e-9


def test_decode_zero_coeffs_is_mean():
    variants, _, _ = _low_rank_variants()
    basis = build_pca_basis(variants, n_components=4)
    recon = basis.decode(np.zeros(basis.n_components))
    assert np.allclose(recon, basis.mean_shape)


def test_encode_shape_and_mean_recovers_zero():
    variants, _, _ = _low_rank_variants()
    basis = build_pca_basis(variants, n_components=4)
    coeffs = basis.encode(basis.mean_shape)
    assert coeffs.shape == (basis.n_components,)
    assert np.allclose(coeffs, 0.0, atol=1e-9)


def test_explained_variance_descending_and_bounded():
    variants, _, _ = _low_rank_variants(rank=4)
    basis = build_pca_basis(variants, n_components=4)
    evr = basis.explained_variance_ratio
    assert np.all(np.diff(evr) <= 1e-12)          # non-increasing
    assert 0.0 <= evr.sum() <= 1.0 + 1e-9


def test_more_components_reduce_error():
    variants, _, _ = _low_rank_variants(rank=3)
    b1 = build_pca_basis(variants, n_components=1)
    b3 = build_pca_basis(variants, n_components=3)
    err1 = np.mean([b1.reconstruction_error(v) for v in variants])
    err3 = np.mean([b3.reconstruction_error(v) for v in variants])
    assert err3 < err1
    assert err3 < 1e-9


def test_components_clamped_to_rank():
    variants, _, _ = _low_rank_variants(n=5)
    basis = build_pca_basis(variants, n_components=50)
    assert basis.n_components <= 5            # cannot exceed data rank (N)


def test_pca_rejects_mismatched_topology():
    a = np.zeros((10, 3))
    b = np.zeros((11, 3))
    with pytest.raises(ValueError):
        build_pca_basis([a, b])


def test_pca_requires_two_variants():
    with pytest.raises(ValueError):
        build_pca_basis([np.zeros((10, 3))])


def test_basis_save_load_roundtrip(tmp_path):
    variants, _, _ = _low_rank_variants()
    basis = build_pca_basis(variants, n_components=4)
    path = str(tmp_path / "basis.npz")
    basis.save(path)
    loaded = PCABasis.load(path)
    assert loaded.n_components == basis.n_components
    assert np.allclose(loaded.mean_shape, basis.mean_shape)
    assert np.allclose(loaded.components, basis.components)


# --- sampling --------------------------------------------------------------

def test_generate_sample_points_count():
    ranges = {"length": (50, 80), "flare": (1.0, 2.0)}
    samples = generate_sample_points(ranges)
    assert len(samples) == 2 * len(ranges) + 1    # center + 2 per dim
    center = {"length": 65, "flare": 1.5}
    assert center in samples


def test_generate_sample_points_includes_extremes():
    ranges = {"length": (50, 80)}
    values = {s["length"] for s in generate_sample_points(ranges)}
    assert {50, 65, 80}.issubset(values)


def test_latin_hypercube_within_ranges_and_deterministic():
    ranges = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
    s1 = latin_hypercube(ranges, 5, seed=42)
    s2 = latin_hypercube(ranges, 5, seed=42)
    assert len(s1) == 5
    assert s1 == s2                                # deterministic
    for s in s1:
        assert 0.0 <= s["a"] <= 1.0 and 10.0 <= s["b"] <= 20.0


def test_latin_hypercube_seed_changes_result():
    ranges = {"a": (0.0, 1.0)}
    assert latin_hypercube(ranges, 5, seed=1) != latin_hypercube(ranges, 5, seed=2)


# --- variant library -------------------------------------------------------

def _synthetic_simulator(V=40, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(V, 3))
    da = rng.normal(size=(V, 3))
    db = rng.normal(size=(V, 3))

    def simulate(params):
        return base + params["a"] * da + params["b"] * db
    return simulate


def test_build_variant_library_reconstructs():
    sim = _synthetic_simulator()
    samples = latin_hypercube({"a": (-1, 1), "b": (-1, 1)}, 8, seed=0)
    lib = build_variant_library(samples, sim, n_components=3)
    assert len(lib.variants) == 8
    for name in lib.variants:
        original = sim(lib.parameters[name])
        assert np.allclose(lib.reconstruct(name), original, atol=1e-7)


def test_build_variant_library_skips_failures():
    sim = _synthetic_simulator()

    def flaky(params):
        return None if params["a"] > 0.5 else sim(params)

    samples = [{"a": a, "b": 0.0} for a in (-1, 0, 0.4, 0.9, -0.5)]
    lib = build_variant_library(samples, flaky, n_components=2)
    assert len(lib.variants) == 4                  # only a=0.9 is skipped


def test_build_variant_library_needs_two():
    with pytest.raises(ValueError):
        build_variant_library([{"a": 0, "b": 0}], _synthetic_simulator())


def test_library_save_load_roundtrip(tmp_path):
    sim = _synthetic_simulator()
    samples = latin_hypercube({"a": (-1, 1), "b": (-1, 1)}, 6, seed=3)
    lib = build_variant_library(samples, sim, n_components=3)
    lib.save(str(tmp_path / "lib"))

    loaded = VariantLibrary.load(str(tmp_path / "lib"))
    assert set(loaded.variants) == set(lib.variants)
    for name in lib.variants:
        assert np.allclose(loaded.reconstruct(name), lib.reconstruct(name), atol=1e-9)
        assert loaded.parameters[name] == lib.parameters[name]

    meta = json.loads((tmp_path / "lib" / "metadata.json").read_text())
    assert meta["n_variants"] == 6


# --- blend shapes ----------------------------------------------------------

def test_blend_shape_targets():
    variants, _, _ = _low_rank_variants(rank=3)
    basis = build_pca_basis(variants, n_components=3)
    targets = blend_shape_targets(basis)
    assert len(targets) == basis.n_components + 1
    assert targets[0].name == "Basis"
    assert np.allclose(targets[0].vertices, basis.mean_shape)
    assert np.allclose(targets[1].vertices, basis.mean_shape + basis.components[0])


def test_coefficients_to_weights_scale():
    coeffs = np.array([2.0, 4.0])
    assert np.allclose(coefficients_to_weights(coeffs, scale=2.0), [1.0, 2.0])


def test_blend_shape_export_requires_bpy():
    mod = importlib.import_module("parametric_cloth.variants.blendshapes")
    variants, _, _ = _low_rank_variants()
    basis = build_pca_basis(variants, n_components=2)
    with pytest.raises(ModuleNotFoundError):
        mod.export_pca_as_blend_shapes(basis, "base.obj", "out.fbx")
