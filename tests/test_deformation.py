import importlib

import numpy as np
import pytest

from parametric_cloth.deformation import (
    DeformationDataset,
    NumpyMLP,
    dims_from_dataset,
    load_amass,
    random_poses,
    subsample_poses,
    zero_pose,
)
from parametric_cloth.deformation.poses import POSE_DIM


def _dataset(N=6, V=8, G=3, S=10, P=POSE_DIM, seed=0):
    rng = np.random.default_rng(seed)
    rest = rng.normal(size=(V, 3))
    return DeformationDataset(
        rest_vertices=rest,
        garment_params=rng.normal(size=(N, G)),
        shapes=rng.normal(size=(N, S)),
        poses=rng.normal(size=(N, P)),
        vertices=rest[None] + rng.normal(scale=0.01, size=(N, V, 3)),
    )


# --- dataset ---------------------------------------------------------------

def test_dataset_dims_and_offsets():
    ds = _dataset(N=6, V=8, G=3)
    assert ds.n_samples == 6 and ds.n_vertices == 8
    assert ds.input_dim == 3 + 10 + POSE_DIM
    assert ds.inputs().shape == (6, ds.input_dim)
    assert ds.offsets().shape == (6, 8 * 3)
    # Offsets are vertices minus rest.
    assert np.allclose(ds.offsets()[0], (ds.vertices[0] - ds.rest_vertices).reshape(-1))


def test_dataset_normalization():
    ds = _dataset()
    mean, std = ds.normalization()
    assert mean.shape == (ds.input_dim,)
    assert np.all(std > 0)


def test_dataset_rejects_inconsistent_counts():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        DeformationDataset(
            rest_vertices=np.zeros((4, 3)),
            garment_params=rng.normal(size=(5, 3)),
            shapes=rng.normal(size=(4, 10)),
            poses=rng.normal(size=(4, POSE_DIM)),
            vertices=np.zeros((4, 4, 3)),
        )


def test_dataset_save_load(tmp_path):
    ds = _dataset()
    ds.save(str(tmp_path / "d.npz"))
    loaded = DeformationDataset.load(str(tmp_path / "d.npz"))
    assert np.allclose(loaded.vertices, ds.vertices)
    assert np.allclose(loaded.rest_vertices, ds.rest_vertices)


def test_dataset_from_samples():
    rest = np.zeros((4, 3))
    samples = [
        {"garment_params": [1, 2], "body_shape": np.zeros(10),
         "body_pose": np.zeros(POSE_DIM), "vertices": np.ones((4, 3))},
        {"garment_params": [3, 4], "body_shape": np.zeros(10),
         "body_pose": np.zeros(POSE_DIM), "vertices": np.full((4, 3), 2.0)},
    ]
    ds = DeformationDataset.from_samples(rest, samples)
    assert ds.n_samples == 2 and ds.garment_params.shape == (2, 2)


# --- poses -----------------------------------------------------------------

def test_pose_helpers():
    assert zero_pose().shape == (POSE_DIM,)
    assert random_poses(7).shape == (7, POSE_DIM)


def test_subsample_even_coverage():
    poses = np.arange(10 * POSE_DIM).reshape(10, POSE_DIM).astype(float)
    sub = subsample_poses(poses, 4)
    assert sub.shape == (4, POSE_DIM)
    assert np.allclose(sub[0], poses[0]) and np.allclose(sub[-1], poses[9])


def test_subsample_returns_all_when_requested_too_many():
    poses = np.zeros((3, POSE_DIM))
    assert subsample_poses(poses, 10).shape == (3, POSE_DIM)


def test_load_amass(tmp_path):
    raw = np.arange(8 * 70).reshape(8, 70).astype(float)   # 3 global + 63 body + extra
    np.savez(str(tmp_path / "amass.npz"), poses=raw)
    body = load_amass(str(tmp_path / "amass.npz"), n=4)
    assert body.shape == (4, POSE_DIM)
    # First selected row is the body slice of frame 0.
    assert np.allclose(body[0], raw[0, 3:3 + POSE_DIM])


# --- numpy runtime ---------------------------------------------------------

def test_numpy_mlp_single_layer_matches_linear():
    rng = np.random.default_rng(1)
    D, V = 5, 4
    W = rng.normal(size=(D, V * 3))
    b = rng.normal(size=(V * 3,))
    mlp = NumpyMLP(weights=[W], biases=[b], n_vertices=V)
    x = rng.normal(size=D)
    expected = (x @ W + b).reshape(V, 3)
    assert np.allclose(mlp.forward(x)[0], expected)


def test_numpy_mlp_two_layer_relu_matches_manual():
    rng = np.random.default_rng(2)
    D, H, V = 6, 7, 3
    W0, b0 = rng.normal(size=(D, H)), rng.normal(size=H)
    W1, b1 = rng.normal(size=(H, V * 3)), rng.normal(size=V * 3)
    mlp = NumpyMLP(weights=[W0, W1], biases=[b0, b1], n_vertices=V)
    x = rng.normal(size=D)
    hidden = np.maximum(x @ W0 + b0, 0.0)
    expected = (hidden @ W1 + b1).reshape(V, 3)
    assert np.allclose(mlp.forward(x)[0], expected)


def test_numpy_mlp_applies_normalization():
    rng = np.random.default_rng(3)
    D, V = 4, 2
    W, b = rng.normal(size=(D, V * 3)), rng.normal(size=V * 3)
    mean, std = rng.normal(size=D), np.abs(rng.normal(size=D)) + 0.5
    mlp = NumpyMLP(weights=[W], biases=[b], n_vertices=V, input_mean=mean, input_std=std)
    x = rng.normal(size=D)
    expected = (((x - mean) / std) @ W + b).reshape(V, 3)
    assert np.allclose(mlp.forward(x)[0], expected)


def test_numpy_mlp_predict_offsets_concatenation_order():
    rng = np.random.default_rng(4)
    G, S, P, V = 2, 10, POSE_DIM, 3
    D = G + S + P
    W, b = rng.normal(size=(D, V * 3)), rng.normal(size=V * 3)
    mlp = NumpyMLP(weights=[W], biases=[b], n_vertices=V)
    gp, sh, po = rng.normal(size=G), rng.normal(size=S), rng.normal(size=P)
    direct = mlp.forward(np.concatenate([gp, sh, po]))[0]
    assert np.allclose(mlp.predict_offsets(gp, sh, po), direct)


def test_numpy_mlp_save_load(tmp_path):
    rng = np.random.default_rng(5)
    mlp = NumpyMLP(weights=[rng.normal(size=(4, 6))], biases=[rng.normal(size=6)],
                   n_vertices=2, input_mean=np.zeros(4), input_std=np.ones(4))
    mlp.save(str(tmp_path / "m.npz"))
    loaded = NumpyMLP.load(str(tmp_path / "m.npz"))
    x = rng.normal(size=4)
    assert np.allclose(loaded.forward(x), mlp.forward(x))


# --- torch-only paths fail gracefully here ---------------------------------

def test_dims_from_dataset_is_pure():
    ds = _dataset(G=4, V=8)
    dims = dims_from_dataset(ds)
    assert dims.n_garment_params == 4
    assert dims.n_vertices == 8
    assert dims.input_dim == 4 + 10 + POSE_DIM


def test_build_model_requires_torch():
    mod = importlib.import_module("parametric_cloth.deformation.model")
    from parametric_cloth.deformation import NetworkDims
    with pytest.raises(ModuleNotFoundError):
        mod.build_model(NetworkDims())
