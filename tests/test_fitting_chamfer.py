import numpy as np

from parametric_cloth.fitting.chamfer import chamfer_distance, chamfer_distance_and_grad


def test_zero_distance_for_identical_sets():
    a = np.random.default_rng(0).normal(size=(10, 3))
    assert chamfer_distance(a, a.copy()) == 0.0


def test_distance_increases_with_offset():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 3))
    near = a + 0.01
    far = a + 1.0
    assert chamfer_distance(a, near) < chamfer_distance(a, far)


def test_loss_matches_chamfer_distance():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(6, 3)), rng.normal(size=(9, 3))
    loss, _ = chamfer_distance_and_grad(a, b)
    assert loss == chamfer_distance(a, b)


def test_gradient_check_against_finite_differences():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(7, 3))
    b = rng.normal(size=(5, 3))

    _, analytic = chamfer_distance_and_grad(a, b)

    eps = 1e-6
    fd = np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        ap, am = a.copy(), a.copy()
        ap[idx] += eps
        am[idx] -= eps
        fd[idx] = (chamfer_distance(ap, b) - chamfer_distance(am, b)) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, atol=1e-4, rtol=1e-3)


def test_gradient_zero_at_exact_match():
    a = np.random.default_rng(4).normal(size=(6, 3))
    _, grad = chamfer_distance_and_grad(a, a.copy())
    assert np.allclose(grad, 0.0, atol=1e-9)
