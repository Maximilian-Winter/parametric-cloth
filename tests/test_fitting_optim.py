import numpy as np

from parametric_cloth.fitting.optim import Adam


def test_adam_converges_on_quadratic_bowl():
    target = np.array([3.0, -2.0, 5.0])
    params = np.zeros(3)
    optimizer = Adam(params.shape, lr=0.1)

    for _ in range(500):
        grad = 2.0 * (params - target)
        params = optimizer.step(params, grad)

    np.testing.assert_allclose(params, target, atol=1e-2)


def test_adam_loss_decreases_monotonically_on_average():
    target = np.array([1.0, 1.0])
    params = np.array([10.0, -10.0])
    optimizer = Adam(params.shape, lr=0.2)

    losses = []
    for _ in range(50):
        losses.append(float(np.sum((params - target) ** 2)))
        grad = 2.0 * (params - target)
        params = optimizer.step(params, grad)

    assert losses[-1] < losses[0] * 0.1
