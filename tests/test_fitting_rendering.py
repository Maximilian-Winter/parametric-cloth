import numpy as np

from parametric_cloth.fitting.rendering import OrthographicCamera, SoftSplatRenderer


def test_coverage_peaks_at_splat_center():
    cam = OrthographicCamera(axis="z", resolution=(32, 32), extent=1.0)
    renderer = SoftSplatRenderer(cam, sigma_px=1.0)
    points = np.array([[0.0, 0.0, 0.0]])           # projects to image center
    coverage = renderer.render(points)
    center = coverage[16, 16]
    corner = coverage[0, 0]
    assert center > corner
    assert 0.0 <= coverage.min() and coverage.max() < 1.0


def test_more_points_increase_coverage():
    cam = OrthographicCamera(resolution=(16, 16), extent=1.0)
    renderer = SoftSplatRenderer(cam)
    one = renderer.render(np.array([[0.0, 0.0, 0.0]]))
    many = renderer.render(np.tile([[0.0, 0.0, 0.0]], (5, 1)))
    assert many.max() > one.max()


def test_render_and_loss_matches_render_then_mse():
    cam = OrthographicCamera(resolution=(20, 20), extent=1.0)
    renderer = SoftSplatRenderer(cam)
    rng = np.random.default_rng(0)
    points = rng.normal(scale=0.3, size=(6, 3))
    target = rng.uniform(size=(20, 20))

    loss, _ = renderer.render_and_loss(points, target)
    direct = float(np.mean((renderer.render(points) - target) ** 2))
    assert loss == direct


def test_gradient_check_against_finite_differences():
    cam = OrthographicCamera(resolution=(24, 24), extent=1.0)
    renderer = SoftSplatRenderer(cam, sigma_px=1.5)
    rng = np.random.default_rng(1)
    points = rng.normal(scale=0.3, size=(5, 3))
    target = rng.uniform(size=(24, 24))

    _, analytic = renderer.render_and_loss(points, target)

    eps = 1e-5
    fd = np.zeros_like(points)
    for idx in np.ndindex(points.shape):
        pp, pm = points.copy(), points.copy()
        pp[idx] += eps
        pm[idx] -= eps
        loss_p, _ = renderer.render_and_loss(pp, target)
        loss_m, _ = renderer.render_and_loss(pm, target)
        fd[idx] = (loss_p - loss_m) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, atol=1e-3, rtol=1e-2)


def test_gradient_check_different_camera_axes():
    rng = np.random.default_rng(2)
    points = rng.normal(scale=0.3, size=(4, 3))
    target = rng.uniform(size=(16, 16))

    for axis in ("x", "y", "z"):
        cam = OrthographicCamera(axis=axis, resolution=(16, 16), extent=1.0)
        renderer = SoftSplatRenderer(cam, sigma_px=1.2)
        _, analytic = renderer.render_and_loss(points, target)

        eps = 1e-5
        fd = np.zeros_like(points)
        for idx in np.ndindex(points.shape):
            pp, pm = points.copy(), points.copy()
            pp[idx] += eps
            pm[idx] -= eps
            loss_p, _ = renderer.render_and_loss(pp, target)
            loss_m, _ = renderer.render_and_loss(pm, target)
            fd[idx] = (loss_p - loss_m) / (2 * eps)

        np.testing.assert_allclose(analytic, fd, atol=1e-3, rtol=1e-2)
