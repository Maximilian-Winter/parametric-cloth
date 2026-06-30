import numpy as np
import pytest

from parametric_cloth.fitting.mass_spring import DifferentiableMassSpring, SpringTopology
from parametric_cloth.simulation.tessellate import tessellate_piece
from parametric_cloth.templates import create_cape


def _grad_check(sim, X0, v, *, eps=1e-6, atol=1e-4, rtol=1e-3):
    """Compare sim.backward(v) to a central-difference directional derivative
    of loss(X0) = sum(forward(X0) * v), the standard VJP correctness check."""
    def loss(X):
        return float(np.sum(sim.forward(X) * v))

    sim.forward(X0)
    analytic = sim.backward(v)

    fd = np.zeros_like(X0)
    it = np.nditer(X0, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        Xp, Xm = X0.copy(), X0.copy()
        Xp[idx] += eps
        Xm[idx] -= eps
        fd[idx] = (loss(Xp) - loss(Xm)) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, atol=atol, rtol=rtol)
    return analytic, fd


# --- topology ----------------------------------------------------------

def test_topology_from_edges():
    topo = SpringTopology.from_edges([(0, 1), (1, 2)], stiffness=10.0)
    assert list(topo.i) == [0, 1]
    assert list(topo.j) == [1, 2]
    assert np.allclose(topo.stiffness, 10.0)


def test_topology_from_edges_per_edge_stiffness():
    topo = SpringTopology.from_edges([(0, 1), (1, 2)], stiffness=[5.0, 15.0])
    assert np.allclose(topo.stiffness, [5.0, 15.0])


def test_topology_from_faces_dedupes_shared_edges():
    # Two triangles sharing edge (1,2) -> 5 unique edges, not 6.
    faces = [(0, 1, 2), (1, 3, 2)]
    topo = SpringTopology.from_faces(faces, stiffness=1.0)
    edges = {tuple(sorted((i, j))) for i, j in zip(topo.i, topo.j)}
    assert edges == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)}


# --- forward sanity ------------------------------------------------------

def test_pinned_particle_does_not_move():
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(4, 3))
    topo = SpringTopology.from_edges([(0, 1), (1, 2), (2, 3), (3, 0)], stiffness=20.0)
    pinned = np.array([True, False, False, False])
    sim = DifferentiableMassSpring(topo, pinned, n_steps=20)
    X_final = sim.forward(X0)
    assert np.allclose(X_final[0], X0[0])


def test_unpinned_system_falls_under_gravity():
    X0 = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    topo = SpringTopology.from_edges([(0, 1)], stiffness=5.0)
    pinned = np.array([False, False])
    sim = DifferentiableMassSpring(topo, pinned, n_steps=10)
    X_final = sim.forward(X0)
    assert X_final[:, 1].mean() < 0.0     # gravity pulls -y


# --- gradient checks (the critical correctness test) ----------------------

def test_gradient_check_small_random_system():
    rng = np.random.default_rng(1)
    n = 6
    X0 = rng.normal(scale=0.1, size=(n, 3))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4), (4, 5), (2, 5)]
    topo = SpringTopology.from_edges(edges, stiffness=rng.uniform(5, 20, size=len(edges)))
    pinned = np.array([True, False, False, False, False, True])
    sim = DifferentiableMassSpring(topo, pinned, n_steps=8)
    v = rng.normal(size=(n, 3))
    _grad_check(sim, X0, v)


def test_gradient_check_no_pinned_vertices():
    rng = np.random.default_rng(2)
    n = 5
    X0 = rng.normal(scale=0.1, size=(n, 3))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    topo = SpringTopology.from_edges(edges, stiffness=12.0)
    pinned = np.zeros(n, dtype=bool)
    sim = DifferentiableMassSpring(topo, pinned, n_steps=6)
    v = rng.normal(size=(n, 3))
    _grad_check(sim, X0, v)


def test_gradient_check_real_tessellated_piece():
    # A real garment panel -> realistic spring topology, not a toy graph.
    cape = create_cape()
    mesh = tessellate_piece(cape.pieces[0], levels=0)
    rng = np.random.default_rng(3)
    X0 = np.zeros((mesh.n_vertices, 3))
    X0[:, :2] = mesh.vertices * 0.01
    X0 += rng.normal(scale=0.005, size=X0.shape)   # perturb off the flat plane

    from parametric_cloth.fitting.fitter import resolve_pin_mask
    pinned = resolve_pin_mask(mesh.vertices, "min_y")
    assert pinned.any()

    topo = SpringTopology.from_faces(mesh.faces, stiffness=30.0)
    sim = DifferentiableMassSpring(topo, pinned, n_steps=5)
    v = rng.normal(size=X0.shape)
    _grad_check(sim, X0, v, atol=1e-3, rtol=1e-2)


def test_gradient_check_with_higher_damping_and_more_steps():
    rng = np.random.default_rng(4)
    n = 4
    X0 = rng.normal(scale=0.1, size=(n, 3))
    topo = SpringTopology.from_edges([(0, 1), (1, 2), (2, 3), (3, 0)], stiffness=8.0)
    pinned = np.array([True, True, False, False])
    sim = DifferentiableMassSpring(topo, pinned, n_steps=15, damping=0.9)
    v = rng.normal(size=(n, 3))
    _grad_check(sim, X0, v)


def test_backward_before_forward_raises():
    topo = SpringTopology.from_edges([(0, 1)], stiffness=1.0)
    sim = DifferentiableMassSpring(topo, np.array([False, False]))
    with pytest.raises(RuntimeError):
        sim.backward(np.zeros((2, 3)))
