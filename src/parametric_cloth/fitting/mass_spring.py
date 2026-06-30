"""A small, fully differentiable mass-spring cloth simulator (Module 10).

Stands in for the design's "custom XPBD/PyTorch solver" option, implemented in
plain NumPy with a hand-derived reverse-mode gradient instead of an autodiff
framework -- so differentiable-fitting works with zero extra dependencies. It
is a force-based mass-spring system (semi-implicit Euler), not true XPBD
constraint projection: simpler to implement and differentiate correctly, at
the cost of being less stiff/stable for very rigid fabrics than a production
constraint solver (Warp, Taichi, DiffCloth) would be.

Rest lengths are computed from the *initial* positions on every forward call
(not frozen), so gradients correctly flow through panel scale, not just
through the post-drape position.

Gradient correctness is the entire point of this module, so every analytic
gradient here is checked against central finite differences in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DT = 1.0 / 60.0
DEFAULT_GRAVITY = np.array([0.0, -9.81, 0.0])
DEFAULT_DAMPING = 0.98


@dataclass
class SpringTopology:
    """Structural springs between particle indices (topology only)."""

    i: np.ndarray          # (E,) int
    j: np.ndarray          # (E,) int
    stiffness: np.ndarray  # (E,) float

    @classmethod
    def from_edges(cls, edges, stiffness=1.0) -> "SpringTopology":
        i = np.array([e[0] for e in edges], dtype=np.int64)
        j = np.array([e[1] for e in edges], dtype=np.int64)
        k = (np.full(len(i), float(stiffness)) if np.isscalar(stiffness)
             else np.asarray(stiffness, dtype=float))
        return cls(i=i, j=j, stiffness=k)

    @classmethod
    def from_faces(cls, faces, stiffness=1.0) -> "SpringTopology":
        """Unique edges from a triangle list (each face contributes 3 edges)."""
        edges = set()
        for a, b, c in faces:
            for u, v in ((a, b), (b, c), (c, a)):
                edges.add((u, v) if u < v else (v, u))
        return cls.from_edges(sorted(edges), stiffness=stiffness)


def _spring_forces(X: np.ndarray, rest_length: np.ndarray, topo: SpringTopology) -> np.ndarray:
    i, j, k = topo.i, topo.j, topo.stiffness
    d = X[i] - X[j]
    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    direction = d / dist[:, None]
    fmag = k * (dist - rest_length)
    Fi = -fmag[:, None] * direction
    F = np.zeros_like(X)
    np.add.at(F, i, Fi)
    np.add.at(F, j, -Fi)
    return F


def _spring_force_backward(X, gF, rest_length, topo: SpringTopology):
    """Backprop an adjoint on per-particle force ``gF`` through one force eval.

    Returns ``(dX, d_rest_length)`` -- the contribution to dL/dX (at this
    timestep) and dL/d(rest_length) (accumulated across calls by the caller).
    """
    i, j, k = topo.i, topo.j, topo.stiffness
    d = X[i] - X[j]
    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    direction = d / dist[:, None]
    fmag = k * (dist - rest_length)

    # Fi_e = -fmag_e * direction_e; F[i] += Fi_e, F[j] += -Fi_e (=Fj_e).
    g_fi = gF[i] - gF[j]                                       # dL/dFi_e, (E,3)

    dL_dfmag = np.einsum("ec,ec->e", g_fi, -direction)         # (E,)
    dL_ddir = -fmag[:, None] * g_fi                            # (E,3)

    dL_ddist = dL_dfmag * k
    dL_drest = -dL_dfmag * k

    # direction = d / dist  (standard normalize() backward)
    dot = np.einsum("ec,ec->e", direction, dL_ddir)
    dL_dd_from_dir = (dL_ddir - direction * dot[:, None]) / dist[:, None]
    dL_dd_from_dist = dL_ddist[:, None] * direction
    dL_dd = dL_dd_from_dir + dL_dd_from_dist

    dX = np.zeros_like(X)
    np.add.at(dX, i, dL_dd)
    np.add.at(dX, j, -dL_dd)
    return dX, dL_drest


@dataclass
class _Cache:
    X: np.ndarray              # (T+1, N, 3)
    rest_length: np.ndarray    # (E,)


class DifferentiableMassSpring:
    """Forward-simulates a mass-spring cloth and backprops analytic gradients.

    Usage mirrors a tiny autograd ``Function``::

        sim = DifferentiableMassSpring(topology, pinned, n_steps=40)
        X_final = sim.forward(X0)
        dL_dX0 = sim.backward(dL_dXfinal)
    """

    def __init__(
        self, topology: SpringTopology, pinned: np.ndarray, *,
        n_steps: int = 40, dt: float = DEFAULT_DT,
        damping: float = DEFAULT_DAMPING, gravity: np.ndarray = DEFAULT_GRAVITY,
    ):
        self.topology = topology
        self.pinned = np.asarray(pinned, dtype=bool)
        self.n_steps = n_steps
        self.dt = dt
        self.damping = damping
        self.gravity = np.asarray(gravity, dtype=float)
        self._cache: _Cache | None = None

    @property
    def inv_mass(self) -> np.ndarray:
        """Unit mass per particle; pinned particles have zero inverse mass."""
        return np.where(self.pinned, 0.0, 1.0)

    def forward(self, X0: np.ndarray) -> np.ndarray:
        X0 = np.asarray(X0, dtype=float)
        n = X0.shape[0]
        i, j = self.topology.i, self.topology.j
        rest_length = np.linalg.norm(X0[i] - X0[j], axis=1)

        Xs = np.empty((self.n_steps + 1, n, 3))
        Xs[0] = X0
        V = np.zeros((n, 3))
        w = self.inv_mass[:, None]

        for t in range(self.n_steps):
            F = _spring_forces(Xs[t], rest_length, self.topology)
            # w (inverse mass) gates *all* acceleration, including gravity, so
            # a pinned particle (w=0) truly stays fixed rather than still falling.
            V = (V + self.dt * w * (F + self.gravity)) * self.damping
            Xs[t + 1] = Xs[t] + self.dt * V

        self._cache = _Cache(X=Xs, rest_length=rest_length)
        return Xs[-1].copy()

    def backward(self, dL_dXfinal: np.ndarray) -> np.ndarray:
        """Gradient of a scalar loss w.r.t. the initial positions ``X0``."""
        if self._cache is None:
            raise RuntimeError("call forward() before backward()")
        Xs, rest_length = self._cache.X, self._cache.rest_length
        n = Xs.shape[1]
        i, j = self.topology.i, self.topology.j
        w = self.inv_mass[:, None]

        gX = np.array(dL_dXfinal, dtype=float, copy=True)      # dL/dX_{t+1}
        gV = np.zeros((n, 3))                                  # dL/dV_{t+1}
        gL0 = np.zeros(len(rest_length))

        for t in range(self.n_steps - 1, -1, -1):
            gv1_total = gV + self.dt * gX           # dL/dV_{t+1}, from both uses
            gV = self.damping * gv1_total            # -> dL/dV_t
            gF = self.damping * self.dt * w * gv1_total

            dX_from_F, dL0_t = _spring_force_backward(Xs[t], gF, rest_length, self.topology)
            gX = gX + dX_from_F                      # gX (identity term) + force term -> dL/dX_t
            gL0 += dL0_t

        # Propagate the rest-length gradient back through rest_length = ||X0[i]-X0[j]||.
        X0 = Xs[0]
        d0 = X0[i] - X0[j]
        dist0 = np.maximum(np.linalg.norm(d0, axis=1), 1e-9)
        dir0 = d0 / dist0[:, None]
        contrib = gL0[:, None] * dir0
        np.add.at(gX, i, contrib)
        np.add.at(gX, j, -contrib)

        return gX
