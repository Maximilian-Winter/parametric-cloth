"""Symmetric Chamfer distance with an analytic gradient w.r.t. the first set.

Used by ``fit_to_3d_scan``: pulls a simulated mesh toward a fixed target scan.
The gradient treats ``b`` (the target) as constant and differentiates only
through ``a`` (the optimized mesh), since that's the only free input in a
fitting loop.
"""

from __future__ import annotations

import numpy as np


def _pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean squared nearest-neighbor distance between two point sets."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d2 = _pairwise_sq_dist(a, b)
    return float(d2.min(axis=1).mean() + d2.min(axis=0).mean())


def chamfer_distance_and_grad(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """Chamfer distance plus its gradient w.r.t. ``a`` (``b`` is fixed)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d2 = _pairwise_sq_dist(a, b)
    na, nb = a.shape[0], b.shape[0]

    j_of_a = np.argmin(d2, axis=1)          # nearest b for each a_i
    term_a = d2[np.arange(na), j_of_a].mean()
    i_of_b = np.argmin(d2, axis=0)          # nearest a for each b_j
    term_b = d2[i_of_b, np.arange(nb)].mean()
    loss = float(term_a + term_b)

    grad = (2.0 / na) * (a - b[j_of_a])                # d(term_a)/da
    contrib = (2.0 / nb) * (a[i_of_b] - b)              # d(term_b)/da, per matched a
    np.add.at(grad, i_of_b, contrib)

    return loss, grad
