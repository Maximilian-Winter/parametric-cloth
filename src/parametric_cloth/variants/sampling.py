"""Parameter-space sampling strategies for variant simulation.

Full factorial sampling explodes combinatorially, so we sample sparsely: the
center plus one-at-a-time extremes (cheap, space-spanning), or a Latin hypercube
for denser, better-distributed coverage. PCA then interpolates the gaps.
"""

from __future__ import annotations

import numpy as np

ParamRanges = dict[str, tuple[float, float]]


def _center(param_ranges: ParamRanges) -> dict[str, float]:
    return {p: (lo + hi) / 2.0 for p, (lo, hi) in param_ranges.items()}


def generate_sample_points(param_ranges: ParamRanges) -> list[dict[str, float]]:
    """Center point plus each parameter pushed to its min/max in turn.

    For D parameters this yields ``2*D + 1`` samples instead of the ``3^D`` of a
    full grid -- enough to span every axis for a low-component PCA basis.
    """
    center = _center(param_ranges)
    samples = [dict(center)]
    for param, (lo, hi) in param_ranges.items():
        for value in (lo, hi):
            sample = dict(center)
            sample[param] = value
            samples.append(sample)
    return _dedupe(samples)


def latin_hypercube(
    param_ranges: ParamRanges, n_samples: int, *, seed: int = 0
) -> list[dict[str, float]]:
    """Latin hypercube sample of the parameter space (deterministic given seed)."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    keys = list(param_ranges)
    rng = np.random.default_rng(seed)

    # One stratified point per row, independently shuffled per dimension.
    edges = np.linspace(0.0, 1.0, n_samples + 1)[:n_samples]
    jitter = rng.uniform(size=(n_samples, len(keys))) / n_samples
    unit = edges[:, None] + jitter
    for j in range(len(keys)):
        rng.shuffle(unit[:, j])

    samples = []
    for i in range(n_samples):
        sample = {}
        for j, key in enumerate(keys):
            lo, hi = param_ranges[key]
            sample[key] = float(lo + unit[i, j] * (hi - lo))
        samples.append(sample)
    return samples


def _dedupe(samples: list[dict[str, float]]) -> list[dict[str, float]]:
    seen: set[tuple] = set()
    unique = []
    for s in samples:
        key = tuple(sorted((k, round(v, 9)) for k, v in s.items()))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique
