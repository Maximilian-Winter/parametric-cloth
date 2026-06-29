"""Lightweight inference benchmarking.

Used to check the per-garment deformation cost against the runtime budget
(the design targets <5 ms per garment).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class BenchmarkResult:
    n_iterations: int
    mean_ms: float
    min_ms: float
    max_ms: float

    @property
    def within_budget(self) -> bool:
        return self.mean_ms < 5.0     # design target: <5 ms per garment


def benchmark(fn: Callable[[], object], *, n_iterations: int = 100,
              warmup: int = 5) -> BenchmarkResult:
    """Time ``fn`` over ``n_iterations`` calls (after ``warmup`` untimed calls)."""
    for _ in range(max(0, warmup)):
        fn()

    samples = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)

    return BenchmarkResult(
        n_iterations=n_iterations,
        mean_ms=sum(samples) / len(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )
