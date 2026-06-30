#!/usr/bin/env python3
"""Module 10: refine a pattern piece's shape to match a target via gradient descent.

Needs: nothing beyond the base install -- the differentiable mass-spring
solver and Chamfer loss are pure NumPy. Optionally `pip install -e ".[viz]"`
for the loss-curve plot.
Run: python examples/05_differentiable_fitting.py
"""

from __future__ import annotations

import numpy as np

from parametric_cloth.fitting import DifferentiableClothFitter
from parametric_cloth.simulation.tessellate import tessellate_piece
from parametric_cloth.templates import create_cape


def span(piece, axis: str) -> float:
    values = [v.x if axis == "x" else v.y for v in piece.vertices]
    return max(values) - min(values)


def main() -> None:
    start = create_cape(neck_half=12.0, length=60.0, flare=1.8)
    target = create_cape(neck_half=20.0, length=95.0, flare=2.6)

    mesh = tessellate_piece(target.pieces[0], levels=0)
    target_scan = np.zeros((mesh.n_vertices, 3))
    target_scan[:, :2] = mesh.vertices * 0.01   # cm -> m

    fitter = DifferentiableClothFitter(n_sim_steps=12, stiffness=40.0, regularization=0.0)
    result = fitter.fit_piece_to_3d_scan(start.pieces[0], target_scan, n_iterations=300, lr=0.5)

    reduction = 100 * (1 - result.losses[-1] / result.losses[0])
    print(f"loss: {result.losses[0]:.4f} -> {result.losses[-1]:.4f}  ({reduction:.1f}% reduction)")
    print()
    print(f"{'':10}{'start':>10}{'target':>10}{'fitted':>10}")
    print(f"{'width':10}{span(start.pieces[0], 'x'):10.1f}{span(target.pieces[0], 'x'):10.1f}"
          f"{span(result.piece, 'x'):10.1f}")
    print(f"{'height':10}{span(start.pieces[0], 'y'):10.1f}{span(target.pieces[0], 'y'):10.1f}"
          f"{span(result.piece, 'y'):10.1f}")

    try:
        from parametric_cloth import viz
        viz.plot_loss_curve(result.losses, label="cape fit (chamfer)")
    except ModuleNotFoundError:
        print("\n(install matplotlib via `pip install -e \".[viz]\"` to see the loss curve)")


if __name__ == "__main__":
    main()
