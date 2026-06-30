#!/usr/bin/env python3
"""Fit a pattern piece to a target silhouette or 3D scan (Module 10).

Runs entirely in pure NumPy -- no Blender, no GPU, no autodiff framework. This
is a from-scratch differentiable mass-spring simulator + soft-splat renderer
(see ``parametric_cloth.fitting``), not a production-grade differentiable
pipeline -- swap in Warp/DiffCloth + nvdiffrast/PyTorch3D for that.

    python scripts/fit_pattern.py --initial cape.json --piece cape \
        --target-scan target_scan.npy --output fitted_cape.json --iterations 100

For silhouette fitting, pass --target-silhouette (a .npy boolean/float HxW
array, e.g. produced by SoftSplatRenderer.render on a reference mesh).
Converting a real photo to a silhouette mask needs an external segmentation
step (e.g. rembg, SAM) not implemented here.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from parametric_cloth.fitting import DifferentiableClothFitter
from parametric_cloth.serialization import load_garment, save_garment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--piece", default=None,
                        help="piece name (default: the garment's only piece)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-silhouette", help=".npy HxW target silhouette")
    group.add_argument("--target-scan", help=".npy (V,3) target point cloud")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.5)
    args = parser.parse_args(argv)

    garment = load_garment(args.initial)
    fitter = DifferentiableClothFitter()

    if args.target_silhouette:
        target = np.load(args.target_silhouette)
        result = fitter.fit_to_silhouette(
            garment, target, piece_name=args.piece,
            n_iterations=args.iterations, lr=args.lr,
        )
    else:
        target = np.load(args.target_scan)
        result = fitter.fit_to_3d_scan(
            garment, target, piece_name=args.piece,
            n_iterations=args.iterations, lr=args.lr,
        )

    save_garment(result, args.output)
    print(f"wrote fitted garment to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
