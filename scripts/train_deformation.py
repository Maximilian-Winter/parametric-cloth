#!/usr/bin/env python3
"""Train a garment deformation model and export it (requires PyTorch).

    python scripts/train_deformation.py \
        --data data/training/tshirt/training_data.npz \
        --epochs 200 --onnx models/tshirt.onnx --npz models/tshirt.npz
"""

from __future__ import annotations

import argparse
import sys

from parametric_cloth.deformation import (
    DeformationDataset,
    export_to_npz,
    export_to_onnx,
    train_deformation_model,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="DeformationDataset .npz")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--onnx", default=None, help="ONNX output path")
    parser.add_argument("--npz", default=None, help="numpy-runtime weights output path")
    args = parser.parse_args(argv)

    dataset = DeformationDataset.load(args.data)
    print(f"training on {dataset.n_samples} samples, {dataset.n_vertices} vertices")

    result = train_deformation_model(
        dataset, n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    print(f"final loss: {result.losses[-1]:.6f}")

    if args.onnx:
        export_to_onnx(result, args.onnx)
        print(f"wrote ONNX model to {args.onnx}")
    if args.npz:
        export_to_npz(result, args.npz)
        print(f"wrote numpy-runtime weights to {args.npz}")
    if not (args.onnx or args.npz):
        print("note: pass --onnx and/or --npz to export the trained model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
