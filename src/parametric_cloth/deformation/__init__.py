"""Module 6: Learned Runtime Deformation.

A small pose-conditioned MLP predicts garment vertex offsets from body pose/shape
and garment parameters, replacing the runtime physics solver. Dataset assembly,
pose sampling, and a numpy inference runtime are pure (testable here); the network
and training/ONNX export use torch (lazy import).
"""

from __future__ import annotations

from .dataset import DeformationDataset
from .model import NetworkDims, build_model
from .poses import (
    POSE_DIM,
    load_amass,
    random_poses,
    subsample_poses,
    zero_pose,
)
from .runtime import NumpyMLP
from .train import (
    TrainResult,
    dims_from_dataset,
    export_to_npz,
    export_to_onnx,
    train_deformation_model,
)

__all__ = [
    "DeformationDataset",
    "NetworkDims",
    "build_model",
    "POSE_DIM",
    "zero_pose",
    "random_poses",
    "subsample_poses",
    "load_amass",
    "NumpyMLP",
    "TrainResult",
    "dims_from_dataset",
    "train_deformation_model",
    "export_to_npz",
    "export_to_onnx",
]
