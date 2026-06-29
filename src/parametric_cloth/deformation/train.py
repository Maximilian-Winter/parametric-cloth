"""Train the deformation network and export it for runtime use.

Requires PyTorch (lazy import). Exports both ONNX (for engine ONNX Runtime) and
a numpy ``.npz`` consumed by :class:`~parametric_cloth.deformation.runtime.NumpyMLP`,
so the model can run without torch or onnxruntime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import DeformationDataset
from .model import NetworkDims, build_model


@dataclass
class TrainResult:
    model: object                     # the trained nn.Module
    dims: NetworkDims
    losses: list[float]
    input_mean: np.ndarray
    input_std: np.ndarray


def dims_from_dataset(dataset: DeformationDataset, **overrides) -> NetworkDims:
    base = dict(
        n_garment_params=dataset.garment_params.shape[1],
        n_shape_params=dataset.shapes.shape[1],
        n_pose_params=dataset.poses.shape[1],
        n_vertices=dataset.n_vertices,
    )
    base.update(overrides)
    return NetworkDims(**base)


def train_deformation_model(
    dataset: DeformationDataset,
    *,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    dims: NetworkDims | None = None,
    seed: int = 0,
) -> TrainResult:
    """Train on ``(inputs -> offsets)`` and return the model plus norm stats."""
    import torch

    torch.manual_seed(seed)
    dims = dims or dims_from_dataset(dataset)

    mean, std = dataset.normalization()
    x = (dataset.inputs() - mean) / std
    y = dataset.offsets()

    g = torch.tensor(dataset.garment_params, dtype=torch.float32)
    # Re-split the normalized inputs back into the three blocks the net expects.
    xt = torch.tensor(x, dtype=torch.float32)
    gp = xt[:, : dims.n_garment_params]
    sp = xt[:, dims.n_garment_params: dims.n_garment_params + dims.n_shape_params]
    pp = xt[:, dims.n_garment_params + dims.n_shape_params:]
    target = torch.tensor(y, dtype=torch.float32)

    dataset_t = torch.utils.data.TensorDataset(gp, sp, pp, target)
    loader = torch.utils.data.DataLoader(dataset_t, batch_size=batch_size, shuffle=True)

    model = build_model(dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    losses: list[float] = []
    for epoch in range(n_epochs):
        total = 0.0
        for gb, sb, pb, tb in loader:
            pred = model(gb, sb, pb).reshape(gb.shape[0], -1)
            loss = loss_fn(pred, tb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total / max(len(loader), 1))
        if epoch % 20 == 0:
            print(f"epoch {epoch}: loss={losses[-1]:.6f}")

    return TrainResult(model=model, dims=dims, losses=losses,
                       input_mean=mean, input_std=std)


def export_to_npz(result: TrainResult, path: str) -> str:
    """Export trained weights for the pure-numpy runtime."""
    import torch  # noqa: F401

    linears = [m for m in result.model.net if m.__class__.__name__ == "Linear"]
    arrays: dict[str, np.ndarray] = {
        "n_layers": np.array(len(linears)),
        "n_vertices": np.array(result.dims.n_vertices),
        "input_mean": result.input_mean,
        "input_std": result.input_std,
    }
    for i, layer in enumerate(linears):
        # torch Linear stores weight as (out, in); numpy runtime uses x @ W -> (in, out).
        arrays[f"W{i}"] = layer.weight.detach().cpu().numpy().T
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    np.savez(path, **arrays)
    return path


def export_to_onnx(result: TrainResult, path: str) -> str:
    """Export the trained model to ONNX for engine deployment."""
    import torch

    model = result.model.eval()
    d = result.dims
    dummy = (
        torch.randn(1, d.n_garment_params),
        torch.randn(1, d.n_shape_params),
        torch.randn(1, d.n_pose_params),
    )
    torch.onnx.export(
        model, dummy, path,
        input_names=["garment_params", "shape_params", "pose_params"],
        output_names=["vertex_offsets"],
        dynamic_axes={
            "garment_params": {0: "batch"},
            "shape_params": {0: "batch"},
            "pose_params": {0: "batch"},
            "vertex_offsets": {0: "batch"},
        },
    )
    return path
