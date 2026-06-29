"""The pose-conditioned deformation network (TailorNet-style).

Requires PyTorch (imported lazily). The factory records the dimensions so the
training/export code and the numpy runtime stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NetworkDims:
    n_garment_params: int = 10
    n_shape_params: int = 10
    n_pose_params: int = 63           # 21 joints x 3
    n_vertices: int = 5000
    hidden_dim: int = 256
    n_hidden_layers: int = 4

    @property
    def input_dim(self) -> int:
        return self.n_garment_params + self.n_shape_params + self.n_pose_params

    @property
    def output_dim(self) -> int:
        return self.n_vertices * 3


def build_model(dims: NetworkDims):
    """Construct a ``GarmentDeformationNet`` for the given dimensions (needs torch)."""
    import torch.nn as nn

    class GarmentDeformationNet(nn.Module):
        """Predicts per-vertex offsets from garment params and body pose.

        Input:  garment_params (G) + body_shape (S) + body_pose (P)
        Output: vertex offsets (V x 3); deformed = rest + offset.
        """

        def __init__(self, dims: NetworkDims):
            super().__init__()
            self.dims = dims
            layers = [nn.Linear(dims.input_dim, dims.hidden_dim), nn.ReLU()]
            for _ in range(dims.n_hidden_layers - 1):
                layers += [nn.Linear(dims.hidden_dim, dims.hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(dims.hidden_dim, dims.output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, garment_params, shape_params, pose_params):
            import torch
            x = torch.cat([garment_params, shape_params, pose_params], dim=-1)
            return self.net(x).view(-1, self.dims.n_vertices, 3)

    return GarmentDeformationNet(dims)
