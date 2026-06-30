"""Learned text-to-fabric-parameter regression for open-ended descriptions.

For descriptions that don't fuzzy-match a preset (``find_closest_preset``
returns ``None``), a small regressor maps a sentence embedding to the six
physical parameters. Requires ``sentence-transformers`` and ``torch``
(imported lazily); training data is paired (description, KES-F measurement)
text/parameter examples -- see :func:`training_pairs_from_presets` for a
zero-cost bootstrap from the preset table itself.

The six output parameters, in fixed order: mass_per_area, stiffness, bending,
damping, friction, stretch_limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fabric import FabricProperties
from .presets import FABRIC_PRESETS_EXTENDED, fabric_properties_for

PARAM_NAMES = (
    "mass_per_area", "stiffness", "bending", "damping", "friction", "stretch_limit",
)
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size


def training_pairs_from_presets() -> tuple[list[str], np.ndarray]:
    """Bootstrap (description, parameters) pairs from the preset table itself.

    Zero-cost starting data: each preset's human-readable name (underscores ->
    spaces) is paired with its own parameter vector. Real training should
    supplement this with KES-F-measured fabric descriptions.
    """
    descriptions = [name.replace("_", " ") for name in FABRIC_PRESETS_EXTENDED]
    targets = np.array([
        [fabric_properties_for(name).__dict__[p] for p in PARAM_NAMES]
        for name in FABRIC_PRESETS_EXTENDED
    ])
    return descriptions, targets


@dataclass
class FabricPredictor:
    """Predicts :class:`FabricProperties` from a free-text description.

    ``weights``/``biases`` follow the same plain-numpy convention as
    :class:`~parametric_cloth.deformation.runtime.NumpyMLP`, so a trained
    predictor can run without torch once exported.
    """

    weights: list[np.ndarray] | None = None
    biases: list[np.ndarray] | None = None
    _encoder: object = None

    def _embed(self, description: str) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # lazy
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return np.asarray(self._encoder.encode(description), dtype=float)

    def predict(self, description: str) -> FabricProperties:
        """Predict fabric properties from a text description (needs encoder + weights)."""
        if self.weights is None:
            raise RuntimeError(
                "FabricPredictor has no trained weights; call train() or load() first"
            )
        x = self._embed(description)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        values = dict(zip(PARAM_NAMES, (float(v) for v in x)))
        return FabricProperties(**values)

    def save(self, path: str) -> str:
        arrays = {"n_layers": np.array(len(self.weights or []))}
        for i, (w, b) in enumerate(zip(self.weights or [], self.biases or [])):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> "FabricPredictor":
        d = np.load(path)
        n = int(d["n_layers"])
        return cls(weights=[d[f"W{i}"] for i in range(n)],
                   biases=[d[f"b{i}"] for i in range(n)])

    def train(
        self,
        descriptions: list[str],
        measurements: np.ndarray,
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        hidden_dim: int = 64,
        seed: int = 0,
    ) -> "FabricPredictor":
        """Train the regressor on paired (description, KES-F measurement) data.

        Requires ``torch``. Stores weights in the plain-numpy format so
        :meth:`predict` (and any later use) doesn't need torch at inference time.
        """
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        embeddings = np.stack([self._embed(d) for d in descriptions])
        x = torch.tensor(embeddings, dtype=torch.float32)
        y = torch.tensor(np.asarray(measurements, dtype=float), dtype=torch.float32)

        net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(PARAM_NAMES)),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            pred = net(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        linears = [m for m in net if m.__class__.__name__ == "Linear"]
        self.weights = [layer.weight.detach().numpy().T for layer in linears]
        self.biases = [layer.bias.detach().numpy() for layer in linears]
        return self
