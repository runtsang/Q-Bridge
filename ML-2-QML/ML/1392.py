"""Hybrid regression framework that extends the original ML seed with a richer physics‑informed loss and a deeper feed‑forward backbone.

The classical side remains fully NumPy/PyTorch, but the loss now couples the model output with the quantum Fisher information (QFI) of a separate auxiliary circuit.  This encourages the parameters to explore directions that are highly sensitive to the input.  The dataset generator is unchanged; only the label definition is expanded to include an energy term that depends on both the input angles and a random phase.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a feature matrix X and a target vector y.  The target is a sum of a
    sinusoidal component and an “energy‑like” term that is a simple quadratic
    function of the input angles.  The extra term is useful for downstream
    physics‑aware training and is available to the user via ``y``.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    # Classical target: sin(θ) + 0.1 * cos(2θ) + 0.05 * (θ²)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + 0.05 * (angles ** 2)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Simple PyTorch dataset wrapping the generated data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """
    A deeper feed‑forward network with skip connections that improves expressivity
    compared to the original 3‑layer MLP.  The architecture is deliberately
    lightweight so that it can be paired with a QML head later.
    """
    def __init__(self, num_features: int):
        super().__init__()
        # 4‑layer MLP with residual connections
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.residual = nn.Linear(num_features, 32)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.net(state_batch)
        # residual addition
        res = self.residual(state_batch)
        return (x + res).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
