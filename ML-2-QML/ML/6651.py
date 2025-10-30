"""Hybrid classical‑quantum regression model with a classical backbone."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random superposition data for classical regression.

    The dataset is identical in statistical properties to the quantum version but
    returns real‑valued features.  This function is kept for compatibility with
    the legacy tests that expect a ``generate_superposition_data`` helper.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields a feature vector and a target value.

    The ``states`` key contains the raw feature vector while the ``target`` key
    contains the continuous target.  The class is intentionally lightweight
    so it can be reused with either the classical or quantum model.
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

class ClassicalFeatureExtractor(nn.Module):
    """Simple MLP that transforms raw features before they are sent to the quantum layer."""
    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QModel(nn.Module):
    """Hybrid model that combines a classical backbone with a variational quantum circuit.

    The architecture follows:
        input --> classical backbone --> quantum encoder --> variational layer
        --> measurement --> linear head --> prediction
    """
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.backbone = ClassicalFeatureExtractor(num_features)
        self.encoder = nn.Identity()  # placeholder for a future quantum encoder
        self.q_layer = None  # will be replaced by a quantum module in subclass
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Forward through classical backbone
        features = self.backbone(state_batch)
        # Pass features to quantum module (expects a subclass to implement)
        q_features = self.q_layer(features)
        return self.head(q_features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
