"""
Classical regression model that now supports a dual‑branch architecture
and a richer dataset generator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Utility: richer generator
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random feature vectors and a target that depends on both
    their sum and a higher‑order interaction term.
    """
    rng = np.random.default_rng(random_state)
    x = rng.standard_normal(size=(samples, num_features), dtype=np.float32)
    # target = sin(sum(x)) + 0.2 * cos(np.dot(x, x))
    y = np.sin(x.sum(axis=1)) + 0.2 * np.cos(np.sum(x * x, axis=1))
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    A PyTorch dataset that returns a dictionary containing the feature vector
    under the key ``states`` and the corresponding target under ``target``.
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
class QuantumRegression__gen140(nn.Module):
    """
    Dual‑branch regression model that processes raw features through a
    classical MLP and a quantum‑inspired feature map, then fuses both
    outputs linearly.
    """
    def __init__(self, num_features: int, quantum_features: int | None = None):
        super().__init__()
        self.num_features = num_features

        # Classical branch
        self.classical_net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Quantum‑inspired branch
        if quantum_features is None:
            quantum_features = num_features
        self.quantum_map = nn.Sequential(
            nn.Linear(num_features, quantum_features),
            nn.Tanh(),
            nn.Linear(quantum_features, quantum_features),
            nn.Sigmoid(),
        )
        self.quantum_head = nn.Linear(quantum_features, 1)

        # Fusion layer
        self.fusion = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, num_features)``.
        """
        # Classical output
        c_out = self.classical_net(x).squeeze(-1)

        # Quantum‑inspired output
        q_features = self.quantum_map(x)
        q_out = self.quantum_head(q_features).squeeze(-1)

        # Fuse
        fused = self.fusion(torch.stack([c_out, q_out], dim=1)).squeeze(-1)
        return fused

__all__ = ["QuantumRegression__gen140", "RegressionDataset", "generate_superposition_data"]
