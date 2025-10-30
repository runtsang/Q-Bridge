from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

def generate_classical_superposition(
    num_features: int,
    samples: int,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset where the target is a noisy
    sinusoidal function of the sum of the input features."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + noise * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """PyTorch dataset that returns feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_classical_superposition(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegression(nn.Module):
    """Classical neural network that mimics the architecture of the quantum
    regression head but uses only dense layers.  The final layer is a
    differentiable logistic transform that can be replaced by a quantum
    expectation in the QML variant."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, num_features)
        Returns
        -------
        torch.Tensor
            Shape (batch,)
        """
        x = self.net(state_batch)
        return self.sigmoid(x).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_classical_superposition"]
