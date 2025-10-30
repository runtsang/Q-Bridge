"""Hybrid classical implementation combining fully connected layer
and quantum-inspired regression.

This module merges the simple linear layer from the original
`FCL` seed with the regression architecture of the
`QuantumRegression` seed.  The resulting class can be used
as a drop‑in replacement for the classical part of the
original project while still exposing the same ``run``
interface.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data used in the quantum seed."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset wrapping the synthetic data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridFCL(nn.Module):
    """Classical hybrid layer combining a linear encoder with a
    small feed‑forward head.  The ``run`` method accepts a list of
    parameters that are fed through the encoder and head, mimicking
    the API of the original `FCL` example.
    """
    def __init__(self, n_features: int = 1, n_hidden: int = 32, n_wires: int = 2) -> None:
        super().__init__()
        # Encode the raw parameters into a qubit‑like feature space
        self.encoder = nn.Linear(n_features, n_wires)
        # Optional quantum‑inspired feature map
        self.quantum_features = nn.Sequential(
            nn.Linear(n_wires, 16),
            nn.ReLU(),
        )
        # Classical regression head
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.encoder(x)
        x = self.quantum_features(x)
        return self.head(x).squeeze(-1)

    def run(self, thetas: list[float]) -> np.ndarray:
        """Run a single forward pass on the supplied parameters."""
        theta_tensor = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.forward(theta_tensor)
        return out.detach().cpu().numpy()

def FCL() -> HybridFCL:
    """Convenience factory matching the original API."""
    return HybridFCL()

__all__ = ["HybridFCL", "RegressionDataset", "generate_superposition_data", "FCL"]
