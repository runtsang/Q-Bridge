import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate superposition‑based regression data.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input vector.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray
        Input features, shape ``(samples, num_features)``.
    y : np.ndarray
        Target values derived from a smooth sinusoidal function.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields superposition‑state features and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridRegression(nn.Module):
    """
    Purely classical hybrid regression model.

    The architecture is deliberately split into three blocks:
    * A residual feature extractor that expands the input dimensionality.
    * A linear head that would normally be replaced by a quantum expectation layer.
    * An optional shift bias that mirrors the shift used in the binary‑classification hybrid.
    """
    def __init__(self, num_features: int, hidden: int = 64, shift: float = 0.0):
        super().__init__()
        self.shift = shift
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        out = self.head(x).squeeze(-1)
        return out + self.shift

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
