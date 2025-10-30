import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic regression dataset where the target is a smooth function
    of the sum of input features.  This mirrors the quantum‑style data but
    remains fully classical.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with keys'states' (feature tensor)
    and 'target' (label tensor).  Compatible with the quantum version.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridFCLRegression(nn.Module):
    """
    Classical fully‑connected regression head that mirrors the quantum circuit
    structure: linear → tanh → linear.  The architecture is deliberately
    shallow to keep the comparison fair.
    """
    def __init__(self, n_features: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Linear(n_features, hidden_dim)
        self.activation = nn.Tanh()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (batch, n_features) → (batch, 1)
        """
        x = self.activation(self.encoder(x))
        return self.head(x).squeeze(-1)

__all__ = ["HybridFCLRegression", "RegressionDataset", "generate_superposition_data"]
