import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from typing import Iterable

class HybridFCL(nn.Module):
    """
    Hybrid fully connected layer for regression tasks.
    Combines ideas from the classical FCL example and the
    regression model in the second seed.
    """
    def __init__(self, n_features: int = 1, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Linear(n_features, hidden)
        self.activation = nn.Tanh()
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for training."""
        return self.head(self.activation(self.encoder(x)))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the quantum 'run' interface.
        Treats the input thetas as a batch of parameters for the
        first linear layer and returns the mean tanh output.
        """
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.activation(self.encoder(x)).mean(dim=0)
        return out.detach().cpu().numpy()

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.
    Uses the same logic as the QML seed but returns real-valued
    features suitable for the classical model.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset wrapping the synthetic data.
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

__all__ = ["HybridFCL", "RegressionDataset", "generate_superposition_data"]
