import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using a superposition of sinusoidal components.
    The function now supports optional polynomial feature expansion and Gaussian noise.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    poly = np.hstack([x, x ** 2, np.sin(x), np.cos(x)])
    angles = poly.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles) + 0.05 * np.random.randn(samples)
    return poly.astype(np.float32), y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and regression targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """A lightweight MLP for regression."""
    def __init__(self, num_features: int, hidden_dim: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(num_features, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
