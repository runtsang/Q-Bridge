import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int, noise_std: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where each feature vector is sampled uniformly
    from [-1, 1] and the target is a smooth non‑linear function of the sum of features.
    Adds Gaussian noise for realism.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += np.random.normal(0.0, noise_std, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    PyTorch dataset that yields a dictionary with keys ``states`` and ``target``.
    ``states`` are the feature vectors and ``target`` is the regression label.
    """
    def __init__(self, samples: int, num_features: int, noise_std: float = 0.05):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise_std)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RegressionModel(nn.Module):
    """
    Dropout‑regularised feed‑forward regressor.
    Supports configurable hidden layers and dropout probability.
    """
    def __init__(self, num_features: int, hidden_dims: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)

__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
