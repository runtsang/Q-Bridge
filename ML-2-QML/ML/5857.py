import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noisy superposition dataset.
    The function mirrors the first seed but adds Gaussian noise to the labels
    to emulate measurement uncertainty."""
    rng = np.random.default_rng()
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y += rng.normal(scale=0.05, size=y.shape)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a dict with both state and target tensors."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ClassicalMLP(nn.Module):
    """A small MLP used as the classical branch."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class UnifiedRegressionML(nn.Module):
    """Hybrid model that keeps a classical MLP and a quantum head."""
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.classical = ClassicalMLP(num_features)
        self.num_wires = num_wires

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.classical(states)

__all__ = ["generate_superposition_data", "RegressionDataset", "ClassicalMLP", "UnifiedRegressionML"]
