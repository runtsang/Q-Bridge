import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition data for regression."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridSamplerRegressor(nn.Module):
    """
    Classical hybrid model: a sampler network followed by a regression head.
    The sampler performs a 2→4→2 softmax mapping, while the head maps the sampled
    probabilities to a scalar regression target.  This mirrors the SamplerQNN
    architecture and adds a regression component inspired by QuantumRegression.
    """
    def __init__(self, num_features: int, num_classes: int = 2):
        super().__init__()
        self.sampler = nn.Sequential(
            nn.Linear(num_features, 4),
            nn.Tanh(),
            nn.Linear(4, num_classes),
        )
        self.head = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.sampler(inputs), dim=-1)
        return self.head(probs).squeeze(-1)

__all__ = ["HybridSamplerRegressor", "RegressionDataset", "generate_superposition_data"]
