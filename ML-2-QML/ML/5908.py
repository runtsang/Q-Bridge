"""Hybrid classical regression model that augments features with a quantum-inspired sampler.

The module defines a dataset, a sampler network, and a regression model that concatenates
classical encoder features with outputs from the sampler.  This design allows the
classical network to benefit from quantum-inspired feature maps while remaining fully
trainable on a CPU/GPU.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using a simple sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class HybridRegressionDataset(Dataset):
    """Dataset that returns feature vectors and continuous targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SamplerQNN(nn.Module):
    """Lightweight neural sampler that maps a 2‑dimensional input to a probability distribution."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class HybridRegressionModel(nn.Module):
    """Classical regression model that concatenates encoder features with sampler outputs."""
    def __init__(self, num_features: int, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
        )
        self.sampler = SamplerQNN()
        self.head = nn.Linear((hidden//2) + 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        encoded = self.encoder(state_batch)
        sampled = self.sampler(state_batch[:, :2])  # use first two dims as sampler input
        features = torch.cat([encoded, sampled], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "HybridRegressionDataset", "generate_superposition_data", "SamplerQNN"]
