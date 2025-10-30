"""Classical regression dataset and hybrid kernel‑neural model."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where the target is a smooth
    function of the sum of the input features.  The data mimics the
    structure used by the quantum seed but is purely classical.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Simple torch Dataset that returns feature vectors and targets.
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

class RBFFeatureExtractor(nn.Module):
    """
    Computes RBF kernel similarities between inputs and a fixed set of
    support vectors.  The support vectors are supplied at construction
    and are treated as learnable but static in this simple example.
    """
    def __init__(self, support_vectors: torch.Tensor, gamma: float = 1.0):
        super().__init__()
        self.register_buffer("support", support_vectors)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim), support: (n_support, dim)
        diff = x.unsqueeze(1) - self.support.unsqueeze(0)   # (batch, n_support, dim)
        sq_norm = torch.sum(diff ** 2, dim=-1)              # (batch, n_support)
        return torch.exp(-self.gamma * sq_norm)             # (batch, n_support)

class QModel(nn.Module):
    """
    Hybrid classical regression model that first projects the input into
    a kernel feature space and then applies a lightweight linear head.
    """
    def __init__(self, num_features: int, support_vectors: torch.Tensor, gamma: float = 1.0):
        super().__init__()
        self.kernel_extractor = RBFFeatureExtractor(support_vectors, gamma)
        self.head = nn.Linear(support_vectors.shape[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k_features = self.kernel_extractor(x)   # (batch, n_support)
        return self.head(k_features).squeeze(-1)

__all__ = ["generate_superposition_data", "RegressionDataset", "QModel"]
