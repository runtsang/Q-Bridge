from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with a sinusoidal target."""
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

class RBFKernelLayer(nn.Module):
    """Classical RBF kernel feature map."""
    def __init__(self, num_centroids: int, dim: int, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.centroids = nn.Parameter(torch.randn(num_centroids, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.centroids.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=-1)
        return torch.exp(-self.gamma * dist_sq)

class HybridRegressionModel(nn.Module):
    """Combined CNN + FC + RBF kernel regression model."""
    def __init__(self,
                 num_features: int = 4,
                 num_centroids: int = 8,
                 gamma: float = 1.0,
                 input_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_features),
        )
        self.norm = nn.BatchNorm1d(num_features)
        self.kernel_layer = RBFKernelLayer(num_centroids, num_features, gamma)
        self.head = nn.Linear(num_centroids, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        fc_out = self.fc(flat)
        normed = self.norm(fc_out)
        k_feat = self.kernel_layer(normed)
        out = self.head(k_feat)
        return out.squeeze(-1)

def kernel_matrix(a: list[torch.Tensor],
                  b: list[torch.Tensor],
                  gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the RBF kernel."""
    a_tensor = torch.stack(a)
    b_tensor = torch.stack(b)
    diff = a_tensor.unsqueeze(1) - b_tensor.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=-1)
    return torch.exp(-gamma * dist_sq).numpy()

__all__ = ["generate_superposition_data",
           "RegressionDataset",
           "HybridRegressionModel",
           "RBFKernelLayer"]
