import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 1‑D superposition‑like regression dataset."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Classic regression dataset compatible with the quantum counterpart."""
    def __init__(self, samples: int, num_features: int):
        self.x, self.y = generate_superposition_data(num_features, samples)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.x[idx], dtype=torch.float32),
            "target": torch.tensor(self.y[idx], dtype=torch.float32),
        }

class QuanvolutionFilter(nn.Module):
    """A lightweight 2‑D convolution that mimics the quanvolution idea."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)
        return feats.view(x.size(0), -1)

class HybridRegression(nn.Module):
    """Classical regression head that can optionally use a quanvolution filter."""
    def __init__(self, num_features: int, use_quanvolution: bool = False, in_channels: int = 1):
        super().__init__()
        self.use_quanvolution = use_quanvolution
        if use_quanvolution:
            self.encoder = QuanvolutionFilter(in_channels=in_channels)
            self.feature_dim = 4 * 14 * 14  # assuming 28x28 input
        else:
            self.encoder = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            self.feature_dim = 16
        self.head = nn.Linear(self.feature_dim, 1)
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            feats = self.encoder(states)
        else:
            feats = self.encoder(states)
        return self.head(feats).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
