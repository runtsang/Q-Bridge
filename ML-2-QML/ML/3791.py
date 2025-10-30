import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    """Generate either 1‑D or 2‑D synthetic regression data."""
    def __init__(self, num_features: int = 10, samples: int = 10000, is_2d: bool = False):
        self.is_2d = is_2d
        if is_2d:
            # 2‑D data: random 28×28 patches, target = summed intensity
            self.data = torch.randn(samples, 1, 28, 28)
            self.target = self.data.view(samples, -1).sum(dim=1)
        else:
            self.data = torch.randn(samples, num_features)
            angles = self.data.sum(dim=1)
            self.target = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    def __len__(self):  # type: ignore[override]
        return len(self.data)
    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"states": self.data[idx], "target": self.target[idx]}

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolutional filter mimicking a quanvolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)

class QuantumHybridRegression(nn.Module):
    """
    Classical hybrid regression model.
    For 1‑D data: a simple MLP head.
    For 2‑D data: a quanvolution filter followed by a linear head.
    """
    def __init__(self, num_features: int = 10):
        super().__init__()
        self.is_2d = False
        self.num_features = num_features
        # 1‑D head
        self.head_1d = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # 2‑D quanvolution
        self.qfilter = QuanvolutionFilter()
        self.head_2d = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def set_2d(self, flag: bool = True) -> None:
        """Toggle between 1‑D and 2‑D mode."""
        self.is_2d = flag
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if self.is_2d:
            features = self.qfilter(state_batch)
            return self.head_2d(features).squeeze(-1)
        else:
            return self.head_1d(state_batch).squeeze(-1)

__all__ = ["RegressionDataset", "QuantumHybridRegression", "QuanvolutionFilter"]
