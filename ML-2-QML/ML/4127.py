"""Classical implementation of a hybrid quantum‑inspired network.

The model starts with a classical quanvolution filter that mimics the
quantum 2×2 patch kernel.  A linear layer maps the flattened features
to either a log‑softmax for binary classification or a scalar for
regression.  The same class can be used for both tasks by setting
`mode` at construction time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class QuanvolutionFilter(nn.Module):
    """2×2 convolutional filter that emulates the quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out.view(x.size(0), -1)

class HybridClassifierHead(nn.Module):
    """Log‑softmax head for binary classification."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.linear(x), dim=-1)

class HybridRegressionHead(nn.Module):
    """Linear head for regression."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

class HybridNet(nn.Module):
    """
    Classical hybrid network that first applies a quanvolution filter
    and then a head chosen at construction time.
    """
    def __init__(self, mode: str = "classification") -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.mode = mode
        if mode == "classification":
            self.head = HybridClassifierHead(4 * 14 * 14)
        elif mode == "regression":
            self.head = HybridRegressionHead(4 * 14 * 14)
        else:
            raise ValueError("mode must be 'classification' or'regression'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        return self.head(features)

class RegressionDataset(Dataset):
    """Dataset that generates superposition‑like data for regression."""
    def __init__(self, samples: int = 10000, num_features: int = 10) -> None:
        self.x, self.y = self._generate(samples, num_features)

    @staticmethod
    def _generate(samples: int, num_features: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"features": torch.tensor(self.x[idx]), "target": torch.tensor(self.y[idx])}

class BinaryClassificationDataset(Dataset):
    """Dummy binary classification dataset using random labels."""
    def __init__(self, samples: int = 10000, img_size: int = 28) -> None:
        self.samples = samples
        self.img_size = img_size
        self.data = np.random.randn(samples, 1, img_size, img_size).astype(np.float32)
        self.labels = np.random.randint(0, 2, size=(samples,)).astype(np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        return self.samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"image": torch.tensor(self.data[idx]), "label": torch.tensor(self.labels[idx])}

__all__ = [
    "HybridNet",
    "RegressionDataset",
    "BinaryClassificationDataset",
]
