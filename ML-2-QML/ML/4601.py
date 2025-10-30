"""Hybrid classical model combining a convolutional pre‑processor and a linear head.
The filter emulates the original quanvolution idea but is fully classical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

# -------------------------------------------------------------
# Data utilities (classical)
# -------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy regression dataset with a sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# -------------------------------------------------------------
# Kernel utilities
# -------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """Radial‑basis‑function (RBF) kernel implemented as a PyTorch module."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper that exposes a single‑call interface for the RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.ansatz(x.view(1, -1), y.view(1, -1)).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0):
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# -------------------------------------------------------------
# Classical quanvolution filter
# -------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    """Extract 2×2 patches with a 2‑stride conv; purely classical."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

# -------------------------------------------------------------
# Hybrid classifier
# -------------------------------------------------------------
class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model that mirrors the behaviour of the original Quanvolution
    pipeline but replaces the quantum kernel with a learnable linear layer.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.head = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.qfilter(x)
        logits = self.head(feats)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
