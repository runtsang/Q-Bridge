import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

class RBFAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class RBFKernel(nn.Module):
    """Wrapper around RBFAnsatz."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.ansatz = RBFAnsatz(gamma)
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

class HybridQuanvolutionFilter(nn.Module):
    """Classical filter that extracts 2×2 patches and maps them through a learnable RBF kernel."""
    def __init__(self, patch_size: int = 2, stride: int = 2, in_channels: int = 1, out_channels: int = 4, gamma: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride)
        self.kernel = RBFKernel(gamma)
        # learnable center in feature space
        self.center = nn.Parameter(torch.randn(out_channels * 14 * 14))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)  # (B, C, H, W)
        flat = features.view(features.size(0), -1)  # (B, D)
        # compute kernel between each sample and the learned center
        k = self.kernel(flat, self.center)
        return k.unsqueeze(1)  # (B, 1)

class HybridQuanvolution(nn.Module):
    """Hybrid classifier that uses the above filter and a linear head."""
    def __init__(self, num_classes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.filter = HybridQuanvolutionFilter(gamma=gamma)
        self.linear = nn.Linear(1, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridQuanvolution", "kernel_matrix"]
