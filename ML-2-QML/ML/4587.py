import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalKernel(nn.Module):
    """Classical RBF kernel suitable for hybrid models."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        # x: (bsz, dim), support: (n_support, dim)
        diff = x.unsqueeze(1) - support.unsqueeze(0)
        norm = torch.sum(diff * diff, dim=2)
        return torch.exp(-self.gamma * norm)

class HybridNAT(nn.Module):
    """Hybrid classical‑quantum inspired model.

    Combines:
      • 2‑layer CNN feature extractor (like QFCModel)
      • A kernel layer that computes similarity to a set of learnable support vectors
      • Final linear classifier.
    """
    def __init__(self,
                 n_support: int = 10,
                 gamma: float = 1.0,
                 n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 16 * 7 * 7
        self.kernel = ClassicalKernel(gamma)
        self.support = nn.Parameter(torch.randn(n_support, self.flatten_dim))
        self.linear = nn.Linear(n_support, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x).view(bsz, -1)
        k = self.kernel(feat, self.support)
        out = self.linear(k)
        return out

__all__ = ["HybridNAT"]
