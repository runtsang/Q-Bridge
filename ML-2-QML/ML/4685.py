import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBFKernel(nn.Module):
    """Classical radial‑basis kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x-y||²)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class ClassicalQuanvolutionFilter(nn.Module):
    """Simple 2‑D convolution that mimics the quantum filter structure."""
    def __init__(self, in_ch: int = 1, out_ch: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model: classical filter → RBF kernel → fully‑connected head.
    Mirrors the quantum implementation for direct comparison.
    """
    def __init__(self, num_classes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.filter = ClassicalQuanvolutionFilter()
        self.kernel = RBFKernel(gamma)
        # Feature size: 4 * 14 * 14 = 784 (same as quantum filter output)
        self.fc = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patch‑based features
        feat = self.filter(x)                     # (batch, 784)
        # Compute diagonal of RBF kernel matrix (self‑similarity)
        diag = torch.diag(self.kernel(feat, feat)).unsqueeze(1)  # (batch, 1)
        # Concatenate kernel statistic with raw features
        combined = torch.cat([feat, diag], dim=1)  # (batch, 785)
        logits = self.fc(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = ["RBFKernel", "ClassicalQuanvolutionFilter", "QuanvolutionHybrid"]
