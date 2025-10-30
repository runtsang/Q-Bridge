import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

class ThresholdedConv(nn.Module):
    """A lightweight conv filter that applies a sigmoid activation with a threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1,2,3])  # mean over spatial dimensions

class Kernel(nn.Module):
    """Classical RBF kernel used as a plug‑in for the hybrid head."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridConvKernelNet(nn.Module):
    """Hybrid network that combines classical CNN, a thresholded filter, and a kernel head."""
    def __init__(self, n_support: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # Convolutional front‑end (borrowed from QCNet)
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
        )
        # Feature extractor producing 84‑dimensional embeddings
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
        )
        # Kernel head
        self.kernel = Kernel(gamma)
        # Trainable support vectors (initialized randomly)
        self.support_vectors = nn.Parameter(torch.randn(n_support, 84))
        self.proj = nn.Linear(n_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.feature_extractor(x)  # shape (batch, 84)
        # Compute kernel similarities with support vectors
        sims = torch.stack([self.kernel(x, sv) for sv in self.support_vectors], dim=1).squeeze(-1)  # (batch, n_support)
        logits = self.proj(sims)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["ThresholdedConv", "Kernel", "HybridConvKernelNet"]
