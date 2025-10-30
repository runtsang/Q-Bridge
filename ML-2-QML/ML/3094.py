"""Hybrid classical model combining convolutional and RBF kernel feature mapping."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalRBFKernelFeature(nn.Module):
    """Maps each 2×2 patch to a feature vector using an RBF kernel against learnable prototypes."""
    def __init__(self, num_prototypes: int = 32, gamma: float = 1.0) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.gamma = gamma
        # prototypes: (num_prototypes, 4)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 4, H, W) after convolution
        """
        batch, channels, H, W = x.shape
        patches = x.view(batch, channels, -1)  # (batch, 4, H*W)
        # compute squared distance between each patch and each prototype
        diff = patches.unsqueeze(2) - self.prototypes.unsqueeze(0).unsqueeze(-1)
        # diff: (batch, 4, num_prototypes, H*W)
        dist_sq = (diff ** 2).sum(dim=1)  # (batch, num_prototypes, H*W)
        features = torch.exp(-self.gamma * dist_sq)
        return features.view(batch, -1)  # (batch, num_prototypes * H * W)

class QuanvolutionHybrid(nn.Module):
    """Classical hybrid model: conv -> RBF kernel feature mapping -> linear head."""
    def __init__(self, num_prototypes: int = 32, gamma: float = 1.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.kernel_feat = ClassicalRBFKernelFeature(num_prototypes, gamma)
        self.linear = nn.Linear(num_prototypes * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        feat = self.kernel_feat(x)
        logits = self.linear(feat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
