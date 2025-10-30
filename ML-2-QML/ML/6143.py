"""QuanvolutionHybrid – classical version using a 2×2 convolution followed by an RBF kernel feature map."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RBFAnsatz(nn.Module):
    """Radial‑basis kernel implemented as a differentiable module."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class RBFKernel(nn.Module):
    """Wraps :class:`RBFAnsatz` to act like a kernel between two batches."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = RBFAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Broadcast to compute pairwise kernel
        x = x.unsqueeze(1)  # (B,1,N)
        y = y.unsqueeze(0)  # (1,B,N)
        return self.ansatz(x, y).squeeze(-1)


class QuanvolutionHybridClassifier(nn.Module):
    """
    Classical hybrid that first extracts 2×2 patches via a 2‑D convolution,
    then applies an RBF kernel to each patch before feeding the flattened
    representation into a linear classifier.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        # 1→4 channel conv to mimic the original quanvolution filter
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.kernel = RBFKernel(gamma)
        self.fc = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches
        patches = self.conv(x)          # (B,4,14,14)
        patches = patches.view(patches.size(0), 4, -1).transpose(1, 2)  # (B,196,4)
        # Apply RBF kernel to each patch
        B, N, C = patches.shape
        patches_k = torch.empty(B, N, C, device=patches.device)
        for i in range(N):
            patches_k[:, i] = self.kernel(patches[:, i], patches[:, i])
        # Flatten and classify
        features = patches_k.view(B, -1)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybridClassifier"]
