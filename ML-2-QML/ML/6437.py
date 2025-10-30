"""Hybrid classical classifier with optional RBF kernel and residual network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Optional

class RBFKernel(nn.Module):
    """RBF kernel feature map for a fixed set of support vectors."""
    def __init__(self, support_vectors: torch.Tensor, gamma: float = 1.0):
        super().__init__()
        self.register_buffer("support_vectors", support_vectors)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        diff = x.unsqueeze(1) - self.support_vectors.unsqueeze(0)  # (batch, n_sv, features)
        dist_sq = torch.sum(diff * diff, dim=-1)  # (batch, n_sv)
        return torch.exp(-self.gamma * dist_sq)

class ResidualBlock(nn.Module):
    """Simple 2-layer residual block."""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return x + out

class HybridClassifier(nn.Module):
    """
    Classical feed‑forward classifier that optionally prepends an RBF kernel
    feature map and stacks a residual network.
    """
    def __init__(self,
                 num_features: int,
                 depth: int,
                 use_kernel: bool = False,
                 gamma: float = 1.0,
                 support_vectors: Optional[torch.Tensor] = None):
        super().__init__()
        if use_kernel:
            if support_vectors is None:
                support_vectors = torch.randn(num_features, num_features)
            self.kernel = RBFKernel(support_vectors, gamma)
            in_dim = support_vectors.size(0)
        else:
            self.kernel = None
            in_dim = num_features

        blocks: List[nn.Module] = []
        for _ in range(depth):
            blocks.append(ResidualBlock(in_dim))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel is not None:
            x = self.kernel(x)
        x = self.body(x)
        return self.head(x)

def build_classifier_circuit(num_features: int,
                             depth: int,
                             use_kernel: bool = False,
                             gamma: float = 1.0,
                             support_vectors: Optional[torch.Tensor] = None) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a HybridClassifier and return metadata compatible with the
    original quantum build interface.
    """
    model = HybridClassifier(num_features, depth, use_kernel, gamma, support_vectors)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = list(range(2))
    return model, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
