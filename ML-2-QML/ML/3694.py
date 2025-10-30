"""Hybrid kernel combining classical RBF and neural sampler for feature mapping."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class SamplerQNN(nn.Module):
    """Simple feed‑forward sampler network."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridKernelMethod:
    """Hybrid kernel that blends an RBF kernel with a neural‑sampler similarity."""
    def __init__(self, gamma: float = 1.0, weight_rbf: float = 0.5, weight_sampler: float = 0.5) -> None:
        self.rbf = KernalAnsatz(gamma)
        self.sampler = SamplerQNN()
        self.weight_rbf = weight_rbf
        self.weight_sampler = weight_sampler

    def _sampler_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Similarity from sampler outputs (cosine / dot product)."""
        sx = self.sampler(x).view(-1)
        sy = self.sampler(y).view(-1)
        return torch.dot(sx, sy) / (torch.norm(sx) * torch.norm(sy) + 1e-12)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return weighted mixture of RBF and sampler similarity."""
        rbf_val = self.rbf(x, y)
        samp_val = self._sampler_similarity(x, y)
        return self.weight_rbf * rbf_val + self.weight_sampler * samp_val

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sequences."""
        mat = torch.stack([torch.stack([self.forward(x, y).squeeze() for y in b]) for x in a])
        return mat.detach().cpu().numpy()

__all__ = ["HybridKernelMethod"]
