"""Hybrid classical kernel with attention‑based feature mapping."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from torch import nn

class SelfAttention(nn.Module):
    """Classical self‑attention transformer used as a feature extractor."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # Reshape parameters
        rot = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        ent = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        query = inputs @ rot
        key = inputs @ ent
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class KernalAnsatz(nn.Module):
    """Wraps a self‑attention feature map followed by an RBF kernel."""
    def __init__(self, gamma: float = 1.0, embed_dim: int = 4):
        super().__init__()
        self.gamma = gamma
        self.embed_dim = embed_dim
        self.attention = SelfAttention(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        # Map inputs through attention
        x_att = self.attention(rotation_params, entangle_params, x)
        y_att = self.attention(rotation_params, entangle_params, y)
        diff = x_att - y_att
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Convenience wrapper that exposes a single forward signature."""
    def __init__(self, gamma: float = 1.0, embed_dim: int = 4):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y, rotation_params, entangle_params).squeeze()

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    kernel = Kernel(gamma, embed_dim=rotation_params.shape[0] // 3)
    return np.array(
        [[kernel(x, y, rotation_params, entangle_params).item() for y in b] for x in a]
    )

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
