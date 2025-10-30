"""
Hybrid classical kernel module with optional variational embedding and hyper‑parameter tuning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Sequence, Optional

class VariationalEmbedding(nn.Module):
    """
    A lightweight trainable embedding that maps d‑dimensional inputs to a lower dimensional space.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class KernalAnsatz(nn.Module):
    """
    RBF kernel with optional variational embedding.
    """
    def __init__(self, gamma: float = 1.0, embedding: Optional[nn.Module] = None):
        super().__init__()
        self.gamma = gamma
        self.embedding = embedding

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
            y = self.embedding(y)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """
    Wrapper that exposes a ``forward`` method compatible with the original API.
    """
    def __init__(self, gamma: float = 1.0, embedding: Optional[nn.Module] = None):
        super().__init__()
        self.ansatz = KernalAnsatz(gamma, embedding)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The original implementation reshaped inputs to 1‑D vectors.
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  embedding: Optional[nn.Module] = None) -> np.ndarray:
    """
    Compute the Gram matrix between two collections of samples.
    """
    kernel = Kernel(gamma, embedding)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def grid_search_gamma(a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      gammas: Sequence[float]) -> tuple[float, np.ndarray]:
    """
    Simple grid search over *gammas* using the Frobenius norm of the difference
    between the kernel matrix and a target matrix (e.g., identity).
    Returns the best gamma and the corresponding Gram matrix.
    """
    best_gamma = gammas[0]
    best_km = kernel_matrix(a, b, best_gamma)
    best_err = np.linalg.norm(best_km - np.eye(len(a)))
    for g in gammas[1:]:
        km = kernel_matrix(a, b, g)
        err = np.linalg.norm(km - np.eye(len(a)))
        if err < best_err:
            best_err = err
            best_gamma = g
            best_km = km
    return best_gamma, best_km

def train_embedding(a: Sequence[torch.Tensor],
                    b: Sequence[torch.Tensor],
                    out_dim: int,
                    epochs: int = 200,
                    lr: float = 1e-3) -> nn.Module:
    """
    Train a simple variational embedding so that the induced kernel aligns
    with an identity target matrix. This is a toy example that demonstrates
    how a trainable embedding can be learned in a differentiable manner.
    """
    in_dim = a[0].shape[-1]
    emb = VariationalEmbedding(in_dim, out_dim)
    opt = Adam(emb.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        km = kernel_matrix(a, b, embedding=emb)
        target = np.eye(len(a))
        loss = torch.tensor(np.linalg.norm(km - target), dtype=torch.float32, requires_grad=True)
        loss.backward()
        opt.step()
    return emb

__all__ = ["VariationalEmbedding", "KernalAnsatz", "Kernel", "kernel_matrix",
           "grid_search_gamma", "train_embedding"]
