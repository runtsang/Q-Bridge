"""Hybrid classical self‑attention with quantum‑inspired kernel and regression head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Sequence

# --- Data utilities (from QuantumRegression seed) ---
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --- Kernel utilities (from QuantumKernelMethod seed) ---
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --- Hybrid Self‑Attention Model (classical) ---
class HybridSelfAttention(nn.Module):
    """
    A classical self‑attention module that uses a quantum‑inspired RBF kernel
    to compute attention weights.  The module is fully differentiable and
    can be trained end‑to‑end as a regression head.
    """

    def __init__(self, embed_dim: int, kernel_gamma: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_gamma = kernel_gamma

        # Linear maps for query/key/value
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Attention scoring via RBF kernel
        self.kernel = Kernel(gamma=kernel_gamma)

        # Regression head
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Predicted scalar per sample: shape (batch,).
        """
        q = self.query(inputs)          # (B, D)
        k = self.key(inputs)            # (B, D)
        v = self.value(inputs)          # (B, D)

        # Compute pairwise RBF kernel similarities
        scores = self.kernel(q, k)       # (B, B) after broadcasting
        weights = torch.softmax(scores, dim=-1)  # (B, B)

        # Weighted sum of values
        attn_out = weights @ v          # (B, D)

        # Regression prediction
        return self.head(attn_out).squeeze(-1)

__all__ = ["HybridSelfAttention", "RegressionDataset", "generate_superposition_data"]
