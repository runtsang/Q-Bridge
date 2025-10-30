"""Enhanced classical regression combining self‑attention, RBF kernel and a dense head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation & dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Create superposition‑style data with periodic labels."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset exposing ``states`` and ``target`` tensors."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical building blocks
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Single‑head self‑attention expressed as a trainable linear layer."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        scores = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ x

class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel as a differentiable layer."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * (diff ** 2).sum(-1))

# --------------------------------------------------------------------------- #
# Composite regression model
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(nn.Module):
    """
    Classical regression model that mimics the quantum workflow:
        1. Self‑attention feature extraction
        2. RBF kernel mapping
        3. Dense regression head
    """
    def __init__(self, embed_dim: int = 32, depth: int = 2, gamma: float = 1.0):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.kernel = RBFKernel(gamma)

        # Dense head operating on kernel‑transformed features
        self.head = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # 1. Attention
        attn = self.attention(states)

        # 2. Kernel mapping with the batch as reference set
        kernel_features = self.kernel(attn, attn).mean(dim=1, keepdim=True)

        # 3. Regression head
        return self.head(kernel_features).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
