"""Unified classical regression model with a bias‑shifted sigmoid head.

The module defines a shared dataset class, a generator that produces
synthetic superposition states, and a classical MLP that uses
a learnable shift in its final activation.  The shift is
implemented via a custom autograd function to keep gradient
flow intact.  The design mirrors the quantum reference
while providing a clean, pure‑Python training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

__all__ = ["RegressionDataset", "generate_superposition_data", "ClassicalRegression"]

# --------------------------------------------------------------------------- #
# Dataset and data‑generation utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
    noise: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics the quantum superposition
    model used in the quantum reference.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that yields feature vectors and target values."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Custom autograd function for a bias‑shifted sigmoid
# --------------------------------------------------------------------------- #
class ShiftedSigmoid(torch.autograd.Function):
    """Differentiable sigmoid with a learnable shift parameter."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: torch.Tensor):
        ctx.save_for_backward(inputs, shift)
        return torch.sigmoid(inputs + shift)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, shift = ctx.saved_tensors
        sigmoid = torch.sigmoid(inputs + shift)
        grad_inputs = grad_output * sigmoid * (1 - sigmoid)
        grad_shift = grad_output * sigmoid * (1 - sigmoid)
        return grad_inputs, grad_shift

# --------------------------------------------------------------------------- #
# Classical regression head with a learnable bias shift
# --------------------------------------------------------------------------- #
class ClassicalRegression(nn.Module):
    """MLP with a bias‑shifted sigmoid head for regression."""
    def __init__(self, num_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        return ShiftedSigmoid.apply(logits, self.shift)
