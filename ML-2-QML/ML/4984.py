"""Hybrid classical convolution module with self‑attention and RBF kernel similarity.

This module merges the classical Conv filter, a self‑attention mechanism,
and an RBF kernel similarity component.  It is fully compatible with the
original `Conv` interface used in the quantum experiments, but replaces
the quantum filter with a powerful classical counterpart.

The module also exposes the same regression utilities found in the
`QuantumRegression.py` reference, enabling end‑to‑end classical training
pipelines.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample data for regression."""
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

class Conv(nn.Module):
    """Hybrid classical convolutional filter.

    Parameters
    ----------
    kernel_size : int
        Size of the 2‑D convolution filter.
    gamma : float
        RBF kernel bandwidth.
    prototype : torch.Tensor, optional
        Reference vector for the kernel similarity.  If ``None``
        a zero vector is used.
    """
    def __init__(
        self,
        kernel_size: int = 3,
        gamma: float = 1.0,
        prototype: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.gamma = gamma

        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Buffers that will be created on the first forward pass
        self.prototype: torch.Tensor | None = prototype

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Shape ``(batch, 1, H, W)`` – 2‑D data to be filtered.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)`` – regression output.
        """
        # Convolution
        out = self.conv(data)          # (batch, 1, H', W')
        out = out.view(out.size(0), -1)  # (batch, N)

        # Create prototype and head on demand
        if self.prototype is None:
            self.register_buffer("prototype", torch.zeros(out.size(1), dtype=torch.float32))
        if not hasattr(self, "head"):
            self.add_module("head", nn.Linear(out.size(1), 1))

        # Self‑attention on the flattened feature map
        query = out
        key   = out
        value = out
        scores = torch.softmax(query @ key.T / np.sqrt(out.size(1)), dim=-1)
        attn_out = scores @ value  # (batch, N)

        # RBF kernel similarity with prototype
        diff = attn_out - self.prototype
        rbf = torch.exp(-self.gamma * torch.sum(diff * diff, dim=1, keepdim=True))

        # Combine and produce output
        out = self.head(attn_out + rbf)
        return out.squeeze(-1)

__all__ = [
    "Conv",
    "generate_superposition_data",
    "RegressionDataset",
]
