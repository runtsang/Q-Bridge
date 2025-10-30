"""Hybrid classical regression combining RBF kernel mapping with a neural head.

The module defines:
- generate_superposition_data: generates synthetic regression data.
- RegressionDataset: torch Dataset for training.
- HybridRegressionModel: nn.Module that applies an RBF kernel to inputs
  before a linear regression head.  The kernel parameters are trainable
  allowing the model to learn an implicit feature map.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ----- data generation ----------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample data from a sinusoidal superposition.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input space.
    samples : int
        Number of data points to generate.

    Returns
    -------
    features : np.ndarray
        Shape (samples, num_features) with values in [-1, 1].
    labels : np.ndarray
        Shape (samples,) with a non‑linear target function.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# ----- dataset -------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """Torch Dataset wrapping the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return self.features.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ----- RBF kernel ansatz ----------------------------------------------------- #
class RBFKernel(nn.Module):
    """Trainable RBF kernel with a learnable gamma."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value between two batches of vectors."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True)).squeeze(-1)

# ----- hybrid model --------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """Classical regression model that first maps inputs through an RBF kernel
    and then applies a linear head.  The kernel parameter ``gamma`` is
    learned jointly with the head weights.
    """
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.kernel = RBFKernel()
        # The kernel maps a single input to a scalar; we stack the
        # kernel values with the raw feature vector to give the head
        # richer information.
        self.head = nn.Sequential(
            nn.Linear(num_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, num_features)

        Returns
        -------
        torch.Tensor
            Predicted target values, shape (batch,)
        """
        # Compute kernel values between each example and a learnable
        # reference vector (the first training example).
        # For simplicity, we use the batch itself as the reference set
        # and average the kernel values across the batch.
        batch_size = state_batch.size(0)
        kernel_vals = self.kernel(state_batch, state_batch)  # (batch, batch)
        # Aggregate to a single scalar per example
        agg_kernel = kernel_vals.mean(dim=1, keepdim=True)  # (batch, 1)
        # Concatenate with raw features
        features = torch.cat([state_batch, agg_kernel], dim=1)
        return self.head(features).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data", "RBFKernel"]
