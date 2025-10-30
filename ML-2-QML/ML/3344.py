"""Hybrid QCNN with classical RBF kernel integration.

The model mirrors the original QCNN architecture but appends a classical
radial‑basis function (RBF) kernel layer before the final linear head.
This allows the network to learn both local convolutional patterns and
global similarity features in a single forward pass.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import sigmoid
import numpy as np

# --- Classical RBF kernel utilities -----------------------------------------

class RBFKernel(nn.Module):
    """Exponential kernel used as a plug‑in layer in the network."""
    def __init__(self, gamma: float = 1.0, trainable: bool = False) -> None:
        super().__init__()
        self.gamma = nn.Parameter(
            torch.tensor(gamma, dtype=torch.float32),
            requires_grad=trainable
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x - y||^2) per sample pair."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (B,1,D)-(1,N,D)
        dist_sq = (diff * diff).sum(dim=-1)     # (B,N)
        return torch.exp(-self.gamma * dist_sq)


# --- Hybrid QCNN model -------------------------------------------------------

class QCNNHybridModel(nn.Module):
    """
    Classical QCNN architecture augmented with a kernel layer.

    The forward pass computes:
        1. A feature map via a shallow FC stack (imitating quantum layers).
        2. A kernel similarity matrix between the batch and a fixed reference
           set (e.g. training centroids).
        3. Concatenation of the feature map and kernel vector.
        4. Final binary classification.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        kernel_dim: int = 32,
        gamma: float = 1.0,
        trainable_gamma: bool = False,
    ) -> None:
        super().__init__()
        # Feature map (mimics quantum convolution)
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Kernel layer
        self.kernel = RBFKernel(gamma, trainable=trainable_gamma)
        self.kernel_buffer = None  # will hold reference set after first forward

        # Final head (feature + kernel)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + kernel_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, D) where D == input_dim
        Returns:
            Tensor of shape (B, 1) with sigmoid activation.
        """
        # Feature extraction
        feat = self.feature_map(x)

        # Kernel similarity
        if self.kernel_buffer is None:
            # On first forward, buffer the current batch as reference
            self.kernel_buffer = x.detach()
        # Compute kernel matrix between current batch and reference buffer
        k_matrix = self.kernel(x, self.kernel_buffer)  # (B, B_ref)
        # Reduce kernel dimension by mean across reference dimension
        k_vec = k_matrix.mean(dim=1, keepdim=True)  # (B, 1)

        # Concatenate
        combined = torch.cat([feat, k_vec], dim=1)

        # Classification
        out = self.head(combined)
        return sigmoid(out)


__all__ = ["QCNNHybridModel"]
