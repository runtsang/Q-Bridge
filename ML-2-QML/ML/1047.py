"""Hybrid classical kernel module with learnable width and batched support."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]


class KernalAnsatz(nn.Module):
    """Learnable RBF kernel width via a small MLP.

    The original seed used a fixed gamma.  Here we replace it with a
    neural network that maps the input feature vector to a positive
    width parameter.  This makes the kernel differentiable and
    trainable end‑to‑end.
    """

    def __init__(self, in_features: int, hidden_dim: int = 32):
        """
        Parameters
        ----------
        in_features
            Dimensionality of the input data.
        hidden_dim
            Size of the hidden layer in the MLP.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensures positivity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a batch of feature vectors to gamma values.

        Parameters
        ----------
        x
            Tensor of shape ``(batch, in_features)``.

        Returns
        -------
        gamma
            Tensor of shape ``(batch,)``.
        """
        return self.mlp(x).squeeze(-1)


class Kernel(nn.Module):
    """RBF kernel that uses :class:`KernalAnsatz` to obtain a learnable width."""

    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.ansatz = KernalAnsatz(in_features, hidden_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between two batches.

        Parameters
        ----------
        x, y
            Tensors of shape ``(batch, features)`` or ``(features,)``.

        Returns
        -------
        K
            Tensor of shape ``(len(x), len(y))`` containing the kernel values.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # Obtain gamma for each sample in x
        gamma = self.ansatz(x).unsqueeze(-1)  # (batch_x, 1)

        diff = x[:, None, :] - y[None, :, :]
        dist_sq = torch.sum(diff * diff, dim=-1)  # (batch_x, batch_y)
        return torch.exp(-gamma * dist_sq)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Evaluate the Gram matrix between two collections of feature vectors.

    Parameters
    ----------
    a, b
        Sequences of 1‑D tensors.

    Returns
    -------
    gram
        NumPy array of shape ``(len(a), len(b))``.
    """
    if not a:
        return np.empty((0, 0))
    in_features = a[0].shape[0]
    kernel = Kernel(in_features)
    a_stack = torch.stack(a)  # (len(a), features)
    b_stack = torch.stack(b)  # (len(b), features)
    return kernel(a_stack, b_stack).detach().cpu().numpy()
