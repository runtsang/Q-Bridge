"""Hybrid classical kernel module with learnable RBF parameters and kernel matrix utilities."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

__all__ = ["QuantumKernelMethod__gen444", "kernel_matrix"]

class QuantumKernelMethod__gen444(nn.Module):
    """
    Classical RBF kernel with per‑feature learnable width parameters.

    The original seed implemented a single scalar gamma; here we expose a vector of gammas
    that can be learned during training.  This allows the model to adapt the kernel
    bandwidth to each feature dimension, improving expressivity for high‑dimensional data.
    """

    def __init__(self, gamma: Union[float, Sequence[float]] = 1.0, learnable: bool = False) -> None:
        super().__init__()
        # Convert scalar to vector
        if isinstance(gamma, (float, int)):
            gamma = [float(gamma)]
        gamma = torch.as_tensor(gamma, dtype=torch.float32)
        if learnable:
            self.gamma = Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value between two batches of vectors.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch_x, features)
        y : torch.Tensor
            Tensor of shape (batch_y, features)

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (batch_x, batch_y)
        """
        # Ensure shapes
        x = x.unsqueeze(1)  # (batch_x, 1, features)
        y = y.unsqueeze(0)  # (1, batch_y, features)
        diff = x - y  # broadcast to (batch_x, batch_y, features)
        # Weighted squared distance
        sq_dist = torch.sum((diff ** 2) * self.gamma, dim=-1)
        return torch.exp(-sq_dist)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: Union[float, Sequence[float]] = 1.0,
                  learnable: bool = False) -> np.ndarray:
    """
    Compute Gram matrix between two lists of tensors using the learnable RBF kernel.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        List of tensors, each of shape (features,)
    b : Sequence[torch.Tensor]
        List of tensors, each of shape (features,)
    gamma : Union[float, Sequence[float]], default=1.0
        Width parameters for the RBF kernel.
    learnable : bool, default=False
        Whether to create a learnable kernel or a fixed one.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b))
    """
    kernel = QuantumKernelMethod__gen444(gamma=gamma, learnable=learnable)
    # Convert list to batch tensors
    a_batch = torch.stack(a)  # (len(a), features)
    b_batch = torch.stack(b)  # (len(b), features)
    with torch.no_grad():
        mat = kernel(a_batch, b_batch).cpu().numpy()
    return mat
