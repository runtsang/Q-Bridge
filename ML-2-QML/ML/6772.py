"""Classical self‑attention module enhanced with RBF kernel similarity and optional quantum kernel fallback.

This module extends the original SelfAttention helper by replacing the simple dot‑product
attention with an RBF kernel similarity.  The kernel is implemented as a PyTorch module
(`Kernel`) that mirrors the interface of the quantum kernel in the QML seed.  The
`SelfAttentionKernel` class accepts a `gamma` parameter to control the width of the
Gaussian kernel and an optional `use_quantum_kernel` flag.  When set to True the
class will call an external quantum kernel routine (via a user‑provided callback)
instead of computing the similarity classically.

The implementation keeps the original `SelfAttention()` factory for backward
compatibility and exposes the same `run()` signature as the quantum version.
"""

import numpy as np
import torch
from torch import nn
from typing import Callable, Optional

class Kernel(nn.Module):
    """RBF kernel wrapper compatible with the quantum kernel interface."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x - y||^2)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class SelfAttentionKernel:
    """Classical self‑attention with kernel‑based similarity."""
    def __init__(self, embed_dim: int = 4, gamma: float = 1.0,
                 use_quantum_kernel: bool = False,
                 quantum_kernel_cb: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the embedded input.
        gamma : float
            RBF kernel width.
        use_quantum_kernel : bool
            If True, compute similarity via the provided quantum kernel callback.
        quantum_kernel_cb : Callable
            Function that accepts two numpy arrays of shape (batch, embed_dim)
            and returns a similarity matrix.
        """
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.kernel = Kernel(gamma)
        self.use_quantum_kernel = use_quantum_kernel
        self.quantum_kernel_cb = quantum_kernel_cb

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute self‑attention output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the linear transformation of queries.
        entangle_params : np.ndarray
            Parameters for the linear transformation of keys.
        inputs : np.ndarray
            Input tensor of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Attention‑weighted output of shape (batch, embed_dim).
        """
        # Linear projections
        q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                            dtype=torch.float32)
        k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                            dtype=torch.float32)
        v = torch.as_tensor(inputs, dtype=torch.float32)

        # Compute similarity matrix
        if self.use_quantum_kernel and self.quantum_kernel_cb is not None:
            # Call external quantum kernel routine
            sim = self.quantum_kernel_cb(q.numpy(), k.numpy())
            sim = torch.as_tensor(sim, dtype=torch.float32)
        else:
            # Classical RBF kernel
            sim = self.kernel(q, k)

        # Attention weights
        attn = torch.softmax(sim / np.sqrt(self.embed_dim), dim=-1)
        return (attn @ v).numpy()

def SelfAttention():
    """Factory for backward compatibility."""
    return SelfAttentionKernel(embed_dim=4)
