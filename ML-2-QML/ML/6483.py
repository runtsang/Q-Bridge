"""Classical QCNN-inspired network with optional RBF kernel feature.

The module reproduces the fully‑connected stack from the original QCNN
seed and augments it with a lightweight RBF kernel that can be used
in downstream kernel‑based classifiers.  It is purely classical
(NumPy/PyTorch) and can be dropped into any PyTorch training loop.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class RBFKernel(nn.Module):
    """Radial basis function kernel implemented as a PyTorch module."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QCNNHybrid(nn.Module):
    """
    Classical QCNN‑inspired network.

    The architecture mirrors the original fully‑connected stack but
    adds an optional RBF kernel layer that can be used as a feature
    extractor for support‑vector‑machine style classifiers.
    """

    def __init__(
        self,
        use_kernel: bool = False,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_kernel = use_kernel

        # Feature extraction: 8 → 16
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())

        # Convolution / pooling emulation
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Output head
        self.head = nn.Linear(4, 1)

        # Optional RBF kernel
        self.kernel: Optional[RBFKernel] = (
            RBFKernel(kernel_gamma) if use_kernel else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch, 8).

        Returns
        -------
        torch.Tensor
            Binary prediction in (0, 1).  If ``use_kernel`` is True
            the same tensor is returned; the kernel is only exposed
            via the ``kernel_matrix`` helper.
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        out = torch.sigmoid(self.head(x))
        return out

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two batches using the RBF kernel.

        Parameters
        ----------
        a, b : torch.Tensor
            Batches of shape (n, d) and (m, d).

        Returns
        -------
        torch.Tensor
            Gram matrix of shape (n, m).
        """
        if not self.use_kernel or self.kernel is None:
            raise RuntimeError("Kernel is not enabled in this instance.")
        # Expand for broadcasting
        a_exp = a.unsqueeze(1)  # (n, 1, d)
        b_exp = b.unsqueeze(0)  # (1, m, d)
        diff = a_exp - b_exp
        return torch.exp(-self.kernel.gamma * torch.sum(diff * diff, dim=-1))

def QCNN() -> QCNNHybrid:
    """Factory returning a default classical QCNN‑hybrid."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNN", "RBFKernel"]
