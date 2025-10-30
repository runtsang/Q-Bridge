"""Hybrid classical QCNN with RBF kernel.

This module defines :class:`QCNNHybrid` that combines a convolution‑style
feature extractor with a learnable RBF kernel layer.  The kernel
acts as a quantum‑inspired similarity operator, mirroring the
quantum kernel used in the companion QML module.
"""

import torch
from torch import nn
from typing import List


class RBFKernel(nn.Module):
    """Learnable Radial Basis Function kernel.

    The kernel is parameterised by a prototype vector and a
    positive scaling factor ``γ``.  It is equivalent to the classical
    RBF kernel from the original `QuantumKernelMethod` seed.
    """

    def __init__(self, dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float))
        self.prototype = nn.Parameter(torch.randn(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, dim).

        Returns
        -------
        torch.Tensor
            Kernel values of shape (batch, 1).
        """
        diff = x - self.prototype
        return torch.exp(-self.gamma * (diff * diff).sum(dim=-1, keepdim=True))


class QCNNHybrid(nn.Module):
    """Classical QCNN architecture augmented with an RBF kernel layer.

    The network mirrors the layer structure of the original QCNN
    seed but replaces the final fully‑connected head with a kernel
    layer that captures non‑linear similarity.  A tiny linear head
    maps the scalar kernel value to a binary prediction.
    """

    def __init__(self, in_features: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, 16), nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())

        # Kernel layer with a learnable prototype
        self.kernel = RBFKernel(dim=2)

        # Final classification head
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, in_features).

        Returns
        -------
        torch.Tensor
            Sigmoid‑activated binary prediction of shape (batch, 1).
        """
        h = self.feature_map(x)
        h = self.conv1(h)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.pool2(h)
        h = self.conv3(h)
        h = self.pool3(h)
        k = self.kernel(h)
        out = torch.sigmoid(self.head(k))
        return out


def create_QCNNHybrid() -> QCNNHybrid:
    """Factory for a default :class:`QCNNHybrid` instance."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "create_QCNNHybrid", "RBFKernel"]
