"""
Classical implementation of a hybrid quanvolution network.
Provides a convolutional front‑end, optional RBF kernel surrogate,
and interchangeable classification/regression heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple


class RBFKernel(nn.Module):
    """Classical RBF kernel used as a surrogate for a quantum kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x, y: (N, D) or (M, D)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, D)
        sq_dist = (diff ** 2).sum(-1)  # (N, M)
        return torch.exp(-self.gamma * sq_dist)


class QuanvolutionHybrid(nn.Module):
    """
    Hybrid quanvolution network.

    Parameters
    ----------
    mode : str, optional
        One of ``'classification'`` or ``'regression'``.  Determines the
        default head used in :meth:`forward`.
    n_classes : int, optional
        Number of classes for classification head.
    gamma : float, optional
        RBF kernel width used in the surrogate kernel.
    """

    def __init__(
        self,
        mode: str = "classification",
        n_classes: int = 10,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.mode = mode

        # 2×2 patch extraction via a 2×2 convolution with stride 2
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

        # Surrogate quantum kernel (classical RBF)
        self.kernel = RBFKernel(gamma)

        # Heads
        self.linear_cls = nn.Linear(4 * 14 * 14, n_classes)
        self.linear_reg = nn.Linear(4 * 14 * 14, 1)

    # --------------------------------------------------------------------- #
    # Core forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, 28, 28).
        mode : str, optional
            Overrides the default mode.  One of ``'classification'`` or
            ``'regression'``.
        """
        if mode is None:
            mode = self.mode

        # Extract 2×2 patches via convolution
        features = self.conv(x)  # (B, 4, 14, 14)
        flat = features.view(features.size(0), -1)  # (B, 4*14*14)

        if mode == "classification":
            logits = self.linear_cls(flat)
            return F.log_softmax(logits, dim=-1)
        elif mode == "regression":
            return self.linear_reg(flat)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # --------------------------------------------------------------------- #
    # Kernel utilities
    # --------------------------------------------------------------------- #
    def kernel_matrix(self, a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute the Gram matrix between two collections of samples using the
        surrogate RBF kernel.

        Parameters
        ----------
        a : Iterable[torch.Tensor]
            First collection of samples.  Each element is a tensor of shape
            (1, 28, 28).
        b : Iterable[torch.Tensor]
            Second collection of samples.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (len(a), len(b)).
        """
        a_flat = torch.cat([self.conv(t.unsqueeze(0)).view(-1) for t in a], dim=0)
        b_flat = torch.cat([self.conv(t.unsqueeze(0)).view(-1) for t in b], dim=0)
        return self.kernel(a_flat, b_flat)

    # --------------------------------------------------------------------- #
    # Convenience helpers
    # --------------------------------------------------------------------- #
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Shortcut for classification."""
        return self.forward(x, mode="classification")

    def regress(self, x: torch.Tensor) -> torch.Tensor:
        """Shortcut for regression."""
        return self.forward(x, mode="regression")


__all__ = ["QuanvolutionHybrid"]
