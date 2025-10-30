"""Hybrid fully‑connected layer with optional convolution and kernel.

This module intentionally mirrors the interface of the original FCL example
while adding classical convolution and a radial‑basis function kernel.
The class can be instantiated in ``classical`` mode only, and exposes a
``forward`` method that accepts a list of parameters and optional 2‑D data.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Optional helpers from the seed repo
from Conv import Conv
from QuantumKernelMethod import Kernel as RBFKernel


class HybridFCL(nn.Module):
    """
    Classical hybrid layer.

    Parameters
    ----------
    n_features : int
        Number of input feature dimension for the linear head.
    kernel_gamma : float
        Gamma for the RBF kernel (ignored in pure linear mode).
    conv_kernel_size : int
        Size of the 2‑D convolution filter; if ``None`` the filter is disabled.
    """

    def __init__(
        self,
        n_features: int = 1,
        kernel_gamma: float = 1.0,
        conv_kernel_size: int | None = 2,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.kernel = RBFKernel(gamma=kernel_gamma)
        self.conv = Conv() if conv_kernel_size else None

    def forward(
        self,
        thetas: list[float],
        data: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the output of the hybrid layer.

        The linear head uses the provided parameters, optionally averages
        over the batch, and optionally adds a convolutional score.

        Parameters
        ----------
        thetas : list[float]
            Parameters for the linear head.
        data : np.ndarray, optional
            2‑D array of shape (kernel_size, kernel_size) to feed the Conv filter.

        Returns
        -------
        np.ndarray
            The computed expectation value(s) as a NumPy array.
        """
        # Linear branch
        vals = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        lin_out = torch.tanh(self.linear(vals)).mean(dim=0)

        # Optional convolution branch
        if self.conv is not None and data is not None:
            conv_out = self.conv.run(data)
            return (lin_out + conv_out).detach().numpy()

        return lin_out.detach().numpy()

    def kernel_matrix(
        self, a: list[torch.Tensor], b: list[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute an RBF Gram matrix between two batches of tensors.
        """
        return np.array(
            [[self.kernel(x, y).item() for y in b] for x in a]
        )


__all__ = ["HybridFCL"]
