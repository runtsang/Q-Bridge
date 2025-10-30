"""Classical radial basis function kernel utilities with learnable width and batch support."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from torch import nn


class RBFKernel(nn.Module):
    """
    Differentiable RBF kernel with optional learnable width (gamma).

    Parameters
    ----------
    gamma : float, optional
        Initial kernel width. If ``learnable`` is True, ``gamma`` becomes a
        :class:`torch.nn.Parameter` and can be optimized jointly with other
        model parameters.
    learnable : bool, default False
        Whether ``gamma`` should be a trainable parameter.
    """

    def __init__(self, gamma: float = 1.0, learnable: bool = False) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma)) if learnable else gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix.

        Parameters
        ----------
        x : torch.Tensor
            Shape (N, D) – first set of samples.
        y : torch.Tensor
            Shape (M, D) – second set of samples.

        Returns
        -------
        torch.Tensor
            Shape (N, M) – kernel matrix K_{ij} = exp(-gamma * ||x_i - y_j||^2).
        """
        # Use torch.cdist for efficient pairwise Euclidean distance.
        dists = torch.cdist(x, y, p=2)
        gamma = self.gamma if isinstance(self.gamma, torch.Tensor) else torch.tensor(self.gamma, device=dists.device)
        return torch.exp(-gamma * dists.pow(2))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma={float(self.gamma) if isinstance(self.gamma, torch.Tensor) else self.gamma}, learnable={isinstance(self.gamma, torch.Tensor)})"


def kernel_matrix(
    a: Union[Sequence[torch.Tensor], torch.Tensor],
    b: Union[Sequence[torch.Tensor], torch.Tensor],
    gamma: float = 1.0,
    learnable: bool = False,
) -> np.ndarray:
    """
    Convenience wrapper that accepts a list of tensors or a single tensor and
    returns a NumPy array containing the Gram matrix.

    Parameters
    ----------
    a, b : sequence or tensor
        Input data. If a single tensor is supplied, it is treated as a batch
        of samples.
    gamma : float, default 1.0
        Initial kernel width.
    learnable : bool, default False
        Whether the kernel width should be a learnable parameter. This flag
        is ignored when the function returns a NumPy array, but it allows
        callers that want to keep the kernel as a torch module.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape (len(a), len(b)).
    """
    # Convert inputs to tensors if they are lists of tensors.
    if isinstance(a, torch.Tensor):
        a_tensor = a
    else:
        a_tensor = torch.stack(a, dim=0)
    if isinstance(b, torch.Tensor):
        b_tensor = b
    else:
        b_tensor = torch.stack(b, dim=0)

    kernel = RBFKernel(gamma=gamma, learnable=learnable)
    return kernel(a_tensor, b_tensor).detach().cpu().numpy()


__all__ = ["RBFKernel", "kernel_matrix"]
