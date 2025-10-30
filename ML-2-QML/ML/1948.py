"""
Fully connected layer with explicit trainability and gradient support.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import numpy as np
from torch import nn
from torch.autograd import grad


class FullyConnectedLayer(nn.Module):
    """
    A lightweight, fully‑connected neural layer that can be used as a stand‑in for a
    quantum layer.  The module is fully differentiable, exposes a ``run`` method
    that accepts a parameter vector, and provides helpers for gradient and
    Jacobian computation.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input parameter vector.
    bias : bool, default=True
        Whether to include a learnable bias term.

    Notes
    -----
    The forward pass applies a linear transformation followed by a tanh activation.
    The ``run`` method is a thin wrapper that converts a raw (numpy or iterable)
    array into a tensor before applying the linear layer.  This mirrors the
    interface of the original seed while adding full autograd support.
    """

    def __init__(self, n_features: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        thetas : torch.Tensor
            Input parameters of shape ``(n_features,)``.

        Returns
        -------
        torch.Tensor
            Scaled output of shape ``(1,)``.
        """
        return torch.tanh(self.linear(thetas))

    def run(self, thetas: Iterable[float] | np.ndarray) -> np.ndarray:
        """
        Public API that mimics the original FCL interface.

        Parameters
        ----------
        thetas : Iterable[float] or np.ndarray
            Input parameters.

        Returns
        -------
        np.ndarray
            Output as a NumPy array of shape ``(1,)``.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(list(thetas), dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            return self.forward(tensor).detach().numpy()

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the output with respect to the input parameters.

        Parameters
        ----------
        thetas : np.ndarray
            Input parameters of shape ``(n_features,)``.

        Returns
        -------
        np.ndarray
            Gradient vector of shape ``(n_features,)``.
        """
        self.zero_grad()
        tensor = torch.as_tensor(thetas, dtype=torch.float32, requires_grad=True)
        out = self.forward(tensor)
        out.backward()
        return tensor.grad.numpy()

    def jacobian(self, thetas: np.ndarray) -> np.ndarray:
        """
        Return the Jacobian (same as gradient for scalar output).

        Parameters
        ----------
        thetas : np.ndarray
            Input parameters.

        Returns
        -------
        np.ndarray
            Jacobian matrix of shape ``(1, n_features)``.
        """
        return self.gradient(thetas).reshape(1, -1)

    def train_step(
        self,
        thetas: np.ndarray,
        target: float,
        lr: float = 0.01,
        criterion: nn.Module = nn.MSELoss(),
    ) -> Tuple[float, np.ndarray]:
        """
        Perform a single gradient‑descent update on the parameters.

        Parameters
        ----------
        thetas : np.ndarray
            Initial parameters.
        target : float
            Desired output.
        lr : float, default=0.01
            Learning rate.
        criterion : nn.Module, default=MSELoss()
            Loss function.

        Returns
        -------
        loss_val : float
            Current loss value.
        updated_params : np.ndarray
            Updated parameter vector.
        """
        self.zero_grad()
        tensor = torch.as_tensor(thetas, dtype=torch.float32, requires_grad=True)
        out = self.forward(tensor)
        loss = criterion(out, torch.tensor(target, dtype=torch.float32))
        loss.backward()
        with torch.no_grad():
            tensor -= lr * tensor.grad
        return loss.item(), tensor.detach().numpy()


def FCL(n_features: int = 1, bias: bool = True) -> FullyConnectedLayer:
    """
    Factory that returns an instance of the fully connected layer.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input vector.
    bias : bool
        Flag to include bias in the linear layer.

    Returns
    -------
    FullyConnectedLayer
        Initialized layer ready for ``run`` or training.
    """
    return FullyConnectedLayer(n_features, bias)


__all__ = ["FCL"]
