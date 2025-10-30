"""Classical implementation of a hybrid quantum kernel method.

The class `QuantumKernelMethod` can operate in classical mode only. It
provides an RBF kernel whose inputs are first mapped through a fully
connected layer (mirroring the FCL example).  The feature map is a
single `nn.Linear` layer followed by a Tanh non‑linearity.  The kernel
matrix can be computed efficiently for two collections of samples.
"""

import numpy as np
import torch
from torch import nn
from typing import Sequence

class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with a learnable feature map.

    Parameters
    ----------
    gamma : float, default=1.0
        Width of the RBF kernel.
    n_features : int, default=1
        Size of the hidden feature space produced by the linear layer.
    device : str or torch.device, default='cpu'
        Device on which tensors are allocated.
    """

    def __init__(self,
                 gamma: float = 1.0,
                 n_features: int = 1,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__()
        self.gamma = gamma
        self.device = torch.device(device)
        # Feature map: linear layer + tanh
        self.feature_map = nn.Sequential(
            nn.Linear(1, n_features, bias=True),
            nn.Tanh()
        ).to(self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value for two 1‑D tensors.

        Parameters
        ----------
        x, y : torch.Tensor
            Input vectors of shape (n,) or (batch, n).

        Returns
        -------
        torch.Tensor
            Scalar kernel value or batch of values.
        """
        # Ensure correct device and type
        x = x.to(self.device).float()
        y = y.to(self.device).float()

        # Map to feature space
        x_f = self.feature_map(x.view(-1, 1))
        y_f = self.feature_map(y.view(-1, 1))

        diff = x_f - y_f
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True)).squeeze()

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      gamma: float = 1.0,
                      n_features: int = 1,
                      device: str | torch.device = 'cpu') -> np.ndarray:
        """Compute Gram matrix between two collections of vectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.
        gamma, n_features, device : same as ``__init__``

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        km = QuantumKernelMethod(gamma=gamma,
                                 n_features=n_features,
                                 device=device)
        return np.array([[km(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod"]
