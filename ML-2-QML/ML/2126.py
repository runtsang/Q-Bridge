"""Classical RBF kernel with learnable length‑scale and batched support."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from torch import nn

__all__ = ["QuantumKernelMethod", "kernel_matrix"]

class QuantumKernelMethod(nn.Module):
    """RBF kernel with a trainable gamma parameter.

    Parameters
    ----------
    gamma : float, optional
        Initial value for the length‑scale.  The actual parameter is
        stored as ``log_gamma`` to keep it strictly positive during
        optimisation.
    device : str, optional
        Target device for tensor operations (``'cpu'`` or ``'cuda'``).

    Notes
    -----
    The forward method accepts two 2‑D tensors ``x`` and ``y`` and
    returns a Gram matrix of shape ``(len(x), len(y))``.  It is fully
    differentiable, enabling direct optimisation of ``gamma`` in
    downstream pipelines.
    """
    def __init__(self, gamma: float = 1.0, device: str = "cpu") -> None:
        super().__init__()
        # store log‑gamma to guarantee positivity
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma, dtype=torch.float32)))
        self.device = device

    @property
    def gamma(self) -> torch.Tensor:
        """Current positive value of the length‑scale."""
        return torch.exp(self.log_gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel matrix between ``x`` and ``y``."""
        x = x.to(self.device)
        y = y.to(self.device)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        # pairwise squared Euclidean distance
        d2 = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)
        return torch.exp(-self.gamma * d2)

def kernel_matrix(a: Sequence[Union[np.ndarray, torch.Tensor]],
                  b: Sequence[Union[np.ndarray, torch.Tensor]],
                  gamma: float = 1.0) -> np.ndarray:
    """Convenience wrapper that returns a NumPy array of the Gram matrix.

    Parameters
    ----------
    a, b : sequences of tensors or arrays
        Data points for which the kernel matrix is evaluated.
    gamma : float, optional
        Initial length‑scale used by the underlying :class:`QuantumKernelMethod`.

    Returns
    -------
    np.ndarray
        The Gram matrix ``K_{ij} = k(a_i, b_j)``.
    """
    # Convert to torch tensors
    a_t = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in a]
    b_t = [torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in b]
    kernel = QuantumKernelMethod(gamma)
    return kernel.forward(torch.stack(a_t), torch.stack(b_t)).cpu().numpy()
