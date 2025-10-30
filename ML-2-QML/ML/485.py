"""Robust radial basis function kernel with adaptive gamma, GPU support, and batched Gram matrix."""

import torch
from torch import nn
import numpy as np
from typing import Sequence, Optional

class Kernel(nn.Module):
    """
    Enhanced RBF kernel.

    Parameters
    ----------
    gamma : float | None
        Base width parameter. If None, gamma is learned as a positive scalar.
    learnable : bool
        Whether to make gamma a learnable parameter.
    """

    def __init__(self, gamma: Optional[float] = 1.0, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(gamma if gamma is not None else 1.0))
            self.learnable = True
        else:
            self.gamma = torch.tensor(gamma if gamma is not None else 1.0)
            self.learnable = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches.
        Supports broadcasting and GPU tensors.
        """
        x = x.unsqueeze(1)  # (batch_x, 1, dim)
        y = y.unsqueeze(0)  # (1, batch_y, dim)
        diff = x - y
        sq_norm = (diff * diff).sum(dim=-1)
        gamma = self.gamma.clamp_min(1e-6)
        return torch.exp(-gamma * sq_norm)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], **kwargs) -> np.ndarray:
    """
    Compute Gram matrix between two collections of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Collections of feature vectors.
    **kwargs : dict
        Forwarded to :class:`Kernel`.
    """
    kernel = Kernel(**kwargs)
    mat = torch.stack([kernel(torch.tensor(x), torch.tensor(y)).squeeze()
                       for x in a for y in b])
    mat = mat.view(len(a), len(b))
    return mat.cpu().numpy()

__all__ = ["Kernel", "kernel_matrix"]
