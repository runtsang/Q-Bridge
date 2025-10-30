import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Optional

class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with optional automatic gamma selection.

    Parameters
    ----------
    gamma : float, optional
        Kernel width. If ``None`` and ``auto_gamma=True`` the median
        heuristic is used on the training data.
    auto_gamma : bool, default=True
        Whether to compute ``gamma`` automatically from the data.
    device : str or torch.device, default='cpu'
        Device on which to perform computations.
    """

    def __init__(self,
                 gamma: Optional[float] = None,
                 auto_gamma: bool = True,
                 device: torch.device | str = 'cpu') -> None:
        super().__init__()
        self.gamma = gamma
        self.auto_gamma = auto_gamma
        self.device = torch.as_tensor([], dtype=torch.float32, device=device).device

    def _compute_gamma(self, x: torch.Tensor) -> float:
        """Median heuristic: gamma = 1/(2 * median(||x_i-x_j||^2))."""
        with torch.no_grad():
            dists = torch.cdist(x, x, p=2)
            mask = ~torch.eye(dists.shape[0], dtype=bool, device=dists.device)
            median = torch.median(dists[mask]).item()
            return 1.0 / (2 * median**2 + 1e-8)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel value ``k(x, y)``."""
        x, y = x.to(self.device), y.to(self.device)
        if self.auto_gamma and self.gamma is None:
            self.gamma = self._compute_gamma(x)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      gamma: Optional[float] = None,
                      auto_gamma: bool = True) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        a = torch.stack(a)
        b = torch.stack(b)
        if gamma is None and auto_gamma:
            combined = torch.cat([a, b], dim=0)
            dists = torch.cdist(combined, combined, p=2)
            mask = ~torch.eye(dists.shape[0], dtype=bool, device=dists.device)
            median = torch.median(dists[mask]).item()
            gamma = 1.0 / (2 * median**2 + 1e-8)
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        K = torch.exp(-gamma * torch.sum(diff * diff, dim=-1))
        return K.cpu().numpy()

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
