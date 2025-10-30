import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class QuantumKernelMethod(nn.Module):
    """
    Classical RBF kernel implementation with GPU acceleration and batch support.
    Provides a kernel matrix factory compatible with FastBaseEstimator.
    """
    def __init__(self, gamma: float = 1.0, device: torch.device | str | None = None):
        super().__init__()
        self.gamma = gamma
        self.device = torch.device(device) if device else torch.device('cpu')

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (n, d), y: (m, d)
        diff = x[:, None, :] - y[None, :, :]
        dist_sq = torch.sum(diff * diff, dim=2)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel value for two 1‑D tensors.
        """
        x = x.to(self.device).view(1, -1)
        y = y.to(self.device).view(1, -1)
        return self._rbf(x, y).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two sets of vectors.
        Accepts either torch or numpy arrays; all are cast to torch tensors on the
        configured device.
        """
        a_t = torch.stack([torch.as_tensor(v, dtype=torch.float32, device=self.device).view(-1)
                           for v in a])
        b_t = torch.stack([torch.as_tensor(v, dtype=torch.float32, device=self.device).view(-1)
                           for v in b])
        mat = self._rbf(a_t, b_t)
        return mat.cpu().numpy()

__all__ = ["QuantumKernelMethod"]
