# Enhanced classical RBF kernel module with learnable bandwidth and GPU support.

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Sequence

__all__ = ["Kernel", "kernel_matrix"]

class Kernel(nn.Module):
    """Learnable RBF kernel with optional GPU acceleration."""
    def __init__(self, gamma: float = 1.0, learn_gamma: bool = False, device: str = 'cpu'):
        super().__init__()
        if learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32, device=device))
        else:
            self.gamma = gamma
        self.device = device

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        dist2 = torch.sum(diff * diff, dim=-1, keepdim=True)
        gamma = self.gamma if isinstance(self.gamma, torch.Tensor) else torch.tensor(self.gamma, dtype=torch.float32, device=self.device)
        return torch.exp(-gamma * dist2)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_exp = a.unsqueeze(1)
        b_exp = b.unsqueeze(0)
        diff = a_exp - b_exp
        dist2 = torch.sum(diff * diff, dim=-1)
        gamma = self.gamma if isinstance(self.gamma, torch.Tensor) else torch.tensor(self.gamma, dtype=torch.float32, device=self.device)
        return torch.exp(-gamma * dist2)

    def kernel_matrix_np(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        a_t = torch.tensor(a, dtype=torch.float32, device=self.device)
        b_t = torch.tensor(b, dtype=torch.float32, device=self.device)
        return self.kernel_matrix(a_t, b_t).cpu().numpy()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma=gamma)
    return kernel.kernel_matrix_np(a, b)
