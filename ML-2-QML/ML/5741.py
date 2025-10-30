import numpy as np
import torch
from torch import nn
from typing import Sequence

class QuantumKernelMethod__gen126(nn.Module):
    """
    Hybrid RBF kernel with trainable width (gamma) and optional data normalization.
    Provides forward(x, y) -> kernel value and kernel_matrix(a, b) -> Gram matrix.
    """
    def __init__(self, gamma: float = 1.0, normalize: bool = False, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, device=device, dtype=torch.float64))
        self.normalize = normalize
        self.device = device

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-8
            return (x - mean) / std
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return scalar kernel value for two input vectors.
        """
        x = self._preprocess(x)
        y = self._preprocess(y)
        diff = x - y
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_norm).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute Gram matrix between two sequences of tensors.
        """
        a = torch.stack(a).to(self.device)
        b = torch.stack(b).to(self.device)
        if self.normalize:
            a = self._preprocess(a)
            b = self._preprocess(b)
        # Compute pairwise squared Euclidean distances
        a_sq = torch.sum(a * a, dim=1, keepdim=True)  # (len(a), 1)
        b_sq = torch.sum(b * b, dim=1, keepdim=True).t()  # (1, len(b))
        sq_dist = a_sq + b_sq - 2 * a @ b.t()
        kernel = torch.exp(-self.gamma * sq_dist)
        return kernel.cpu().numpy()

__all__ = ["QuantumKernelMethod__gen126"]
