import torch
import torch.nn as nn
import numpy as np

class QuantumKernelMethod(nn.Module):
    """Classical RBF kernel with vectorized computation and GPU support."""
    def __init__(self, gamma: float = 1.0, device: torch.device | str = 'cpu'):
        super().__init__()
        self.gamma = gamma
        self.device = torch.device(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the RBF kernel matrix for two batches of samples."""
        x = x.to(self.device)
        y = y.to(self.device)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        sq_norm = (diff * diff).sum(-1)
        return torch.exp(-self.gamma * sq_norm)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that forwards to ``forward``."""
        return self.forward(a, b)

    @staticmethod
    def from_numpy(a: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Compute the RBF kernel matrix directly with NumPy."""
        diff = a[:, None, :] - b[None, :, :]
        sq_norm = np.sum(diff * diff, axis=-1)
        return np.exp(-gamma * sq_norm)

__all__ = ["QuantumKernelMethod"]
