import numpy as np
import torch
from torch import nn
from typing import Sequence, Dict, Any, Optional

class QuantumKernelMethod(nn.Module):
    """
    Hybrid kernel framework that supports classical RBF, quantum-inspired, and custom kernels.
    Features GPU acceleration, kernel matrix caching, and a simple grid‑search routine.
    """
    def __init__(self,
                 kernel_type: str = 'rbf',
                 gamma: float = 1.0,
                 device: str = 'cpu',
                 cache: bool = True):
        super().__init__()
        if kernel_type not in ('rbf', 'hybrid', 'custom'):
            raise ValueError(f"Unsupported kernel_type {kernel_type!r}")
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.device = torch.device(device)
        self.cache = cache
        self._cache: Dict[tuple, torch.Tensor] = {}
        # For hybrid kernel: random Fourier features
        if self.kernel_type == 'hybrid':
            self.feature_dim = 64  # number of random features
            # Random projection matrix will be generated on demand
            self._W = None
            self._b = None

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the classical RBF kernel.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n_x, n_y, d)
        dist_sq = torch.sum(diff * diff, dim=-1)  # (n_x, n_y)
        return torch.exp(-self.gamma * dist_sq)

    def _hybrid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Hybrid kernel: product of RBF and random Fourier feature kernel.
        """
        rbf = self._rbf(x, y)
        # Random Fourier features
        if self._W is None or self._b is None:
            self._W = torch.randn(self.feature_dim, x.shape[1], device=self.device) * np.sqrt(2 * self.gamma)
            self._b = torch.rand(self.feature_dim, device=self.device) * 2 * np.pi
        z_x = torch.sqrt(2 / self.feature_dim) * torch.cos(x @ self._W.t() + self._b)
        z_y = torch.sqrt(2 / self.feature_dim) * torch.cos(y @ self._W.t() + self._b)
        feature_kernel = z_x @ z_y.t()
        return rbf * feature_kernel

    def _custom(self, x: torch.Tensor, y: torch.Tensor, func: Any) -> torch.Tensor:
        """
        Custom kernel: user-provided callable.
        """
        return func(x, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor, custom_func: Optional[Any] = None) -> torch.Tensor:
        """
        Compute the kernel matrix between x and y.
        """
        key = (tuple(x.shape), tuple(y.shape), self.kernel_type, self.gamma)
        if self.cache and key in self._cache:
            return self._cache[key]
        if self.kernel_type == 'rbf':
            K = self._rbf(x, y)
        elif self.kernel_type == 'hybrid':
            K = self._hybrid(x, y)
        else:  # custom
            if custom_func is None:
                raise ValueError("custom_func must be provided for custom kernel type")
            K = self._custom(x, y, custom_func)
        if self.cache:
            self._cache[key] = K
        return K

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                      kernel_type: str = 'rbf',
                      gamma: float = 1.0,
                      device: str = 'cpu',
                      cache: bool = True) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.
        """
        # Flatten each tensor to 2D
        a_flat = [t.reshape(t.shape[0], -1).cpu().numpy() for t in a]
        b_flat = [t.reshape(t.shape[0], -1).cpu().numpy() for t in b]
        a_torch = torch.tensor(np.vstack(a_flat))
        b_torch = torch.tensor(np.vstack(b_flat))
        model = QuantumKernelMethod(kernel_type=kernel_type, gamma=gamma, device=device, cache=cache)
        K = model.forward(a_torch, b_torch)
        return K.cpu().numpy()

    def grid_search(self, X: torch.Tensor, gamma_values: Sequence[float]) -> float:
        """
        Simple grid search over gamma to maximize the average kernel value.
        Returns the best gamma.
        """
        best_gamma = None
        best_score = -np.inf
        for gamma in gamma_values:
            self.gamma = gamma
            K = self.forward(X, X)
            score = torch.mean(K).item()
            if score > best_score:
                best_score = score
                best_gamma = gamma
        self.gamma = best_gamma
        return best_gamma

__all__ = ['QuantumKernelMethod']
