"""Hybrid kernel module combining classical RBF and quantum‑inspired random Fourier features."""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
from torch import nn

__all__ = ["QuantumKernelMethod", "kernel_matrix", "AdaptiveKernelSelector"]


class AdaptiveKernelSelector:
    """
    Selects a suitable gamma for the RBF kernel using the median heuristic.
    """
    def __init__(self, gamma: Optional[float] = None):
        self.gamma = gamma

    def fit(self, X: torch.Tensor) -> float:
        if self.gamma is not None:
            return self.gamma
        distances = torch.cdist(X, X, p=2).pow(2)
        median = torch.median(distances.flatten())
        self.gamma = 1.0 / (2 * median.item())
        return self.gamma

    def __call__(self, X: torch.Tensor) -> float:
        return self.fit(X)


class QuantumKernelMethod(nn.Module):
    """
    Hybrid kernel that can operate in three modes:
        * 'classical'  – pure RBF kernel
        * 'quantum'    – quantum‑inspired random Fourier feature kernel
        * 'hybrid'     – weighted sum of the above two kernels
    """
    def __init__(
        self,
        mode: str = "classical",
        gamma: Optional[float] = None,
        alpha: float = 0.5,
        n_wires: int = 4,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if mode not in {"classical", "quantum", "hybrid"}:
            raise ValueError(f"Unsupported mode {mode!r}")
        self.mode = mode
        self.alpha = alpha
        self.n_wires = n_wires
        self.random_state = random_state

        # Gamma selector for classical RBF
        self.gamma_selector = AdaptiveKernelSelector(gamma)
        self.gamma = gamma

        # Random Fourier feature parameters for the quantum‑inspired kernel
        if mode in {"quantum", "hybrid"}:
            self.num_features = 2 * n_wires
            self.W = None
            self.b = None

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        sq_dist = torch.sum(diff * diff, dim=-1, keepdim=True)
        gamma = self.gamma if self.gamma is not None else self.gamma_selector.fit(x.unsqueeze(0))
        return torch.exp(-gamma * sq_dist)

    def _random_feature_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.W is None or self.b is None:
            rng = np.random.default_rng(self.random_state)
            d = x.shape[-1]
            scale = np.sqrt(2 * self.gamma if self.gamma is not None else 1.0)
            self.W = torch.tensor(rng.normal(scale=scale, size=(self.num_features, d)), dtype=torch.float32)
            self.b = torch.tensor(rng.uniform(0.0, 2 * np.pi, size=(self.num_features,)), dtype=torch.float32)
        z_x = torch.cos(x @ self.W.T + self.b) / torch.sqrt(torch.tensor(self.num_features, dtype=torch.float32))
        z_y = torch.cos(y @ self.W.T + self.b) / torch.sqrt(torch.tensor(self.num_features, dtype=torch.float32))
        return torch.sum(z_x * z_y, dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "classical":
            return self._rbf_kernel(x, y).squeeze()
        if self.mode == "quantum":
            return self._random_feature_kernel(x, y).squeeze()
        # Hybrid
        rbf = self._rbf_kernel(x, y).squeeze()
        q = self._random_feature_kernel(x, y).squeeze()
        return self.alpha * rbf + (1 - self.alpha) * q

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        a = torch.stack(a)
        b = torch.stack(b)
        mat = torch.zeros((len(a), len(b)), dtype=torch.float32)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                mat[i, j] = self.forward(x, y)
        return mat.numpy()
