"""Hybrid kernel estimator combining classical RBF, quanvolution, and a small neural net."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalKernel(nn.Module):
    """Exponentiated quadratic (RBF) kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class ClassicalQuanvolutionFilter(nn.Module):
    """Simple 2×2 patch extractor followed by a flattening."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)  # (batch, 4*14*14)


class HybridKernelEstimator(nn.Module):
    """
    Kernel ridge regression with an optional quanvolution front‑end and a small neural
    network that operates on the kernel matrix rows.
    """

    def __init__(
        self,
        use_quanvolution: bool = False,
        gamma: float = 1.0,
        hidden_sizes: list[int] | tuple[int,...] = (8, 4),
        regularization: float = 1e-5,
    ) -> None:
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.kernel = ClassicalKernel(gamma)
        self.hidden_sizes = list(hidden_sizes)
        self.regularization = regularization
        self.alpha: torch.Tensor | None = None
        self.n_train: int | None = None
        self.X_train: torch.Tensor | None = None
        self.fc: nn.Sequential | None = None
        if self.use_quanvolution:
            self.qfilter = ClassicalQuanvolutionFilter()

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Vectorised pairwise RBF kernel matrix."""
        a = a.unsqueeze(1)  # (N_a,1,D)
        b = b.unsqueeze(0)  # (1,N_b,D)
        diff = a - b
        sq_dist = torch.sum(diff * diff, dim=2)
        return torch.exp(-self.kernel.gamma * sq_dist)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Fit kernel ridge regression followed by a feed‑forward network."""
        if self.use_quanvolution:
            X = self.qfilter(X)
        K_train = self.kernel_matrix(X, X)
        K_reg = K_train + self.regularization * torch.eye(K_train.size(0), device=K_train.device)
        self.alpha = torch.linalg.solve(K_reg, y.squeeze())
        self.n_train = X.size(0)
        self.X_train = X
        # Build small neural net that takes a full kernel row as input
        layers: list[nn.Module] = []
        inp_dim = self.n_train
        for h in self.hidden_sizes:
            layers.append(nn.Linear(inp_dim, h))
            layers.append(nn.Tanh())
            inp_dim = h
        layers.append(nn.Linear(inp_dim, 1))
        self.fc = nn.Sequential(*layers)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return predictions for new samples."""
        if self.use_quanvolution:
            X = self.qfilter(X)
        K_test = self.kernel_matrix(X, self.X_train)
        out = self.fc(K_test)
        return out.squeeze()


__all__ = ["HybridKernelEstimator"]
