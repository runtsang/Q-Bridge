"""Hybrid classical estimator combining a feed‑forward network with an RBF kernel layer.

The implementation retains the original EstimatorQNN API while exposing a richer
model that learns a set of support vectors and applies a Gaussian kernel to
compute similarities.  This design mirrors the quantum kernel construction
in the QML version, allowing direct comparison of classical versus quantum
feature maps."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable


class KernalAnsatz(nn.Module):
    """Gaussian RBF kernel function."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wrapper that normalises input shapes for the kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


class EstimatorQNN(nn.Module):
    """
    Hybrid estimator that augments a shallow feed‑forward network with a
    learnable kernel layer.  The network first projects the input into a
    hidden space, then computes kernel similarities between the projected
    input and a set of support vectors.  A linear read‑out produces the
    final regression output.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 kernel_dim: int = 8,
                 gamma: float = 1.0) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.kernel = Kernel(gamma)
        # Support vectors are initialized as learnable parameters
        self.support = nn.Parameter(torch.randn(kernel_dim, hidden_dim))
        self.readout = nn.Linear(kernel_dim, 1, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid kernel‑augmented regression output."""
        feats = self.feature_extractor(inputs)
        # Compute kernel between input features and support vectors
        k = self.kernel(feats, self.support)
        return self.readout(k)

    def fit_support(self,
                    X: Iterable[torch.Tensor],
                    y: Iterable[torch.Tensor],
                    epochs: int = 200,
                    lr: float = 1e-3) -> None:
        """
        Simple training loop that optimises both the feature extractor and the
        support vectors.  The loss is MSE between predictions and targets.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        X = torch.stack(X)
        y = torch.stack(y).unsqueeze(-1)
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()


def EstimatorQNN() -> EstimatorQNN:
    """Return a fresh instance of the hybrid estimator with default hyper‑parameters."""
    return EstimatorQNN()


__all__ = ["EstimatorQNN"]
