"""Classical radial basis function kernel with learnable feature extraction."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
#  Feature extractor
# --------------------------------------------------------------------------- #
class _FeatureExtractor(nn.Module):
    """Learnable feature transformation before the RBF kernel."""
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  RBF kernel that operates on extracted features
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """RBF kernel with a trainable feature extractor and length‑scale."""
    def __init__(self, in_dim: int, gamma: float = 1.0, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_extractor = _FeatureExtractor(in_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Ensure proper shape: (batch, features)
        x_f = self.feature_extractor(x)
        y_f = self.feature_extractor(y)
        diff = x_f - y_f
        sq_norm = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * sq_norm)


# --------------------------------------------------------------------------- #
#  Kernel wrapper
# --------------------------------------------------------------------------- #
class Kernel(nn.Module):
    """Convenience wrapper that exposes a device and trainable flag."""
    def __init__(self,
                 in_dim: int,
                 gamma: float = 1.0,
                 hidden_dim: int = 64,
                 device: torch.device | str | None = None,
                 trainable: bool = True) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")
        self.ansatz = KernalAnsatz(in_dim, gamma, hidden_dim).to(self.device)
        self.trainable = trainable
        if not trainable:
            for p in self.ansatz.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x.to(self.device), y.to(self.device)).squeeze()


# --------------------------------------------------------------------------- #
#  Gram matrix utility
# --------------------------------------------------------------------------- #
def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  in_dim: int,
                  gamma: float = 1.0,
                  hidden_dim: int = 64,
                  device: torch.device | str | None = None,
                  trainable: bool = True) -> np.ndarray:
    kernel = Kernel(in_dim, gamma, hidden_dim, device, trainable)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["_FeatureExtractor", "KernalAnsatz", "Kernel", "kernel_matrix"]
