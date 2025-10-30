import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Radial basis function kernel building block."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module with a matrix helper."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack([torch.stack([self.forward(x, y) for y in b]) for x in a])

class QuantumNATHybrid(nn.Module):
    """Classical CNN + kernel‑ridge classifier."""
    def __init__(self, n_classes: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )
        self.kernel = Kernel(gamma)
        self.register_buffer('support', torch.empty(0))
        self.register_buffer('alpha', torch.empty(0))

    def fit(self, X: torch.Tensor, y: torch.Tensor, reg: float = 1e-3) -> None:
        """Fit kernel ridge regression on extracted features."""
        with torch.no_grad():
            feats = self.features(X).view(X.shape[0], -1)
            K = self.kernel.kernel_matrix(feats, feats)
            alpha = torch.linalg.solve(K + reg * torch.eye(K.size(0), device=K.device), y.float())
            self.register_buffer('support', feats)
            self.register_buffer('alpha', alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x).view(x.shape[0], -1)
        if self.support.numel() == 0:
            return self.fc(feats)
        K_test = self.kernel.kernel_matrix(feats, self.support)
        return K_test @ self.alpha

__all__ = ["QuantumNATHybrid"]
