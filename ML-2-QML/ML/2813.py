import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around KernalAnsatz to provide a callable kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

class HybridKernelMethod(nn.Module):
    """Hybrid classical kernel method combining optional CNN feature extraction with RBF kernel."""
    def __init__(self, gamma: float = 1.0, use_cnn: bool = True) -> None:
        super().__init__()
        self.gamma = gamma
        self.use_cnn = use_cnn
        if self.use_cnn:
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.feature_dim = 16 * 7 * 7
        else:
            self.feature_dim = None

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            with torch.no_grad():
                feat = self.features(x)
                return feat.view(x.size(0), -1)
        else:
            return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_feat = self._extract(x)
        y_feat = self._extract(y)
        diff = x_feat[:, None, :] - y_feat[None, :, :]
        sq_norm = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sq_norm)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "HybridKernelMethod", "kernel_matrix"]
