import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFKernel(nn.Module):
    """Gaussian radial basis function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (n, d), y: (m, d)
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))

class HybridNAT(nn.Module):
    """Classical hybrid model: CNN feature extractor + RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix between two batches using RBF kernel."""
        k = self.kernel(a, b)
        return k.detach().cpu().numpy()

__all__ = ["HybridNAT"]
