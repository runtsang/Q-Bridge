import numpy as np
import torch
from torch import nn
from typing import Sequence

class ClassicalQFCModel(nn.Module):
    """CNN followed by fully‑connected projection (classical analogue of QFCModel)."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = features.view(x.size(0), -1)
        out = self.fc(flattened)
        return self.norm(out)

class RBFKernel(nn.Module):
    """Radial‑basis function kernel with optional normalisation."""
    def __init__(self, gamma: float = 1.0, normalize: bool = True) -> None:
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff * diff, dim=2)
        kernel = torch.exp(-self.gamma * dist_sq)
        if self.normalize:
            diag = torch.diagonal(kernel)
            kernel = kernel / torch.sqrt(diag.unsqueeze(1) * diag.unsqueeze(0))
        return kernel

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, shots: int | None = None,
                  seed: int | None = None) -> np.ndarray:
    """Compute the Gram matrix for two datasets using a classical RBF kernel.
    Optionally add Gaussian shot noise to emulate a noisy quantum device."""
    if isinstance(a, Sequence) and not isinstance(a, torch.Tensor):
        a = torch.stack(a)
    if isinstance(b, Sequence) and not isinstance(b, torch.Tensor):
        b = torch.stack(b)
    kernel = RBFKernel(gamma=gamma)
    mat = kernel(a, b).detach().cpu().numpy()
    if shots is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=1.0/np.sqrt(shots), size=mat.shape)
        mat += noise
    return mat

__all__ = ["ClassicalQFCModel", "RBFKernel", "kernel_matrix"]
