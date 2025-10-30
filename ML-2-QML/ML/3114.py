import torch
import torch.nn as nn
import numpy as np

class FeatureExtractor(nn.Module):
    """CNN feature extractor producing 4‑dimensional embeddings."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class RBFKernel(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (n, m, d)
        dist_sq = torch.sum(diff * diff, dim=-1)        # (n, m)
        return torch.exp(-self.gamma * dist_sq)

def kernel_matrix(a: torch.Tensor, b: torch.Tensor, gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sets of embeddings."""
    kernel = RBFKernel(gamma)
    return kernel(a, b).cpu().numpy()

class QuantumKernelMethod(nn.Module):
    """Hybrid kernel using CNN features and classical RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel matrix between two batches of images."""
        emb_x = self.feature_extractor(x)
        emb_y = self.feature_extractor(y)
        return self.kernel(emb_x, emb_y)

    def kernel_matrix(self, x: torch.Tensor, y: torch.Tensor, gamma: float | None = None) -> np.ndarray:
        """Convenience wrapper returning a NumPy array."""
        if gamma is None:
            gamma = self.kernel.gamma
        return kernel_matrix(self.feature_extractor(x), self.feature_extractor(y), gamma)

__all__ = ["FeatureExtractor", "RBFKernel", "kernel_matrix", "QuantumKernelMethod"]
