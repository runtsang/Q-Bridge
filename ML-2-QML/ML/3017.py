import torch
import torch.nn as nn
import numpy as np
from typing import Sequence

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz with gamma parameter."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` providing a callable kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridKernelClassifier(nn.Module):
    """Hybrid kernel-based classifier that blends a CNN feature extractor with a classical RBF kernel."""
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 120,
        num_classes: int = 2,
        kernel_gamma: float = 1.0,
        support_vectors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        # Feature extractor mirroring the CNN from the binary‑classification seed
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(120, feature_dim),
        )
        # Classical RBF kernel
        self.kernel = Kernel(kernel_gamma)
        # Support vectors used to build the kernel feature map
        if support_vectors is None:
            support_vectors = torch.randn(10, feature_dim)
        self.support_vectors = nn.Parameter(support_vectors, requires_grad=True)
        # Linear classifier on the kernel feature map
        self.classifier = nn.Linear(self.support_vectors.size(0), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        feat = self.features(x)
        # Compute pairwise RBF kernel between feat and support vectors
        diff = feat.unsqueeze(1) - self.support_vectors.unsqueeze(0)  # (B, S, D)
        dist_sq = torch.sum(diff * diff, dim=2)
        kernel_vals = torch.exp(-self.kernel.gamma * dist_sq)
        # Classify
        logits = self.classifier(kernel_vals)
        return logits

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "HybridKernelClassifier"]
