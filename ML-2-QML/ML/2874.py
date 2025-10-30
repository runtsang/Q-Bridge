import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KernalAnsatz(nn.Module):
    """Radial Basis Function kernel with configurable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class KernelLayer(nn.Module):
    """Computes RBF kernel similarities between inputs and learnable support vectors."""
    def __init__(self, n_support: int, dim: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_support = n_support
        self.gamma = gamma
        # Support vectors are learnable parameters with the same dimensionality as the feature vector
        self.support_vectors = nn.Parameter(torch.randn(n_support, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        diff = x.unsqueeze(1) - self.support_vectors.unsqueeze(0)  # (batch, n_support, dim)
        dist_sq = torch.sum(diff * diff, dim=2)  # (batch, n_support)
        return torch.exp(-self.gamma * dist_sq)

class Hybrid(nn.Module):
    """Simple dense head that returns a sigmoid probability."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return torch.sigmoid(logits + self.shift)

class HybridKernelQCNet(nn.Module):
    """Convolutional backbone with kernel augmentation and hybrid quantum‑inspired head."""
    def __init__(self, n_support: int = 8, gamma: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.kernel = KernelLayer(n_support, dim=84, gamma=gamma)
        self.hybrid = Hybrid(84 + n_support, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        k = self.kernel(x)
        combined = torch.cat([x, k], dim=-1)
        probs = self.hybrid(combined)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["KernalAnsatz", "KernelLayer", "Hybrid", "HybridKernelQCNet"]
