import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 quanvolution filter implemented as a Conv2d layer."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QCNNFeatureExtractor(nn.Module):
    """QCNN-inspired fully‑connected stack."""
    def __init__(self, in_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class HybridHead(nn.Module):
    """Simple dense head that emulates the quantum expectation output."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = x.view(x.size(0), -1)
        return torch.sigmoid(logits + self.shift)

class Kernel(nn.Module):
    """Radial‑basis‑function kernel for similarity estimation."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

class QuanvolutionHybrid(nn.Module):
    """
    Hybrid network that stitches together a classical quanvolution filter, a QCNN‑style
    feature extractor, and a hybrid head.  The architecture mirrors the QCNet backbone
    and the QCNN fully‑connected stack, while the filter can be swapped with a quantum
    variant in the quantum module.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2, n_qubits: int = 4):
        super().__init__()
        self.filter = QuanvolutionFilter(in_channels, out_channels, kernel_size, stride)

        # QCNet‑style convolutional backbone
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Feature extractor expects concatenated backbone + filter features
        combined_features_dim = 1 + out_channels * 14 * 14  # 1 from fc3 + filter dim
        self.feature_extractor = QCNNFeatureExtractor(combined_features_dim)
        self.head = HybridHead(self.fc3.out_features)
        self.kernel = Kernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical quanvolution filter
        f = self.filter(x)

        # Convolutional backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Combine backbone and filter features
        combined = torch.cat([x, f], dim=1)
        features = self.feature_extractor(combined)

        # Hybrid head
        logits = self.head(features)
        return torch.cat((logits, 1 - logits), dim=-1)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return self.kernel.kernel_matrix(a, b)

__all__ = ["QuanvolutionHybrid", "QuanvolutionFilter", "QCNNFeatureExtractor", "HybridHead", "Kernel"]
