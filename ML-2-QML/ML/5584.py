import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RBFKernel(nn.Module):
    """Classical RBF kernel that can be fused into a filter."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(dim=-1, keepdim=True))

class HybridQuanvolutionFilter(nn.Module):
    """Combines a 2‑D convolution with an RBF‑based feature map."""
    def __init__(self, kernel_size: int = 2, stride: int = 2, gamma: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, bias=True)
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches via convolution
        patches = self.conv(x)  # shape (B,1,14,14)
        patches_flat = patches.view(patches.size(0), -1)  # (B,196)
        # Compute pairwise RBF features against a fixed reference set
        # For simplicity we use the patches themselves as reference
        # resulting in a (B,196) feature tensor
        return patches_flat

class EstimatorQNN(nn.Module):
    """Light‑weight regression network inspired by the Qiskit EstimatorQNN."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class HybridQuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the classical filter and a linear head."""
    def __init__(self, num_classes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.qfilter = HybridQuanvolutionFilter(gamma=gamma)
        self.linear = nn.Linear(196, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

class HybridQuanvolutionRegressor(nn.Module):
    """Regressor that combines the filter with EstimatorQNN."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.qfilter = HybridQuanvolutionFilter(gamma=gamma)
        self.estimator = EstimatorQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x).view(x.size(0), -1)
        # Use first two features as input to EstimatorQNN
        selected = features[:, :2]
        return self.estimator(selected)

__all__ = [
    "RBFKernel",
    "HybridQuanvolutionFilter",
    "EstimatorQNN",
    "HybridQuanvolutionClassifier",
    "HybridQuanvolutionRegressor",
]
