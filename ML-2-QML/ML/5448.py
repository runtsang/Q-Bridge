import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 convolutional filter as in the original quanvolution example."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class RBFKernel(nn.Module):
    """Classical radial basis function kernel with learnable gamma."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff * diff, dim=2)
        return torch.exp(-self.gamma * dist_sq)

class HybridQuanvolution(nn.Module):
    """Hybrid classifier that uses a classical quanvolution filter followed by an RBF kernel head and a linear classifier."""
    def __init__(self, num_classes: int = 10, num_prototypes: int = 20, feature_dim: int = 4 * 14 * 14):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.kernel = RBFKernel()
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        k_values = self.kernel(features, self.prototypes)
        logits = self.classifier(k_values)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "RBFKernel", "HybridQuanvolution"]
