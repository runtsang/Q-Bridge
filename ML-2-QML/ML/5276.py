"""
Hybrid regression model that fuses quanvolutional features, RBF kernel similarity, and a linear head.
The architecture is inspired by EstimatorQNN, Quanvolution, QuantumKernelMethod, and QuantumRegression.
"""

import torch
from torch import nn
import torch.nn.functional as F

# Classical quanvolution filter (from the original Quanvolution example)
class QuanvolutionFilter(nn.Module):
    """Apply a 2×2 classical convolution to extract patch features."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

# RBF kernel implemented directly for efficiency
def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """
    Compute the RBF kernel matrix between two batches.
    x: (batch, features)
    y: (num_support, features)
    Returns: (batch, num_support)
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)          # (batch, num_support, features)
    dist_sq = torch.sum(diff ** 2, dim=-1)           # (batch, num_support)
    return torch.exp(-gamma * dist_sq)

class EstimatorQNNGen160(nn.Module):
    """
    Hybrid estimator that combines:
      - Classical quanvolutional feature extraction
      - RBF kernel similarity to 160 support vectors
      - Linear regression head
    """
    def __init__(self, num_support: int = 160, gamma: float = 10.0) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()

        # Support vectors sampled from a uniform distribution over the image space
        support = torch.randn(num_support, 28 * 28, dtype=torch.float32)
        self.register_buffer("support", support)
        self.register_buffer("support_labels", torch.randn(num_support, dtype=torch.float32))

        self.gamma = gamma
        # Linear head that consumes concatenated classical + kernel features
        self.head = nn.Linear(28 * 28 + num_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape (batch, 1, 28, 28)
        Returns:
            Tensor of shape (batch,) with regression predictions.
        """
        # Classical quanvolutional features
        cls_feat = self.qfilter(x)                      # (batch, 28*28)

        # Kernel similarity to support vectors
        kernel_vals = rbf_kernel(cls_feat, self.support, gamma=self.gamma)  # (batch, num_support)

        # Concatenate features and feed through linear head
        combined = torch.cat([cls_feat, kernel_vals], dim=1)
        return self.head(combined).squeeze(-1)

__all__ = ["EstimatorQNNGen160"]
