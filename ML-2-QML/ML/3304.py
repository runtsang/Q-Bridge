import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    """2‑D convolution that extracts non‑overlapping 2×2 patches."""
    def __init__(self, in_channels=1, out_channels=4, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)                     # shape: (B, 4, 14, 14)
        return features.view(x.size(0), -1)         # flatten to (B, 4*14*14)

class HybridEstimatorQNN(nn.Module):
    """
    Classical regression network that uses a quanvolution filter as a feature extractor
    followed by a linear head.  This mirrors the original EstimatorQNN but replaces
    the hand‑crafted fully‑connected layers with a learnable convolutional front‑end.
    """
    def __init__(self, in_channels=1, num_features=4, hidden_dim=4, output_dim=1):
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter(in_channels, num_features)
        self.linear = nn.Linear(num_features * 14 * 14, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        out = self.linear(features)
        return out

__all__ = ["HybridEstimatorQNN"]
