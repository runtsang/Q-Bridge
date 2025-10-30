"""Hybrid classical network combining a classical convolution filter (inspired by Quanvolution)
and a fully connected layer (inspired by FCL)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 convolution filter producing 4 feature maps."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class FullyConnectedLayer(nn.Module):
    """Simple fully connected layer with tanh activation and mean reduction."""
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x)).mean(dim=0, keepdim=True)

class SharedClassName(nn.Module):
    """Hybrid network: Quanvolution filter followed by a fully connected layer."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # The number of features after conv: 4 * 14 * 14 for 28x28 input
        self.fc = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["SharedClassName"]
