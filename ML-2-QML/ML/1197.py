import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class QuanvolutionFilter(nn.Module):
    """Classical depthwise separable convolution with optional dropout."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2, dropout: float = 0.1):
        super().__init__()
        # Depthwise convolution: groups=in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.dropout(x)
        # Flatten to (batch, -1)
        return x.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier using the classical quanvolution filter followed by a small MLP head."""
    def __init__(self, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # The filter outputs 4 * 14 * 14 features for 28x28 MNIST images
        self.mlp = nn.Sequential(
            nn.Linear(4 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
