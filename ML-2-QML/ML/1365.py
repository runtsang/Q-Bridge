"""
Hybrid classical model with a residual deep CNN backbone and a quantum-inspired filter.

The module defines:
- ResidualBlock: a standard residual block with batch‑norm and ReLU.
- DeepCNNBackbone: a 3‑stage residual CNN that reduces the spatial dimension and outputs a 128‑dim feature vector.
- QuanvolutionFilter: a lightweight 2×2 convolution that extracts 4‑channel patches.
- QuanvolutionClassifier: combines the filter, the deep backbone and a linear head for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Standard residual block with batch‑norm and ReLU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class DeepCNNBackbone(nn.Module):
    """Three‑stage residual CNN that outputs a 128‑dim vector."""
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

class QuanvolutionFilter(nn.Module):
    """Simple 2×2 convolutional filter that extracts 4‑channel patches."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the quanvolution filter, a deep backbone and a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.backbone = DeepCNNBackbone()
        self.linear = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)          # shape: (B, 4, 14, 14)
        features = self.backbone(features)  # shape: (B, 128)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
