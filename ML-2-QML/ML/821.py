import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depth‑wise separable convolution (depth‑wise + point‑wise)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ResidualBlock(nn.Module):
    """Simple residual block built from depth‑wise separable convs."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return self.relu(x)

class ChannelAttention(nn.Module):
    """Squeeze‑and‑excitation style channel attention."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.mlp(y)
        y = self.sigmoid(y)
        return x * y

class QFCModel(nn.Module):
    """Classical hybrid model with depth‑wise separable convs, residuals, attention, and multi‑scale pooling."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            DepthwiseSeparableConv(1, 8),
            nn.MaxPool2d(2),
            ResidualBlock(8),
            ChannelAttention(8),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(8, 16),
            nn.MaxPool2d(2),
            ResidualBlock(16),
            ChannelAttention(16)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.norm(x)

__all__ = ["QFCModel"]
