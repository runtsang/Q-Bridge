"""Enhanced classical model with depthwise‑separable convolutions and residual skip‑connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise‑separable convolution block."""
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

class QuantumNATEnhanced(nn.Module):
    """Classical model with multi‑scale depthwise‑separable convs and residual skip‑connections."""
    def __init__(self, num_classes=4, in_channels=1):
        super().__init__()
        # Multi‑scale backbone
        self.conv1 = DepthwiseSeparableConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DepthwiseSeparableConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DepthwiseSeparableConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=4, bias=False),
            nn.BatchNorm2d(64)
        )
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        # x shape: (B, C, H, W)
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        # Residual skip
        res = self.residual(x)
        out = out + res
        # Flatten
        out_flat = out.view(out.size(0), -1)
        logits = self.fc(out_flat)
        logits = self.bn(logits)
        # Auxiliary feature vector (pre‑fc)
        aux = out_flat
        return logits, aux

__all__ = ["QuantumNATEnhanced"]
