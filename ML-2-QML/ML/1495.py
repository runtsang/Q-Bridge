"""Classical hybrid binary classifier with an enhanced residual CNN backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A lightweight residual block with optional down‑sampling."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = (stride!= 1 or in_channels!= out_channels)
        if self.downsample:
            self.avg = nn.AvgPool2d(kernel_size=stride, stride=stride)
            self.conv_skip = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1, stride=stride, bias=False)
            self.bn_skip = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.conv_skip(self.avg(identity))
            identity = self.bn_skip(identity)
        out += identity
        return self.relu(out)


class HybridHead(nn.Module):
    """Classical head that maps high‑level features to a binary probability."""
    def __init__(self, in_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)


class HybridBinaryNet(nn.Module):
    """Deep residual CNN followed by a hybrid classical head."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.head = HybridHead(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)


__all__ = ["HybridBinaryNet"]
