"""Enhanced classical binary classifier with residual blocks and temperature‑scaled sigmoid.

This module defines QCNet, a convolutional neural network extended with residual connections,
batch normalisation, and a temperature‑controlled sigmoid head. The network is fully
classical and compatible with the original interface used in the hybrid quantum model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block with Conv-BN-ReLU."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class TemperatureSigmoid(nn.Module):
    """Temperature‑adjusted sigmoid activation for the binary head."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.temperature)

class QCNet(nn.Module):
    """Extended CNN with residual blocks and temperature‑controlled sigmoid head."""
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, stride=2),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        # Flatten and fully connected
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1, bias=False),
        )
        self.head = TemperatureSigmoid(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x).squeeze(-1)
        probs = self.head(logits)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["QCNet"]
