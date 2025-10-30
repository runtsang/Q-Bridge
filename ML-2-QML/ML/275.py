"""Advanced classical backbone for hybrid binary classification.

The model extends the original QCNet by adding residual blocks, an
adaptive pooling layer, and a dense head that learns a weight matrix
to fuse the final feature vector.  No quantum operations are used
here; the class is fully compatible with CPU training pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels!= out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class AdvancedHybridBinaryClassifier(nn.Module):
    """Classical backbone with residual blocks and a dense head."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        # Residual block
        self.res_block = ResidualBlock(6, 12, kernel_size=3, stride=1)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Adaptive pooling to a fixed size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layers
        self.fc1 = nn.Linear(12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)  # 84‑dimensional feature vector
        # Classical head
        self.classifier = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.res_block(x)
        x = self.drop2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 84‑dimensional features
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["ResidualBlock", "AdvancedHybridBinaryClassifier"]
