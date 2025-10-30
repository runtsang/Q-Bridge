"""Classical hybrid network combining a classical quanvolution filter with convolutional layers and a dense head for binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch filter implemented as a single convolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridQuanvolutionNet(nn.Module):
    """Hybrid network that first applies a classical quanvolution filter,
    then standard convolutional layers, and finally outputs class probabilities."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected head
        self.fc1 = nn.Linear(15 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply classical quanvolution filter
        x = self.qfilter(x)  # shape: (batch, 4*14*14)
        x = x.view(x.size(0), 4, 14, 14)  # reshape to (batch, 4, 14, 14)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuanvolutionFilter", "HybridQuanvolutionNet"]
