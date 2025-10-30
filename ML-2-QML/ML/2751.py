"""Classical hybrid network integrating a classical quanvolution filter
and a dense head.

This module mirrors the architecture of the original hybrid QCNet while
replacing the quantum expectation head with a fully classical sigmoid
head.  A classical QuanvolutionFilter is inserted after the convolutional
backbone to emulate the quantum kernel behaviour from the QML seed."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridQuanvolutionNet", "ClassicalQuanvolutionFilter"]

class ClassicalQuanvolutionFilter(nn.Module):
    """Simple 2×2 convolution acting as a classical analogue of the quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Expected input shape: (batch, 1, 28, 28)
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridQuanvolutionNet(nn.Module):
    """Classical counterpart to the hybrid QCNet, augmented with a quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone (same as original QCNet)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical quanvolution filter
        self.qfilter = ClassicalQuanvolutionFilter()
        # Linear head producing a single logit
        self.classifier = nn.Linear(4 * 14 * 14, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Convert scalar to 1×28×28 feature map
        x_img = F.adaptive_avg_pool2d(x.unsqueeze(1), (28, 28))
        x_img = x_img.mean(dim=1, keepdim=True)

        # Apply the classical quanvolution filter
        features = self.qfilter(x_img)

        # Linear head and sigmoid activation
        logits = self.classifier(features)
        probs = self.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)
