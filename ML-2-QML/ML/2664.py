"""Classical hybrid network with a classical quanvolution filter and a linear head for binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalQuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution filter that mimics the quantum quanvolution filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, H, W]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class ConvBranch(nn.Module):
    """Convolutional backbone producing a probability."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.fc3(x)  # shape: [batch, 1]
        probs = torch.sigmoid(x)
        return probs


class QuanvolutionBranch(nn.Module):
    """Classical quanvolution filter branch producing a probability."""
    def __init__(self) -> None:
        super().__init__()
        self.quanvolution = ClassicalQuanvolutionFilter()
        # 32×32 input → 15×15 output after 2×2 stride 2 → 4*15*15 = 900
        self.linear = nn.Linear(900, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)  # [batch, 1, H, W]
        features = self.quanvolution(gray)
        logits = self.linear(features)
        probs = torch.sigmoid(logits)
        return probs


class HybridQuanvolutionNet(nn.Module):
    """Classical hybrid network combining conv and quanvolution branches."""
    def __init__(self) -> None:
        super().__init__()
        self.conv_branch = ConvBranch()
        self.quanvolution_branch = QuanvolutionBranch()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs_conv = self.conv_branch(x)  # [batch, 1]
        probs_q = self.quanvolution_branch(x)  # [batch, 1]
        probs = (probs_conv + probs_q) / 2
        return torch.cat((probs, 1 - probs), dim=-1)
