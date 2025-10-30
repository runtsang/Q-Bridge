"""QuantumHybridBinaryClassifier – classical counterpart with attention and multi‑head heads.

This module implements a PyTorch model that mirrors the original
architecture, but extends it with a learned attention mechanism
and a multi‑head dense layer that produces two logits for binary
classification.  The forward pass is fully differentiable and
compatible with standard optimizers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """Learned spatial attention for feature maps."""
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = torch.mean(x, dim=(2, 3), keepdim=True)
        attn = self.conv1(gap)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        return x * torch.sigmoid(attn)

class HybridFunction(nn.Module):
    """Simple dense head that mimics the quantum expectation head."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits + self.shift)

class QCNet(nn.Module):
    """CNN-based binary classifier with attention and hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.attn1 = AttentionBlock(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.attn2 = AttentionBlock(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = HybridFunction(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.attn1(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.attn2(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["AttentionBlock", "HybridFunction", "QCNet"]
