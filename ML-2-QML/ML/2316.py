"""Hybrid classical QCNN model inspired by QCNN and Quanvolution."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """Classical network that mimics a quantum convolutional neural network
    and incorporates a quanvolution-inspired random feature map."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Feature extraction: 2x2 patches -> 4 channels
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Quantum-inspired random linear kernel (fixed orthogonal matrix)
        self.random_kernel = nn.Linear(4 * 14 * 14, 4 * 14 * 14, bias=False)
        nn.init.orthogonal_(self.random_kernel.weight)
        # Classical convolutional layers
        self.conv1 = nn.Conv2d(4, 8, kernel_size=2, stride=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(2)
        # Classifier
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 2 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Apply quanvolution filter
        x = self.qfilter(x)  # shape: [B, 4, 14, 14]
        # Flatten patches and apply random kernel
        x_flat = x.view(x.size(0), -1)
        x_flat = self.random_kernel(x_flat)
        x = x_flat.view(x.size(0), 4, 14, 14)
        # Classical conv + pool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)
