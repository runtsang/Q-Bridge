"""Hybrid binary classifier with classical CNN and QCNN analog head.

This module defines a pure classical implementation of the hybrid architecture.
The head is a fully connected network that mimics the QCNN quantum circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QCNNModel(nn.Module):
    """Fully connected network that emulates quantum convolution layers."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridBinaryClassifier(nn.Module):
    """CNN followed by a QCNN analog head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Reduce high-dimensional flattened feature to 8 features for QCNN analog
        self.fc_reduce = nn.Linear(540, 8)
        self.qcnn_head = QCNNModel()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.fc_reduce(x)
        probs = self.qcnn_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier", "QCNNModel"]
