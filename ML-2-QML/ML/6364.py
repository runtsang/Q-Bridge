"""Hybrid classical‑quantum convolutional network with QCNN‑style head.

This module implements a purely classical replica of the quantum
hybrid model.  The backbone (conv → FC) is identical to the
original `QCNet`.  The quantum head is replaced by a classical
`QCNNModel` that emulates the convolution‑pooling layers of the QCNN
ansatz.  Keeping the dimensionality of the feature vector (8) the
same as the quantum head allows direct comparison of training
behaviour and loss landscapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the classical QCNN surrogate defined in QCNN.py
from.QCNN import QCNNModel


class HybridQCNNNet(nn.Module):
    """Convolutional backbone followed by a QCNN‑style classical head."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        # 8‑dimensional output feeds into the QCNN head
        self.fc3 = nn.Linear(84, 8)

        # Classical QCNN surrogate
        self.qcnn_head = QCNNModel()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        x = self.fc3(x)  # shape (batch, 8)
        probs = self.qcnn_head(x)  # shape (batch, 1)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridQCNNNet"]
