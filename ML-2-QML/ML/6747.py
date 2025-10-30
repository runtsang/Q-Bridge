"""Hybrid binary classifier with classical fully connected head.

This module implements a convolutional feature extractor followed by a
parameterised fully connected layer that is *classically* emulated.
The design mirrors the quantum reference but replaces the quantum expectation
layer with a differentiable tanh‑based head, allowing the model to train
entirely on a CPU or GPU without a quantum backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable


class ClassicalFullyConnectedHead(nn.Module):
    """Classical analogue of the quantum fully connected layer used in the
    original reference. It applies a linear transform followed by a tanh
    activation and returns the mean expectation value.
    """
    def __init__(self, in_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs expected to be a 1D tensor of shape (batch,)
        values = inputs.view(-1, 1).float()
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation


class HybridBinaryClassifier(nn.Module):
    """Convolutional binary classifier with a classical fully connected head.

    The architecture follows the original QCNet but replaces the quantum
    expectation layer with the ClassicalFullyConnectedHead.  This allows
    the model to be trained with standard back‑propagation while
    preserving the overall feature extraction pipeline.
    """
    def __init__(self, use_quantum: bool = False) -> None:
        super().__init__()
        if use_quantum:
            raise ValueError("Quantum head is not available in the classical module.")
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Classical head
        self.head = ClassicalFullyConnectedHead(self.fc3.out_features)

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
        x = self.fc3(x)
        # Pass through classical head
        logits = self.head(x)
        # Convert logits to probabilities
        prob = torch.sigmoid(logits)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["ClassicalFullyConnectedHead", "HybridBinaryClassifier"]
