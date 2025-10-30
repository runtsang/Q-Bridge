"""Enhanced classical binary classifier with attention and a learnable shift for the quantum head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumHybridClassifier(nn.Module):
    """Classical backbone with attention and a learnable shift for the quantum head."""

    def __init__(self, shift_init: float = 0.0) -> None:
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
        self.fc3 = nn.Linear(84, 1)
        # Attention weights
        self.attn = nn.Linear(84, 84, bias=False)
        # Learnable shift for the hybrid head
        self.shift = nn.Parameter(torch.tensor(shift_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        # Flatten
        x = torch.flatten(x, 1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        # Attention weighting
        attn_weights = torch.sigmoid(self.attn(x))
        x = x * attn_weights
        x = self.fc3(x)
        # Hybrid head: sigmoid with learnable shift
        probs = torch.sigmoid(x + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier"]
