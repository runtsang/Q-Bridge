"""Classical implementation of a hybrid quantum‑kernel inspired binary classifier.

This module defines a neural network that processes images through convolutional layers,
flattens the feature map, and passes the representation through a trainable
RBF kernel layer followed by a sigmoid head. The kernel prototypes are learnable
parameters, enabling the model to mimic a quantum kernel's behaviour while
remaining fully classical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalRBFKernel(nn.Module):
    """Trainable RBF kernel layer using learnable prototypes."""
    def __init__(self, in_features: int, num_prototypes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, in_features))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (batch, prototypes, in_features)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (batch, prototypes)
        return torch.exp(-self.gamma * dist_sq)  # (batch, prototypes)

class HybridQuantumKernelNet(nn.Module):
    """Classical hybrid network that mimics the quantum hybrid classifier."""
    def __init__(self, num_prototypes: int = 10, gamma: float = 1.0) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Kernel head
        self.kernel = ClassicalRBFKernel(in_features=1, num_prototypes=num_prototypes, gamma=gamma)
        # Final linear layer to map kernel similarities to logits
        self.kernel_head = nn.Linear(num_prototypes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extractor
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
        x = self.fc3(x)  # (batch, 1)
        # Kernel similarity
        kernel_sim = self.kernel(x)  # (batch, prototypes)
        logits = self.kernel_head(kernel_sim)  # (batch, 1)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumKernelNet"]
