"""Hybrid classical‑quantum kernel binary classifier (classical version).

This module combines a CNN backbone with a hybrid head that uses a
classical RBF kernel over learned prototype vectors.  The design mirrors
the quantum implementation in the sibling module while keeping all
operations on the CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple RBF kernel implementation
class RBFKernel(nn.Module):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x - y||^2)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# Sigmoid activation with an optional shift
class ShiftedSigmoid(nn.Module):
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class HybridQuantumKernelNet(nn.Module):
    """CNN backbone followed by a hybrid head using a classical RBF kernel."""

    def __init__(
        self,
        num_prototypes: int = 8,
        gamma: float = 1.0,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Prototype vectors that will be compared with the feature vector
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, self.fc3.out_features))
        self.kernel = RBFKernel(gamma)
        self.kernel_linear = nn.Linear(num_prototypes, 1)

        # Activation
        self.shifted_sigmoid = ShiftedSigmoid(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
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
        x = self.fc3(x)  # shape: (batch, 1)

        # Kernel head
        feat = x.squeeze(-1)  # shape: (batch,)
        sims = torch.stack([self.kernel(feat, p) for p in self.prototypes], dim=1).squeeze(-1)  # (batch, num_prototypes)
        kernel_logits = self.kernel_linear(sims)  # (batch, 1)

        # Total logit
        logits = x + kernel_logits  # (batch, 1)
        probs = self.shifted_sigmoid(logits)

        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumKernelNet"]
