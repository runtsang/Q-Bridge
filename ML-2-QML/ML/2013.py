"""Classical neural network with a hybrid sigmoid head for binary classification.

This module extends the original seed by adding a configurable shift and a
dropout‑regularised hybrid head that can be swapped with a quantum version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidShiftFunction(torch.autograd.Function):
    """Differentiable sigmoid with a learnable shift."""
    @staticmethod
    def forward(ctx, logits, shift):
        ctx.save_for_backward(logits, shift)
        return torch.sigmoid(logits + shift)

    @staticmethod
    def backward(ctx, grad_output):
        logits, shift = ctx.saved_tensors
        sig = torch.sigmoid(logits + shift)
        return grad_output * sig * (1 - sig), None

class HybridHead(nn.Module):
    """Hybrid activation head that can be replaced by a quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = nn.Parameter(torch.tensor(shift, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        return SigmoidShiftFunction.apply(logits, self.shift)

class HybridNet(nn.Module):
    """Convolutional network with a hybrid head for binary classification."""
    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=dropout)

        # The number 55815 is the flattened size after the conv layers for 32x32 inputs.
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridHead(self.fc3.out_features)

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
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridHead", "HybridNet", "SigmoidShiftFunction"]
