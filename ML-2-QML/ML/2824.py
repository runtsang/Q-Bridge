"""Classical hybrid classifier with a quantum‑inspired head.

This module implements a CNN backbone followed by a hybrid head that
mimics a quantum expectation value.  The head can be replaced with a
fully classical dense layer, but the current implementation uses a
differentiable sigmoid that is intended to be swapped with a quantum
circuit during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantumExpectationApprox(torch.autograd.Function):
    """Approximate quantum expectation with a sigmoid + linear transform."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float, scale: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.scale = scale
        out = torch.sigmoid(inputs + shift) * scale
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        out, = ctx.saved_tensors
        grad = grad_output * out * (1 - out) * ctx.scale
        return grad, None, None


class HybridHead(nn.Module):
    """Hybrid head that forwards a linear activation through a
    quantum expectation approximation.  The shift and scale parameters
    can be tuned to match the output range of a real quantum circuit.
    """
    def __init__(self, in_features: int, shift: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return QuantumExpectationApprox.apply(logits, self.shift, self.scale)


class HybridClassifier(nn.Module):
    """CNN backbone with a hybrid head for binary classification."""
    def __init__(self, shift: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Compute flattened size after conv layers
        dummy = torch.zeros(1, 3, 32, 32)
        x = self._forward_conv(dummy)
        flat_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridHead(1, shift=shift, scale=scale)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridClassifier", "HybridHead", "QuantumExpectationApprox"]
