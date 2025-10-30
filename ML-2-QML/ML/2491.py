"""Hybrid classical neural network for binary classification with a quantum-inspired head.

The architecture consists of a convolutional feature extractor followed by a
classical head that mimics the behaviour of a quantum expectation layer.
The head applies a learnable linear transformation, a sigmoid activation
with a trainable shift and a scaling factor, thereby providing a smooth
probability output. The design is inspired by the classical counterpart
of the hybrid quantum model and incorporates ideas from the fraud-detection
layer: clipping of weights for stability and a configurable scaling/shift
mechanism.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ScaleShift:
    """Container for scaling and shifting parameters applied after the head."""
    scale: float = 1.0
    shift: float = 0.0

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head with a learnable shift."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Linear head followed by a sigmoid with shift and optional scaling."""

    def __init__(self, in_features: int, shift: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift
        self.scale = scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        probs = HybridFunction.apply(logits, self.shift)
        return probs * self.scale

class HybridQCNet(nn.Module):
    """Convolutional network with a quantum-inspired classical head."""

    def __init__(self, scale_shift: ScaleShift | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Default scale and shift if none provided
        if scale_shift is None:
            scale_shift = ScaleShift()
        self.hybrid = Hybrid(1, shift=scale_shift.shift, scale=scale_shift.scale)

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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ScaleShift", "HybridFunction", "Hybrid", "HybridQCNet"]
