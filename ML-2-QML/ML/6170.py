"""HybridQCNet – classical PyTorch implementation with a sigmoid head.

The model mirrors the original hybrid architecture but replaces the quantum
circuit with a lightweight differentiable layer.  The design allows easy
switching to the quantum version by changing the `Hybrid` class.

Author: gpt-oss-20b
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that emulates the quantum expectation."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None


class Hybrid(nn.Module):
    """Simple linear layer followed by a sigmoid activation."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class HybridQCNet(nn.Module):
    """CNN + fully‑connected head with a classical sigmoid output."""

    def __init__(self, shift: float = 0.0) -> None:
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

        # Classical hybrid head
        self.hybrid = Hybrid(self.fc3.out_features, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
        probs = self.hybrid(x).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)


__all__ = ["HybridFunction", "Hybrid", "HybridQCNet"]
