"""Classical hybrid CNN for binary classification.

This module defines a purely classical counterpart of the hybrid quantum
network.  The architecture mirrors the quantum version but replaces the
quantum expectation head with a differentiable sigmoid head.  The
class can be used as a drop‑in replacement in experiments where a
classical baseline is needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalHybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that emulates a quantum expectation."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class ClassicalHybrid(nn.Module):
    """Classical head that replaces the quantum block."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return ClassicalHybridFunction.apply(self.linear(logits), self.shift)


class HybridQuantumCNN(nn.Module):
    """Classical CNN that mimics the hybrid quantum binary classifier."""

    def __init__(self, n_qubits: int = 4, shift: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Flatten size depends on input image size; assume 32x32.
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_qubits)
        self.hybrid = ClassicalHybrid(n_qubits, shift=shift)

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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["ClassicalHybridFunction", "ClassicalHybrid", "HybridQuantumCNN"]
