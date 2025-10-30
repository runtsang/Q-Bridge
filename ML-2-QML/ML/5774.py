"""Pure‑Python hybrid classifier for binary tasks.

The module mirrors the structure of the original hybrid network but
replaces the quantum expectation with a lightweight classical head.
It is fully compatible with PyTorch and can be dropped into existing
training pipelines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalHybridFunction(torch.autograd.Function):
    """
    Forward pass through a simple linear layer followed by a sigmoid.
    Mimics the quantum expectation for consistency with the QML variant.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        logits = inputs + shift
        probs = torch.sigmoid(logits)
        ctx.save_for_backward(probs)
        return probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (probs,) = ctx.saved_tensors
        grad_inputs = grad_output * probs * (1 - probs)
        return grad_inputs, None


class ClassicalHybridLayer(nn.Module):
    """
    Classical head that can be swapped with a quantum layer at runtime.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return ClassicalHybridFunction.apply(logits, self.shift)


class HybridClassifier(nn.Module):
    """
    Convolutional backbone followed by a configurable hybrid head.
    """
    def __init__(self, use_quantum: bool = False, **kwargs) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Flattened size after conv layers for 32x32 RGB inputs
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        if self.use_quantum:
            from.qml_code import QuantumHybridLayer  # type: ignore
            self.hybrid = QuantumHybridLayer(self.fc3.out_features, **kwargs)
        else:
            self.hybrid = ClassicalHybridLayer(self.fc3.out_features, **kwargs)

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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["ClassicalHybridFunction", "ClassicalHybridLayer", "HybridClassifier"]
