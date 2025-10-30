"""
HybridBinaryClassifier – Classical PyTorch implementation.

This module defines a convolutional backbone followed by a purely classical
dense head that mimics the behaviour of the quantum expectation layer
present in the original hybrid design.  The head is a differentiable
sigmoid that can be seamlessly swapped with its quantum counterpart
during experimentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """
    Differentiable sigmoid activation that emulates a quantum expectation.

    The forward pass applies a sigmoid with an optional shift; the backward
    pass implements the exact derivative of the sigmoid, ensuring smooth
    gradients for the dense head.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None


class ClassicalHybridHead(nn.Module):
    """
    Dense head that replaces the quantum layer in the classical baseline.

    Parameters
    ----------
    in_features : int
        Number of input features from the preceding fully‑connected layer.
    shift : float, optional
        Shift applied before the sigmoid; defaults to 0.0.
    """

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class HybridBinaryClassifier(nn.Module):
    """
    Classical convolutional network for binary classification.

    The architecture mirrors the original hybrid design but replaces the
    quantum head with a classical sigmoid head.  The network can be
    extended with additional convolutional layers or regularisation
    techniques without affecting the head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: Iterable[int] = (120, 84),
        dropout: Iterable[float] = (0.2, 0.5),
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=dropout[0])
        self.drop2 = nn.Dropout2d(p=dropout[1])
        self.fc1 = nn.Linear(55815, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        self.head = ClassicalHybridHead(self.fc3.out_features, shift=shift)

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
        prob = self.head(x)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["HybridFunction", "ClassicalHybridHead", "HybridBinaryClassifier"]
