"""
Classical implementation of the hybrid binary classifier.
The network mirrors the quantum architecture but replaces the quantum
expectation head with a parametrised sigmoid function that can be
trained by back‑propagation.  This makes the model fully
PyTorch‑compatible while still exposing the same API as the quantum
counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head mimicking a quantum expectation."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        logits = inputs + shift
        outputs = torch.sigmoid(logits)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1.0 - outputs)
        return grad_inputs, None


class ClassicalHybridHead(nn.Module):
    """Linear layer followed by a parametrised sigmoid."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs).squeeze(-1)
        return HybridFunction.apply(logits, self.shift)


class HybridBinaryClassifier(nn.Module):
    """
    Classical CNN backbone with a hybrid sigmoid head.
    Mirrors the layer architecture of the quantum version so that
    both models consume the same feature maps.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional front‑end
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully‑connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid head
        self.head = ClassicalHybridHead(1, shift=0.0)

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
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridFunction", "ClassicalHybridHead", "HybridBinaryClassifier"]
