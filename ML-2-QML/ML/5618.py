"""Classical implementation of the hybrid binary classifier.

This module defines a classical surrogate for the quantum
components used in the original hybrid architecture.  The
class `HybridBinaryClassifier` mirrors the interface of the
quantum‑enabled version but replaces quantum layers with
efficient PyTorch equivalents.  The design intentionally
captures the structure of the original model (convolutional
backbone, filter head, fully‑connected head, sigmoid head)
while staying fully classical.

Key components:
  * `HybridFunction` – differentiable sigmoid head.
  * `ClassicalFilter` – a lightweight 2×2 convolutional filter
    with a thresholded sigmoid activation, mimicking the
    quantum‐filter behaviour.
  * `ClassicalFullyConnected` – a single linear layer that
    aggregates the flattened features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""

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


class ClassicalFilter(nn.Module):
    """Classical 2×2 filter that emulates the quantum filter behaviour."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # stride 2 produces a 14×14 feature map from a 28×28 image
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations  # shape (batch, 1, 14, 14)


class ClassicalFullyConnected(nn.Module):
    """Linear head that aggregates the flattened features."""

    def __init__(self, in_features: int = 15 * 7 * 7) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class HybridBinaryClassifier(nn.Module):
    """Hybrid binary classifier that uses classical surrogates for quantum layers."""

    def __init__(self, shift: float = 0.0, threshold: float = 0.0) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Classical filter head
        self.filter_head = ClassicalFilter(kernel_size=2, threshold=threshold)

        # Fully‑connected head
        # Compute flattened size after conv layers
        dummy_input = torch.zeros(1, 1, 28, 28)
        x = self.filter_head(dummy_input)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        flat_features = x.shape[1]
        self.fc_head = ClassicalFullyConnected(in_features=flat_features)

        self.shift = shift
        self.hybrid = HybridFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Filter head
        x = self.filter_head(inputs)  # (batch,1,14,14)
        # Convolutional backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        # Fully‑connected head
        x = self.fc_head(x)
        # Hybrid sigmoid
        probs = self.hybrid(x, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridBinaryClassifier", "HybridFunction"]
