"""Classical-only variant of the hybrid Quantum‑NAT architecture.

The module implements a CNN backbone, a dense head, and a differentiable
sigmoid activation that mirrors the quantum expectation used in the
original hybrid implementation.  It can be used for both binary
classification and regression by simply changing the final linear layer
or the activation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that emulates a quantum expectation head."""
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


class Hybrid(nn.Module):
    """Dense head that replaces the quantum circuit in the hybrid model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class QuantumNATGen191(nn.Module):
    """Classical CNN + dense head suitable for classification or regression."""
    def __init__(self, num_input_channels: int = 1, num_classes: int = 1,
                 use_sigmoid: bool = True, shift: float = 0.0) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flatten to 16*7*7 = 784 for 28x28 input
        self.flatten = nn.Flatten()
        # Dense projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        # Classification / regression head
        self.head = Hybrid(4, shift) if use_sigmoid else nn.Linear(4, num_classes)
        self.use_sigmoid = use_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        flattened = self.flatten(features)
        out = self.fc(flattened)
        out = self.norm(out)
        if self.use_sigmoid:
            logits = self.head(out)
            return torch.cat((logits, 1 - logits), dim=-1)
        return self.head(out)


__all__ = ["HybridFunction", "Hybrid", "QuantumNATGen191"]
