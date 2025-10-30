"""Hybrid classical-quantum binary classifier – classical implementation.

This module implements a fully classical neural network that mimics the
architecture of the original hybrid model while replacing the quantum
components with classical approximations.  The design incorporates
the following ideas from the seed projects:

* Dense head with a differentiable sigmoid (HybridFunction) – from
  ClassicalQuantumBinaryClassification.py.
* Classical convolutional filter that emulates the quanvolution
  layer – from Conv.py.
* Drop‑out, pooling and fully‑connected stages identical to the
  original CNN backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvFilter(nn.Module):
    """
    Classical emulation of the quanvolution filter.
    Uses a depth‑wise convolution followed by a sigmoid threshold.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, in_channels: int = 15):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # depth‑wise conv: one filter per input channel
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape (batch, channels, h, w).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, channels) containing the average
            sigmoid activation for each channel.
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # average over spatial dimensions
        return activations.mean(dim=(2, 3))

class HybridFunction(torch.autograd.Function):
    """Simple differentiable sigmoid head used as the classical hybrid."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class HybridQCNet(nn.Module):
    """Classical CNN with a classical quanvolution filter and a sigmoid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(15, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum_filter = ClassicalQuanvFilter(kernel_size=2, threshold=0.0, in_channels=15)
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        # apply classical quanvolution filter
        x = self.quantum_filter(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)

__all__ = ["HybridQCNet"]
