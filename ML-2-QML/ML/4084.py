"""Classical hybrid model combining a quanvolution filter, a classical dense head, and a quantum‑style expectation layer.

The architecture merges the ideas from:
- the classical quanvolution filter (Pair 1),
- the hybrid head that mimics a quantum expectation value (Pair 2),
- and a tiny regression head inspired by EstimatorQNN (Pair 3).

The network can be used for binary classification (output probabilities) and a lightweight regression task in parallel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQuanvolutionFilter(nn.Module):
    """
    2×2 patch convolution that emulates the classical part of a quanvolution layer.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the spatial dimensions after the convolution.
        return self.conv(x).view(x.size(0), -1)

class HybridFunction(torch.autograd.Function):
    """
    Differentiable sigmoid that behaves like a quantum expectation value.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        outputs, = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

class Hybrid(nn.Module):
    """
    Dense head that forwards activations through the hybrid function.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class EstimatorQNN(nn.Module):
    """
    Tiny regression network inspired by the EstimatorQNN example.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class QuanvolutionHybridNet(nn.Module):
    """
    Complete hybrid network that combines the classical quanvolution filter,
    a dense classification head, and a regression head.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        # Classical quanvolution filter (Pair 1)
        self.qfilter = ClassicalQuanvolutionFilter(in_channels, 4)
        # Dense head (Pair 2)
        self.fc = nn.Linear(4 * 14 * 14, 120)
        self.dropout = nn.Dropout(0.5)
        # Regression head (Pair 3)
        self.regressor = EstimatorQNN()
        # Hybrid expectation layer
        self.hybrid = Hybrid(120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features with the quanvolution filter
        features = self.qfilter(x)
        # Classification head
        x = F.relu(self.fc(features))
        x = self.dropout(x)
        logits = self.hybrid(x)
        # Binary probabilities
        probs = torch.cat([logits, 1 - logits], dim=-1)
        # Regression output (uses first two features as dummy input)
        reg = self.regressor(x[:, :2])
        return probs, reg
