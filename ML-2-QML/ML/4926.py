"""Classical implementation of the hybrid Quanvolution model.

This module mirrors the structure of the original Quanvolution
filter while replacing the quantum expectation head with a
purely classical sigmoid‐based hybrid layer.  It can be used
directly with PyTorch training loops or as a baseline for
quantum experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridFunction(torch.autograd.Function):
    """Purely classical sigmoid activation that mimics a quantum expectation."""

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
    """Dense head that replaces the quantum circuit in the original model."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)


class QuanvolutionFilter(nn.Module):
    """Classic 2×2 patch extractor with a single convolutional layer."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionHybrid(nn.Module):
    """Hybrid model that replaces the quantum head with a classical sigmoid head."""

    def __init__(self, num_classes: int = 10, shift: float = 0.0) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        # Feature dimension after flattening: 4 * 14 * 14
        self.fc1 = nn.Linear(4 * 14 * 14, 1)
        self.hybrid = Hybrid(1, shift)
        self.fc2 = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        hidden = self.fc1(features)
        hybrid_out = self.hybrid(hidden)
        logits = self.fc2(hybrid_out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridFunction", "Hybrid", "QuanvolutionFilter", "QuanvolutionHybrid"]
