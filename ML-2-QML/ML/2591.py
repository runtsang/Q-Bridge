"""Classical binary classifier with optional quantum‑inspired filter.

This module defines HybridQCNet, a CNN with a hybrid sigmoid head, and
provides a ClassicalConvFilter that emulates a quantum quanvolution
filter for speed or as a drop‑in replacement.

The architecture mirrors the original hybrid model but replaces the
quantum circuit with a learnable linear layer and a sigmoid
activation.  The filter can be inserted before the first convolution
layer to pre‑process the image.

Both the network and the filter expose a consistent interface that
mirrors the quantum counterpart, enabling easy swapping between the
classical and quantum implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalConvFilter(nn.Module):
    """Drop‑in replacement for a quantum quanvolution filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square patch to process.
    threshold : float, default 0.0
        Threshold applied to the convolution bias before sigmoid.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # A single‑channel 1×1 conv to mimic the single‑output
        # of the quantum filter.
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the filter to a batch of images.

        The input is expected to have shape (B, C, H, W) with C=1.
        The output is a scalar per image, averaged over all
        convolution windows.
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # (B, 1)


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that emulates the quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Hybrid head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class HybridQCNet(nn.Module):
    """CNN followed by a hybrid sigmoid head.

    The architecture is identical to the original hybrid model but
    uses a classical linear layer instead of a quantum circuit.
    """
    def __init__(self, use_filter: bool = False, filter_kwargs: dict | None = None) -> None:
        super().__init__()
        self.use_filter = use_filter
        if use_filter:
            self.filter = ClassicalConvFilter(**(filter_kwargs or {}))
        else:
            self.filter = None

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Optional quanvolution replacement
        if self.use_filter:
            # Convert RGB to single channel by averaging
            gray = inputs.mean(dim=1, keepdim=True)
            inputs = self.filter(gray)

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


__all__ = ["HybridQCNet", "ClassicalConvFilter", "HybridFunction", "Hybrid"]
