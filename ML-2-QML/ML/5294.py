"""
Classical implementation of the hybrid quanvolution network.
This module mirrors the structure of the original Quanvolution example,
but replaces the quantum filter with a lightweight 2×2 convolution
and the quantum head with a differentiable sigmoid layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """Drop‑in classical replacement for the quantum filter.

    Uses a single 2×2 convolution with a sigmoid activation and
    a user‑settable threshold to mimic the quantum measurement
    behaviour.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean((2, 3), keepdim=True)


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation mimicking a quantum expectation."""
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
    """Classical dense head that optionally adds a shift."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class QuanvolutionHybridClassifier(nn.Module):
    """
    Hybrid network that accepts 1‑channel images, applies either a classical
    convolution or a quantum filter, and produces binary class probabilities.
    """
    def __init__(
        self,
        use_quantum_filter: bool = False,
        use_quantum_head: bool = False,
        threshold: float = 0.0,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_quantum_filter = use_quantum_filter
        self.use_quantum_head = use_quantum_head

        # Filter stage
        if self.use_quantum_filter:
            # Placeholder: user can plug in a quantum filter implementation
            raise NotImplementedError("Quantum filter not available in classical build.")
        else:
            self.filter = ConvFilter(kernel_size=2, threshold=threshold)

        # Head stage
        if self.use_quantum_head:
            # Placeholder: user can plug in a quantum head implementation
            raise NotImplementedError("Quantum head not available in classical build.")
        else:
            self.head = Hybrid(in_features=1, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Log probabilities for two classes (class 1 and class 0).
        """
        features = self.filter(x)
        logits = self.head(features)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return F.log_softmax(probs, dim=-1)


__all__ = ["QuanvolutionHybridClassifier"]
