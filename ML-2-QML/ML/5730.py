"""Hybrid classical convolutional network with a differentiable quantum-inspired head.

The model reuses the CNN backbone of the original Quantum‑NAT,
but replaces the quantum layer with a lightweight
hybrid head that mimics a quantum expectation via a sigmoid
activation.  This permits a fully classical implementation while
retaining the expressive power of the original design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that emulates a quantum expectation value."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1.0 - outputs)
        return grad_inputs, None


class HybridHead(nn.Module):
    """Linear head followed by HybridFunction."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class HybridQFCModel(nn.Module):
    """Classical CNN + hybrid expectation head inspired by Quantum‑NAT."""

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor identical to the original classical QFCModel
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Flattened dimension: 16 * 7 * 7
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

        # Hybrid head mimicking the quantum layer
        self.hybrid = HybridHead(4, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)
        # Pass through hybrid head to produce binary probability
        prob = self.hybrid(out)
        return torch.cat((prob, 1 - prob), dim=-1)


__all__ = ["HybridQFCModel"]
