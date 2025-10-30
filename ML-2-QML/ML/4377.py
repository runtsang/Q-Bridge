"""Hybrid classical-quantum quanvolution network (classical implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

# Classical quanvolution filter
class QuanvolutionFilter(nn.Module):
    """2x2 patch-based convolution using a small CNN."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

# Hybrid head approximating quantum expectation
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation mimicking quantum expectation."""
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
    """Dense layer with sigmoid shift used as a quantum surrogate."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

# Build a classical classifier mirroring the quantum interface
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier with metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# Simple sampler network
class SamplerQNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

# Main hybrid quanvolution network
class HybridQuanvolutionNet(nn.Module):
    """Classical network with optional quantum‑style head."""
    def __init__(self, num_classes: int = 2, use_quantum_head: bool = False,
                 shift: float = 0.0, depth: int = 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)
        self.use_quantum_head = use_quantum_head
        if use_quantum_head:
            self.hybrid = Hybrid(4 * 14 * 14, shift=shift)
        else:
            self.hybrid = None
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        if self.hybrid is not None:
            logits = self.hybrid(features)
        else:
            logits = self.classifier(features)
        sampler_out = self.sampler(logits)
        return F.log_softmax(logits, dim=-1), sampler_out

__all__ = [
    "QuanvolutionFilter",
    "HybridFunction",
    "Hybrid",
    "build_classifier_circuit",
    "SamplerQNN",
    "HybridQuanvolutionNet",
]
