"""Hybrid quantum-inspired classifier implemented in PyTorch.

This module extends the original QuantumClassifierModel by incorporating
photonic-inspired scaling and shift operations in each hidden layer,
mirroring the FraudDetection example.  The network is fully
classical but mimics the structure of a variational circuit:
data is fed through a stack of linear layers with Tanh activations,
followed by a final linear head.  Each hidden layer is equipped with
trainable scaling and shift buffers that emulate the displacement
operations from the photonic circuit.

The public API is identical to the original build_classifier_circuit
function: it returns a tuple (nn.Module, encoding, weight_sizes,
observables).  The encoding is simply a list of input feature indices,
weight_sizes records the number of learnable parameters per linear
layer, and observables are trivial class labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn

@dataclass
class LayerParams:
    """Parameters describing a single hidden layer."""
    clip: bool = False
    bound: float = 5.0

def _layer_with_scaling(in_features: int,
                        out_features: int,
                        clip: bool = False,
                        bound: float = 5.0) -> nn.Module:
    """Return a linear‑Tanh‑scale‑shift block."""
    linear = nn.Linear(in_features, out_features)
    activation = nn.Tanh()
    scale = nn.Parameter(torch.ones(out_features))
    shift = nn.Parameter(torch.zeros(out_features))
    if clip:
        with torch.no_grad():
            linear.weight.clamp_(-bound, bound)
            linear.bias.clamp_(-bound, bound)
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_parameter("scale", scale)
            self.register_parameter("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift
    return Block()

def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a classical feed‑forward classifier with a depth of hidden
    layers.  The first layer is unclipped; subsequent layers are clipped
    to avoid exploding gradients, mimicking the photonic clipping in
    FraudDetection.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for i in range(depth):
        layer = _layer_with_scaling(in_dim, num_features,
                                    clip=(i > 0), bound=5.0)
        layers.append(layer)
        weight_sizes.append(layer.linear.weight.numel() + layer.linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]
    return network, encoding, weight_sizes, observables

class HybridClassifier(nn.Module):
    """Convenience wrapper exposing a consistent API."""
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

__all__ = ["build_classifier_circuit", "LayerParams", "HybridClassifier"]
