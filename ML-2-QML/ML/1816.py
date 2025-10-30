"""Hybrid classical‑quantum classifier with a shallow neural encoder and a variational quantum layer."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
#  Classical feature encoder
# --------------------------------------------------------------------------- #
class ClassicalEncoder(nn.Module):
    """A simple feature‑wise linear encoder that transforms input features into a larger hidden space."""

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_features, hidden_dim)
        # Bias optional: keep default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).relu()


# --------------------------------------------------------------------------- #
#  Hybrid classifier
# --------------------------------------------------------------------------- #
class QuantumClassifier(nn.Module):
    """
    Combines a classical encoder with a quantum circuit that outputs a probability
    distribution over two classes.
    The quantum part is represented as a placeholder PyTorch Module that forwards
    the encoded features through a simulated variational circuit.
    """

    def __init__(self, in_features: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.encoder = ClassicalEncoder(in_features, hidden_dim)
        # The quantum circuit is simulated via a custom autograd Function below
        self.quantum_layer = QuantumLayer(hidden_dim, depth)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder → quantum layer → classification head."""
        encoded = self.encoder(x)
        quantum_out = self.quantum_layer(encoded)
        logits = self.head(quantum_out)
        return logits


# --------------------------------------------------------------------------- #
#  Quantum layer simulation (autograd friendly)
# --------------------------------------------------------------------------- #
class QuantumLayer(nn.Module):
    """
    Simulate a variational quantum circuit with parameters shared across all data points.
    The layer is differentiable thanks to a custom autograd Function that emits a
    simple linear transformation parameterized by a learnable weight matrix.
    """

    def __init__(self, hidden_dim: int, depth: int) -> None:
        super().__init__()
        # We use a learnable matrix that mimics a quantum circuit's output
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For demonstration we apply a simple power‑of‑depth transformation.
        # In practice this would be replaced by a true quantum simulation.
        out = x
        for _ in range(self.depth):
            out = torch.matmul(out, self.weight)
            out = out.relu()
        return out


# --------------------------------------------------------------------------- #
#  Builder function
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a hybrid network with a classical encoder and a simulated quantum layer.
    Returns (model, encoding_indices, weight_sizes, observables).
    """
    hidden_dim = num_features * 2  # example expansion
    model = QuantumClassifier(num_features, hidden_dim, depth)

    # Encoding indices correspond to the original features
    encoding_indices = list(range(num_features))
    # Weight sizes: encoder, quantum layer, head
    weight_sizes = [
        model.encoder.encoder.weight.numel() + model.encoder.encoder.bias.numel(),
        model.quantum_layer.weight.numel(),
        model.head.weight.numel() + model.head.bias.numel(),
    ]
    # Observables placeholder: just indices of the output logits
    observables = [0, 1]
    return model, encoding_indices, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
