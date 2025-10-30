"""Hybrid classical-quantum classifier factory using PyTorch.

The module provides a feed‑forward network that mirrors the quantum
ansatz signature, an estimator that can add Gaussian shot noise,
and utilities to extract parameter vectors that can be used as
inputs to a quantum circuit.

This file is designed to be dropped alongside the original
QuantumClassifierModel.py but offers a richer interface that
facilitates hybrid experiments.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable

import torch
import torch.nn as nn

from.FastBaseEstimator import FastEstimator

# ----------------------------------------------------------------------
# Classical classifier construction
# ----------------------------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a simple feed‑forward network that mimics the structure of
    the quantum classifier used in the seed.  The function returns
    (network, encoding, weight_sizes, observables) so that the signature
    matches the quantum implementation.
    """
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


# ----------------------------------------------------------------------
# Hybrid model definition
# ----------------------------------------------------------------------
class HybridClassifier(nn.Module):
    """
    Feed‑forward network that emulates the quantum ansatz interface.
    Users can call ``forward`` to obtain logits, or ``get_parameters``
    to extract a flattened vector that can be bound to a quantum
    circuit.
    """

    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_parameters(self) -> torch.Tensor:
        """Return a single 1‑D tensor that contains all learnable weights."""
        return torch.cat([p.flatten() for p in self.parameters()])


# ----------------------------------------------------------------------
# Estimator utilities
# ----------------------------------------------------------------------
class HybridEstimator(FastEstimator):
    """
    Wrapper around FastEstimator that accepts a HybridClassifier.
    It evaluates the model on batches of inputs and returns a list
    of observation values.  The optional ``shots`` argument injects
    Gaussian noise to emulate finite‑shot sampling.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: List[List[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        return super().evaluate(observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["build_classifier_circuit", "HybridClassifier", "HybridEstimator"]
