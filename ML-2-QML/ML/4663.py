"""Hybrid classical‑quantum classifier factory.

This module defines a `HybridClassifierModel` that can operate in
classical or quantum mode.  The classical mode is a simple feed‑forward
network that mirrors the original seed.  The quantum mode is a
placeholder that returns zeros, but the class structure is
designed so that the quantum implementation can be swapped in
without touching the training loop.

The function `build_classifier_circuit` returns a tuple that matches
the original API: (model, encoding, weight_sizes, observables).

A small synthetic dataset generator is included for quick prototyping.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import torch
import torch.nn as nn
import numpy as np


def _check_features(num_features: int, depth: int) -> None:
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")


def generate_superposition_data(
    num_features: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data where labels are a noisy
    sinusoid of the sum of inputs.  The same scheme is used in the
    regression seed but returned as a binary label for classification."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = (np.sin(angles) > 0).astype(np.float32)
    return x, y


class HybridClassifierModel(nn.Module):
    """Hybrid classifier that can delegate to a classical feed‑forward
    network or a quantum circuit (stub)."""

    def __init__(self, num_features: int, depth: int, mode: str = "classical"):
        super().__init__()
        _check_features(num_features, depth)
        self.num_features = num_features
        self.depth = depth
        self.mode = mode.lower()

        if self.mode == "classical":
            layers = []
            for _ in range(depth):
                layers.append(nn.Linear(num_features, num_features))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(num_features, 2))
            self.model = nn.Sequential(*layers)

            self.weight_sizes = [
                layer.weight.numel() + layer.bias.numel()
                for layer in self.model[:-1]
            ]
            self.weight_sizes.append(
                self.model[-1].weight.numel() + self.model[-1].bias.numel()
            )
        else:
            # Quantum branch – a thin wrapper around the Qiskit circuit
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector
            from qiskit.quantum_info import SparsePauliOp

            encoding = ParameterVector("x", num_features)
            weights = ParameterVector("theta", num_features * depth)

            circuit = QuantumCircuit(num_features)
            for param, qubit in zip(encoding, range(num_features)):
                circuit.rx(param, qubit)

            index = 0
            for _ in range(depth):
                for qubit in range(num_features):
                    circuit.ry(weights[index], qubit)
                    index += 1
                for qubit in range(num_features - 1):
                    circuit.cz(qubit, qubit + 1)

            observables = [
                SparsePauliOp(f"I" * i + "Z" + "I" * (num_features - i - 1))
                for i in range(num_features)
            ]

            self.model = nn.Module()
            self.model.circuit = circuit
            self.model.encoding = list(encoding)
            self.model.weights = list(weights)
            self.model.observables = observables

            # weight_sizes: number of parameters in the quantum circuit
            self.weight_sizes = [len(weights)]

        # encoding indices for API parity
        self.encoding = list(range(num_features))
        # observables: placeholder for classical branch
        self.observables = self.model.observables if self.mode!= "classical" else [0, 1]

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.mode == "classical":
            return self.model(x)
        else:
            # quantum forward is a stub – replace with a real simulator
            batch = x.shape[0]
            return torch.zeros(batch, 2, dtype=x.dtype, device=x.device)


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[HybridClassifierModel, Iterable[int], Iterable[int], List[int]]:
    """Provide a classical hybrid classifier matching the original API."""
    model = HybridClassifierModel(num_features, depth, mode="classical")
    encoding = model.encoding
    weight_sizes = list(model.weight_sizes)
    observables = list(model.observables)
    return model, encoding, weight_sizes, observables
