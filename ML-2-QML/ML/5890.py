"""Hybrid Graph Neural Network combining classical and quantum layers.

This module implements `GraphQNNHybrid`, a PyTorch `nn.Module` that
follows the architecture of the original GraphQNN seed but ends with
a Qiskit‑based single‑qubit estimator.  The last classical layer
produces a scalar rotation angle for an Ry gate; the expectation
value of a Pauli‑Y observable is returned as the network output.
The implementation is fully self‑contained and uses
`Statevector` from Qiskit to compute the expectation value
without external primitives.
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Iterable, Sequence, Tuple, List

import torch
from torch import nn
import networkx as nx
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector, Pauli

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Utility functions – adapted from the original GraphQNN module
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a classical linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(target: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (x, y) pairs where y = target @ x."""
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(target.size(1), dtype=torch.float32)
        y = target @ x
        data.append((x, y))
    return data

def random_network(arch: Sequence[int], samples: int):
    """Return an architecture, classical weights, training data and a random target."""
    weights: List[Tensor] = [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(arch), weights, training_data, target_weight

def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Compute forward activations for all samples."""
    activations: List[List[Tensor]] = []
    for x, _ in samples:
        layer_out = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            layer_out.append(current)
        activations.append(layer_out)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Fidelity of two pure state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# Hybrid estimator – combines classical NN with a single‑qubit Qiskit circuit
# --------------------------------------------------------------------------- #

class GraphQNNHybrid(nn.Module):
    """
    A lightweight hybrid network that mirrors the EstimatorQNN example.

    Architecture:
        * Classical fully‑connected layers defined by *arch*.
        * The last classical layer outputs a single value that is used as
          the rotation angle of a single‑qubit Ry gate.
        * A Pauli‑Y observable is measured; its expectation value is the
          network output.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(arch)
        layers: List[nn.Module] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.Tanh())
        self.classical = nn.Sequential(*layers)
        # Observable – Pauli Y
        self.observable = Pauli.from_label("Y")
        self.simulator = Aer.get_backend("statevector_simulator")

    def _quantum_circuit(self, angle: float) -> QuantumCircuit:
        """Build a parameterised single‑qubit circuit."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(angle, 0)
        return qc

    def forward(self, x: Tensor) -> Tensor:
        """Return the expectation value of Y after feeding *x* through the network."""
        # Classical part
        h = self.classical(x)
        # Use the last output as rotation angle
        angle = h.squeeze(-1).item()
        # Build quantum circuit
        qc = self._quantum_circuit(angle)
        # Evaluate expectation value
        state = Statevector.from_instruction(qc)
        exp_val = np.real(state.expectation_value(self.observable))
        return torch.tensor(exp_val, dtype=x.dtype, device=x.device)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
