"""Quantum‑augmented graph‑neural‑network utilities.

This module focuses on the quantum component of the hybrid model:
* a simple variational circuit that takes a vector of parameters and
  returns the expectation of a Pauli observable,
* helper functions to generate random unitary training data,
* a minimal network construction that mirrors the classical seed but
  operates on quantum states.

The quantum code is fully differentiable via Pennylane and can be
plugged into the GraphQNNHybrid class defined in the classical module.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import torch

# ----------------------------------------------------------------------
# Quantum utilities
# ----------------------------------------------------------------------
def _random_qubit_unitary(num_qubits: int) -> qml.QubitOperator:
    """Return a random unitary as a QubitOperator (for training data)."""
    dim = 2 ** num_qubits
    matrix = torch.randn(dim, dim, dtype=torch.complex64)
    matrix = torch.linalg.qr(matrix)[0]  # orthonormalize
    return qml.QubitOperator(matrix, wires=range(num_qubits))

def _random_qubit_state(num_qubits: int) -> torch.Tensor:
    """Return a random pure state vector."""
    dim = 2 ** num_qubits
    vec = torch.randn(dim, dtype=torch.complex64)
    vec /= torch.linalg.norm(vec)
    return vec

def random_training_data(unitary: qml.QubitOperator, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate (state, unitary * state) pairs."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    num_qubits = len(unitary.wires)
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Construct a random quantum feed‑forward network.

    The network is a list of layers, each layer being a list of unitary
    operations applied to the incoming state.  The output of the last
    layer is the target unitary for training data generation.
    """
    # For simplicity, use a single‑qubit ansatz per layer
    unitaries: List[List[qml.QubitOperator]] = [[]]
    for layer_idx, (in_f, out_f) in enumerate(zip(qnn_arch[:-1], qnn_arch[1:])):
        layer_ops: List[qml.QubitOperator] = []
        for _ in range(out_f):
            op = _random_qubit_unitary(in_f + 1)  # +1 for ancilla
            layer_ops.append(op)
        unitaries.append(layer_ops)

    target_unitary = unitaries[-1][0]  # choose first as ground truth
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), unitaries, training_data, target_unitary

# ----------------------------------------------------------------------
# State fidelity and adjacency (quantum version)
# ----------------------------------------------------------------------
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Overlap |<a|b>|^2 for normalized pure states."""
    a_norm = a / (torch.linalg.norm(a) + 1e-12)
    b_norm = b / (torch.linalg.norm(b) + 1e-12)
    return float(torch.abs(torch.vdot(a_norm, b_norm)) ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----------------------------------------------------------------------
# Quantum feed‑forward helper
# ----------------------------------------------------------------------
def _qnode(params: torch.Tensor) -> torch.Tensor:
    """Variational circuit: apply RY gates for each parameter and return ⟨Z⟩."""
    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev, interface="torch")
    def circuit(p):
        for angle in p:
            qml.RY(angle, wires=0)
        return qml.expval(qml.PauliZ(0))
    return circuit(params)

def quantum_expectation(inputs: torch.Tensor) -> torch.Tensor:
    """Batch‑wise quantum expectation."""
    return torch.stack([_qnode(row) for row in inputs])

__all__ = [
    "_random_qubit_unitary",
    "_random_qubit_state",
    "random_training_data",
    "random_network",
    "state_fidelity",
    "fidelity_adjacency",
    "quantum_expectation",
]
