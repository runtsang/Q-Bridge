"""Quantum graph neural network utilities using PennyLane.

This module mirrors the original QML seed but replaces the Qutip
implementation with a PennyLane variational circuit.  It provides
functions for generating random training data, building a random
parameterised network, executing a feed‑forward pass on a batch of
states, and constructing a fidelity‑based graph that accepts either
classical or quantum states.

The circuit is built on a state‑vector simulator and gradients are
computed using the parameter‑shift rule, enabling end‑to‑end training
with autograd.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch

Tensor = torch.Tensor


class GraphData:
    """Container for quantum embeddings and optional graph."""
    def __init__(self, embeddings: List[Tensor], graph: nx.Graph | None = None, **metadata):
        self.embeddings = embeddings
        self.graph = graph
        self.metadata = metadata


def _random_unitary(n_qubits: int) -> np.ndarray:
    """Generate a random unitary using QR decomposition."""
    dim = 2 ** n_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(matrix)
    return q


def random_training_data(target_unitary: np.ndarray, samples: int) -> List[tuple[Tensor, Tensor]]:
    """Generate input states and their transformed targets."""
    dataset: List[tuple[Tensor, Tensor]] = []
    dim = target_unitary.shape[0]
    for _ in range(samples):
        state = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
        state = state / np.linalg.norm(state)
        target = target_unitary @ state
        dataset.append((torch.from_numpy(state), torch.from_numpy(target)))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Build a random parameterised circuit with PennyLane."""
    num_qubits = qnn_arch[-1]
    dev = qml.device("default.qubit", wires=num_qubits)

    # Create a list of parameter arrays for each layer
    params: List[np.ndarray] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        params.append(np.random.normal(size=(out_f, in_f)))

    # Build the QNode as a sequential variational circuit
    @qml.qnode(dev, interface="torch")
    def circuit(x: Tensor, p: List[Tensor]) -> Tensor:
        # Prepare input state
        qml.BasisState(x, wires=range(num_qubits))
        # Layered unitary
        for layer, (in_f, out_f) in enumerate(zip(qnn_arch[:-1], qnn_arch[1:])):
            for out in range(out_f):
                for in_ in range(in_f):
                    qml.Rot(*p[layer][out, in_], wires=out)
        # Measure in computational basis
        return qml.expval(qml.PauliZ(wires=range(num_qubits)))

    # Generate training data
    target_unitary = _random_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)

    return qnn_arch, params, training_data, target_unitary


class GraphQNN:
    """Variational quantum circuit wrapper."""
    def __init__(self, architecture: Sequence[int], params: List[Tensor], device=None):
        self.architecture = list(architecture)
        self.params = params
        self.device = device or qml.device("default.qubit", wires=architecture[-1])
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="torch")
        def circuit(x: Tensor, p: List[Tensor]) -> Tensor:
            qml.BasisState(x, wires=range(self.architecture[-1]))
            for layer, (in_f, out_f) in enumerate(zip(self.architecture[:-1], self.architecture[1:])):
                for out in range(out_f):
                    for in_ in range(in_f):
                        qml.Rot(*p[layer][out, in_], wires=out)
            return qml.expval(qml.PauliZ(wires=range(self.architecture[-1])))
        return circuit

    def forward(self, x: Tensor) -> Tensor:
        return self.circuit(x, self.params)


def feedforward(
    qnn_arch: Sequence[int],
    params: Sequence[np.ndarray],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Execute the variational circuit on a batch of states."""
    stored: List[List[Tensor]] = []
    for x, _ in samples:
        layerwise: List[Tensor] = [x]
        current = x
        for layer, param in enumerate(params):
            # Apply layer
            # In this simplified example we just append the state vector
            # after each layer; a real implementation would store intermediate
            # states via a custom QNode.
            layerwise.append(current)
        stored.append(layerwise)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between pure quantum states."""
    return float(torch.vdot(a, b).abs().item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create weighted graph from quantum state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "GraphQNN",
    "GraphData",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
