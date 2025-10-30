"""Hybrid Graph Neural Network – Quantum backend.

This module builds a variational circuit that mirrors the classical
GraphQNN architecture.  Each layer is a parameterized unitary acting on
the current register and the target qubits.  Training data consists of
input statevectors and their images under the target unitary.
Feedforward runs the circuit on a PennyLane device.  Fidelity
adjacency is computed with the statevector overlap.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

__all__ = [
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "GraphQNN",
]

# --------------------------------------------------------------------------- #
# 1. Helper functions
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random Haar‑distributed unitary matrix."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate training pairs (state, target_state)."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state = state / np.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[Tuple[int, int, np.ndarray]]], List[Tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
    """Construct a layered variational circuit.

    Parameters
    ----------
    qnn_arch : List[int]
        Layer sizes, e.g. ``[2, 3, 1]``.
    samples : int
        Number of training samples.

    Returns
    -------
    arch : List[int]
        Architecture list.
    circuits : List[List[Tuple[int, int, np.ndarray]]]
        For each layer a list of tuples ``(control, target, unitary)``.
    training_data : List[Tuple[np.ndarray, np.ndarray]]
        (input statevector, target statevector).
    target_unitary : np.ndarray
        The unitary that defines the ground‑truth mapping.
    total_wires : int
        Total number of qubits required by the circuit.
    """
    arch = list(qnn_arch)
    target_unitary = _random_qubit_unitary(arch[-1])
    training_data = random_training_data(target_unitary, samples)

    circuits: List[List[Tuple[int, int, np.ndarray]]] = [[]]
    for layer_idx in range(1, len(arch)):
        layer_ops: List[Tuple[int, int, np.ndarray]] = []
        for out_q in range(arch[layer_idx]):
            ctrl = layer_idx - 1  # control is previous layer
            tgt = out_q
            gate = _random_qubit_unitary(2)  # 2‑qubit unitary
            layer_ops.append((ctrl, tgt, gate))
        circuits.append(layer_ops)

    # Compute total wires needed
    total_wires = 0
    for layer in circuits:
        for ctrl, tgt, _ in layer:
            total_wires = max(total_wires, ctrl, tgt) + 1

    return arch, circuits, training_data, target_unitary, total_wires


# --------------------------------------------------------------------------- #
# 2. Variational circuit construction
# --------------------------------------------------------------------------- #
def _create_qnode(arch: Sequence[int], circuits: Sequence[List[Tuple[int, int, np.ndarray]]], dev: qml.Device):
    @qml.qnode(dev, interface="autograd")
    def circuit(state):
        qml.StatePrep(state, wires=range(len(state)))
        outputs = []
        for layer in circuits:
            for ctrl, tgt, gate in layer:
                qml.QubitUnitary(gate, wires=[ctrl, tgt])
            outputs.append(qml.state())
        return outputs
    return circuit


def feedforward(
    qnn_arch: Sequence[int],
    circuits: Sequence[List[Tuple[int, int, np.ndarray]]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    device_name: str = "default.qubit",
    total_wires: int | None = None,
) -> List[List[np.ndarray]]:
    """Execute the variational circuit on a PennyLane device.

    Returns a list of statevectors after each layer for every sample.
    """
    wires = total_wires if total_wires is not None else len(qnn_arch[-1])
    dev = qml.device(device_name, wires=wires)
    circuit = _create_qnode(qnn_arch, circuits, dev)
    stored: List[List[np.ndarray]] = []
    for inp, _ in samples:
        outputs = circuit(inp)
        stored.append(outputs)
    return stored


# --------------------------------------------------------------------------- #
# 3. Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap of two statevectors."""
    return float(abs(np.vdot(a, b)) ** 2)


def fidelity_adjacency(
    states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    """Create a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 4. GraphQNN wrapper class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Convenient wrapper around the quantum utilities."""

    def __init__(self, qnn_arch: Sequence[int], samples: int, device_name: str = "default.qubit"):
        self.arch, self.circuits, self.training_data, self.target_unitary, self.total_wires = random_network(
            qnn_arch, samples
        )
        self.device_name = device_name

    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        return feedforward(
            self.arch, self.circuits, samples, device_name=self.device_name, total_wires=self.total_wires
        )

    def fidelity_adjacency(self, states: Sequence[np.ndarray], threshold: float, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
