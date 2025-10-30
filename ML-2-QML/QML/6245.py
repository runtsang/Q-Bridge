"""GraphQNNAdvanced – quantum‑to‑circuit mapping with a parameter‑shared variational circuit.

The implementation reproduces the original seed API while adding a
parameter‑shared variational ansatz that operates on a register of
``qnn_arch[-1]`` qubits.  The circuit is built using PennyLane and
simulated on the default qubit device.  The public methods mirror
the classical counterpart, making it straightforward to benchmark
classical vs. quantum fidelity.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import pennylane as qml
import networkx as nx

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, r = np.linalg.qr(matrix)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), target_unitary, training_data, target_unitary

def _layer_circuit(qnn_arch: Sequence[int], layer: int, wires: List[int]) -> qml.QNode:
    """Construct the variational layer for a given GNN layer."""
    def circuit(*params):
        idx = 0
        for w in wires:
            qml.RX(params[idx], wires=w); idx += 1
            qml.RY(params[idx], wires=w); idx += 1
            qml.RZ(params[idx], wires=w); idx += 1
        for i in range(len(wires)-1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
        return qml.state()
    return qml.QNode(circuit, qml.device("default.qubit", wires=len(wires)))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.QNode]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    stored_states: List[List[np.ndarray]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            circuit = unitaries[layer][0]
            num_params = 3 * qnn_arch[layer]
            params = np.random.randn(num_params)
            current = circuit(*params)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNAdvanced:
    """Quantum graph‑based neural network with a parameter‑shared
    variational ansatz.  The architecture is specified by a sequence
    of integers describing the number of qubits per layer.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        self.qnn_arch = list(qnn_arch)
        self.device = device
        self.shots = shots
        self.unitaries: List[List[qml.QNode]] = [[]]
        for layer in range(1, len(self.qnn_arch)):
            num_qubits = self.qnn_arch[layer]
            wires = list(range(num_qubits))
            circuit = _layer_circuit(self.qnn_arch, layer, wires)
            self.unitaries.append([circuit])

    def feedforward(self, input_state: np.ndarray) -> List[np.ndarray]:
        states = [input_state]
        current = input_state
        for layer in range(1, len(self.qnn_arch)):
            circuit = self.unitaries[layer][0]
            num_params = 3 * self.qnn_arch[layer]
            params = np.random.randn(num_params)
            current = circuit(*params)
            states.append(current)
        return states

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int):
        return random_training_data(unitary, samples)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qml.QNode]],
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    ):
        return feedforward(qnn_arch, unitaries, samples)

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "GraphQNNAdvanced",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
