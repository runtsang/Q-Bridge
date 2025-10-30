"""Quantum Graph Neural Network utilities with a Pennylane variational circuit.

This module keeps the original fidelity‑based adjacency logic while
introducing a `GraphQNN` class that builds a simple variational
quantum circuit.  The circuit can be executed on a simulator or a
real device via Pennylane's device interface.  It mirrors the original
API for compatibility.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch

Tensor = torch.Tensor
QDArray = np.ndarray

# --------------------------------------------------------------------------- #
#  Helper functions (unchanged from the seed)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qml.QubitOperator:
    """Identity operator on `num_qubits` qubits."""
    return qml.Identity(num_qubits)

def _tensored_zero(num_qubits: int) -> qml.QubitOperator:
    """Zero projector on `num_qubits` qubits."""
    return qml.Projector([0] * num_qubits, num_qubits)

def _swap_registers(op: qml.QubitOperator, source: int, target: int) -> qml.QubitOperator:
    if source == target:
        return op
    # PennyLane does not expose a direct permute, so we build a SWAP circuit
    return op @ qml.SWAP([source, target])

def _random_qubit_unitary(num_qubits: int) -> qml.QubitOperator:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.orth(matrix)
    return qml.QubitOperator(unitary, num_qubits)

def _random_qubit_state(num_qubits: int) -> qml.QubitOperator:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    return qml.QubitOperator(amplitudes, num_qubits)

def random_training_data(unitary: qml.QubitOperator, samples: int) -> list[tuple[qml.QubitOperator, qml.QubitOperator]]:
    dataset = []
    num_qubits = unitary.shape[0]
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    """Return architecture, list of unitaries per layer, training data and target unitary."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qml.QubitOperator]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qml.QubitOperator] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qml.Tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qml.QubitOperator, keep: Sequence[int]) -> qml.QubitOperator:
    if len(keep)!= len(state.shape):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qml.QubitOperator, remove: Sequence[int]) -> qml.QubitOperator:
    keep = list(range(len(state.shape)))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.QubitOperator]],
    layer: int,
    input_state: qml.QubitOperator,
) -> qml.QubitOperator:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qml.Tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary

    return _partial_trace_remove(layer_unitary @ state @ layer_unitary.dag(), range(num_inputs))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.QubitOperator]],
    samples: Iterable[tuple[qml.QubitOperator, qml.QubitOperator]],
) -> list[list[qml.QubitOperator]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qml.QubitOperator, b: qml.QubitOperator) -> float:
    """Return the absolute squared overlap between pure states `a` and `b`."""
    return abs((a.dag() @ b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qml.QubitOperator],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
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
#  Hybrid Quantum GraphQNN class
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Quantum graph neural network with a Pennylane variational circuit.

    The class exposes a `forward` method that returns the quantum state
    vector for a given classical embedding.  It can be executed on a
    simulator or a real device via Pennylane's device interface.
    """

    def __init__(self, arch: Sequence[int], quantum_device: str = "default.qubit"):
        self.arch = list(arch)
        self.quantum_device = quantum_device
        self.n_qubits = arch[-1]
        self.dev = qml.device(quantum_device, wires=self.n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: Tensor) -> QDArray:
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    def forward(self, x: Tensor) -> QDArray:
        """Return the quantum state vector for the input embedding."""
        return self.qnode(x)

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
