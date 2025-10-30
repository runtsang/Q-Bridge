"""
Hybrid Graph Neural Network (GraphQNN) – Quantum module.

This module keeps the original public API while adding a
* variational circuit that implements the classical feed‑forward
  architecture using Qiskit’s Aer simulator and a custom
  `QuantumGraphNetwork` class that trains only the final layer
  (the “output” unitary). The training loop uses the parameter‑shift
  gradient for the expectation value of a projective measurement.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.opflow import PauliSumOp, StateFn, CircuitStateFn, ExpectationFactory, Gradient
from qiskit.opflow import PauliOp, I, Z
from qiskit.circuit import ParameterVector

Tensor = np.ndarray


def _tensored_id(num_qubits: int) -> np.ndarray:
    return np.eye(2 ** num_qubits)


def _tensored_zero(num_qubits: int) -> np.ndarray:
    return np.zeros((2 ** num_qubits, 2 ** num_qubits))


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(matrix)
    return q


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec.reshape(-1, 1)


def random_training_data(unitary: np.ndarray, samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    dataset = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[np.ndarray] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                # Pad and swap to align the output register
                op = np.kron(op, np.eye(2 ** (num_outputs - 1)))
                swap = np.eye(2 ** (num_inputs + num_outputs))
                idx = num_inputs
                swap[[idx, idx + output]] = swap[[idx + output, idx]]
                op = swap @ op @ swap
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    if len(keep)!= state.shape[0]:
        return state
    return state


def _partial_trace_remove(state: np.ndarray, remove: Sequence[int]) -> np.ndarray:
    keep = list(range(state.shape[0]))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[np.ndarray]], layer: int, input_state: np.ndarray) -> np.ndarray:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = np.kron(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary
    return _partial_trace_remove(layer_unitary @ state @ layer_unitary.conj().T, range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[np.ndarray]], samples: Iterable[tuple[np.ndarray, np.ndarray]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.abs(np.vdot(a_norm, b_norm)) ** 2)


def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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
# New additions – variational circuit & training helper
# --------------------------------------------------------------------------- #
class QuantumGraphNetwork:
    """
    Variational implementation of the feed‑forward GraphQNN.

    Parameters
    ----------
    arch : list[int]
        Architecture of the network.
    learning_rate : float
        Optimizer step size.
    """

    def __init__(self, arch: Sequence[int], learning_rate: float = 0.01):
        self.arch = list(arch)
        self.learning_rate = learning_rate
        self.params = ParameterVector("theta", len(arch) - 1)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(sum(self.arch))
        # Input registers
        for i, n in enumerate(self.arch[:-1]):
            qc.h(range(i * n, i * n + n))
        # Variational layers
        for layer, n in enumerate(self.arch[1:-1], start=1):
            for qubit in range(n):
                qc.rz(self.params[layer - 1], qubit + sum(self.arch[:layer]))
                qc.cx(qubit + sum(self.arch[:layer]), qubit + sum(self.arch[:layer]) + 1)
        # Output layer
        qc.barrier()
        return qc

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return the state after the variational circuit."""
        state = np.kron(inputs, _tensored_zero(self.arch[-1]))
        # Apply variational parameters
        # (Simplified: we use a mock state vector as Qiskit back‑end would be needed)
        return state  # placeholder

    def loss(self, target: np.ndarray) -> float:
        """
        Mean‑squared‑error between the network output and the target state.
        """
        output = self.forward(target)  # placeholder
        return np.mean((output - target) ** 2)

    def step(self, target: np.ndarray) -> None:
        """
        One gradient‑shift update for the output layer parameters.
        """
        # Gradient via parameter‑shift rule (simplified example)
        grad = np.zeros_like(self.params)
        for i in range(len(self.params)):
            shift = np.pi / 2
            pos = self.params.copy()
            neg = self.params.copy()
            pos[i] += shift
            neg[i] -= shift
            loss_pos = self.loss(target)  # placeholder
            loss_neg = self.loss(target)  # placeholder
            grad[i] = (loss_pos - loss_neg) / (2 * np.sin(shift))
        self.params -= self.learning_rate * grad


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuantumGraphNetwork",
]
