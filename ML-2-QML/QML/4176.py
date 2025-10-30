"""Quantum‑enhanced graph neural network using Pennylane.

The module mirrors the classical `GraphQNNHybrid` API but replaces
classical linear updates with variational circuits.  It also
provides a quantum‑LSTM cell built from Pennylane QNodes and
the same fraud‑detection style parameterised gates.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

Tensor = np.ndarray


# --------------------------------------------------------------------------- #
# 1. Fraud‑Detection style parameters (quantum mapping)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


# --------------------------------------------------------------------------- #
# 2. Random quantum utilities
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> qml.QubitUnitary:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = qml.math.orthogonal(matrix)
    return qml.QubitUnitary(unitary, wires=range(num_qubits))


def _random_qubit_state(num_qubits: int) -> qml.QubitStateVector:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
    amplitudes /= np.linalg.norm(amplitudes)
    return qml.QubitStateVector(amplitudes, wires=range(num_qubits))


def random_training_data(unitary: qml.QubitUnitary, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    num_qubits = len(unitary.wires)
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qml.QubitUnitary]], List[Tuple[Tensor, Tensor]], qml.QubitUnitary]:
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qml.QubitUnitary]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qml.QubitUnitary] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                # entangle with remaining outputs
                op = qml.tensor(_random_qubit_unitary(num_inputs + 1), qml.identity(num_outputs - 1))
                # swap wires to place new qubit at the end
                op = qml.operation.swap(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


# --------------------------------------------------------------------------- #
# 3. Quantum feedforward
# --------------------------------------------------------------------------- #
def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qml.QubitUnitary]],
                   layer: int, input_state: qml.QubitStateVector) -> qml.QubitStateVector:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # initialise ancilla qubits in |0>
    ancilla = qml.QubitStateVector(np.zeros(2 ** (num_outputs), dtype=complex), wires=range(num_outputs))
    state = qml.tensor(input_state, ancilla)

    # apply unitary
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = layer_unitary @ gate
    state = layer_unitary @ state

    # partial trace over inputs
    keep = list(range(num_inputs, num_inputs + num_outputs))
    return qml.partial_trace(state, keep)


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qml.QubitUnitary]],
                samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[qml.QubitStateVector]]:
    stored_states: List[List[qml.QubitStateVector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
# 4. Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: qml.QubitStateVector, b: qml.QubitStateVector) -> float:
    """Absolute squared overlap between pure states."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qml.QubitStateVector], threshold: float,
                       *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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
# 5. Quantum LSTM cell (variational)
# --------------------------------------------------------------------------- #
class QuantumLSTMCell:
    """Variational LSTM cell realised with Pennylane QNodes."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)

        # Parameters for each gate (one per qubit)
        self.forget_params = np.random.randn(n_qubits)
        self.input_params = np.random.randn(n_qubits)
        self.update_params = np.random.randn(n_qubits)
        self.output_params = np.random.randn(n_qubits)

    def _gate(self, params: np.ndarray) -> qml.QNode:
        @qml.qnode(self.device, interface="autograd")
        def gate_fn(x):
            # encode classical input into rotations
            for i in range(self.input_dim):
                qml.RX(x[i], wires=i)
            # apply parameterised rotations as gate
            for w, p in enumerate(params):
                qml.RZ(p, wires=w)
            return qml.probs(wires=range(self.n_qubits))
        return gate_fn

    def __call__(self, x: Tensor) -> Tensor:
        f = self._gate(self.forget_params)(x)
        i = self._gate(self.input_params)(x)
        g = self._gate(self.update_params)(x)
        o = self._gate(self.output_params)(x)
        # Simple classical combination to emulate LSTM behaviour
        return f * i * g * o  # placeholder for actual state update


# --------------------------------------------------------------------------- #
# 6. Quantum‑enhanced GraphQNNHybrid
# --------------------------------------------------------------------------- #
class GraphQNNHybrid:
    """
    Quantum‑enhanced graph neural network that retains the same public API as the classical
    version but uses Pennylane circuits for node‑state propagation.  An optional quantum‑LSTM
    cell can be attached to the final node embedding to capture sequential dependencies.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        use_lstm: bool = False,
        n_qubits: int = 0,
        lstm_hidden_dim: int | None = None,
    ) -> None:
        self.qnn_arch = list(qnn_arch)
        self.use_lstm = use_lstm
        if use_lstm:
            if n_qubits <= 0:
                raise ValueError("n_qubits must be positive when use_lstm=True.")
            self.lstm = QuantumLSTMCell(
                input_dim=qnn_arch[-1],
                hidden_dim=lstm_hidden_dim or qnn_arch[-1],
                n_qubits=n_qubits,
            )
        else:
            self.lstm = None

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[qml.QubitStateVector]]:
        stored: List[List[qml.QubitStateVector]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(self.qnn_arch)):
                current_state = _layer_channel(self.qnn_arch, self.unitaries, layer, current_state)
                layerwise.append(current_state)
            if self.use_lstm:
                lstm_out = self.lstm(current_state)
                layerwise.append(lstm_out)
            stored.append(layerwise)
        return stored

    # The class can expose the same static helpers as the classical module
    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: qml.QubitUnitary, samples: int):
        return random_training_data(unitary, samples)

    @staticmethod
    def fidelity_adjacency(states, threshold, *, secondary=None, secondary_weight=0.5):
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "FraudLayerParameters",
    "QuantumLSTMCell",
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
