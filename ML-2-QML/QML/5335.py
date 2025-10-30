"""GraphQNNGen224QML: Quantum‑classical hybrid graph neural network.

This module builds a graph‑based quantum neural network that
mirrors the classical GraphQNNGen224.  It incorporates:
* QCNN‑style convolution and pooling layers
* a self‑attention block implemented with Qiskit
* a fidelity‑based adjacency graph
* a lightweight estimator that runs on a Qiskit backend.

The design follows the structure of the original QML GraphQNN
while adding the QCNN and self‑attention circuits.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class SelfAttentionCircuit:
    """Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the block.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumCircuit(n_qubits, name="SelfAttention")
        self._build()

    def _build(self) -> None:
        # Random single‑qubit rotations for each qubit
        for q in range(self.n_qubits):
            self.qr.rx(np.random.uniform(0, 2 * np.pi), q)
            self.qr.ry(np.random.uniform(0, 2 * np.pi), q)
            self.qr.rz(np.random.uniform(0, 2 * np.pi), q)
        # Entangling layer
        for q in range(self.n_qubits - 1):
            self.qr.cx(q, q + 1)

    def circuit(self, params: np.ndarray | None = None) -> QuantumCircuit:
        return self.qr.copy()


def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="ConvLayer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = conv_circuit(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def pool_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="PoolLayer")
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = pool_circuit(params[idx : idx + 3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        idx += 3
    return qc


def _random_state(num_qubits: int) -> Statevector:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[QuantumCircuit]], List[Tuple[Statevector, Statevector]], Statevector]:
    """Generate a random QCNN‑style network and training data."""
    # Build identity circuits for each layer as placeholders
    unitaries: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[QuantumCircuit] = []
        for _ in range(num_outputs):
            # identity of size num_inputs+1 to match _layer_channel expectations
            circ = QuantumCircuit(num_inputs + 1)
            circ.name = "I"
            layer_ops.append(circ)
        unitaries.append(layer_ops)

    # Random target state for the last layer
    target_state = _random_state(qnn_arch[-1])

    # Training data: random input states and their target
    training_data: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        inp = _random_state(qnn_arch[0])
        training_data.append((inp, target_state))

    return list(qnn_arch), unitaries, training_data, target_state


def _partial_trace_keep(state: Statevector, keep: Sequence[int]) -> Statevector:
    return state.ptrace(keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[QuantumCircuit]], layer: int, input_state: Statevector) -> Statevector:
    num_inputs = qnn_arch[layer - 1]
    # Apply each unitary sequentially
    state = input_state
    for gate in unitaries[layer]:
        state = state.evolve(gate)
    # Trace out unused qubits
    keep = list(range(num_inputs))
    return state.ptrace(keep)


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[QuantumCircuit]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    stored: List[List[Statevector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


def state_fidelity(a: Statevector, b: Statevector) -> float:
    return abs((a.data.conj().T @ b.data)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
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


class FastBaseEstimatorQML:
    """Fast estimator that evaluates expectation values for a parametrized circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = [
    "SelfAttentionCircuit",
    "conv_circuit",
    "pool_circuit",
    "conv_layer",
    "pool_layer",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FastBaseEstimatorQML",
]
