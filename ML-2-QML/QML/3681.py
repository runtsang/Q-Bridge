"""Quantum implementation of QCNNGraphHybrid.

This module mirrors the classical module but uses Qiskit to build a
variational circuit.  It provides the same public API: a class
`QCNNGraphHybrid` that holds an `EstimatorQNN`, and utility
functions for random network generation, feedforward, and
fidelity‑based adjacency.  The class can optionally seed its
parameters from a graph of fidelities.
"""

from __future__ import annotations

import itertools
import numpy as np
from collections.abc import Iterable
from typing import List, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorQiskit
from qiskit_machine_learning.neural_networks import EstimatorQNN

Tensor = qt.Qobj


# --------------------------------------------------------------------------- #
#  Quantum utilities
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    return qt.tensor([qt.qeye(2) for _ in range(num_qubits)])


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    return qt.tensor([qt.fock(2, 0) for _ in range(num_qubits)])


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    return qt.Qobj(unitary, dims=[[2] * num_qubits, [2] * num_qubits])


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    return qt.Qobj(vec, dims=[[2] * num_qubits, [1] * num_qubits])


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.dims[0][0])
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """Generate a random quantum network and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_f = qnn_arch[layer - 1]
        out_f = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(out_f):
            op = _random_qubit_unitary(in_f + 1)
            if out_f > 1:
                op = qt.tensor(_random_qubit_unitary(in_f + 1), _tensored_id(out_f - 1))
                op = _swap_registers(op, in_f, in_f + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: List[int]) -> qt.Qobj:
    return state.ptrace(keep)


def _partial_trace_remove(state: qt.Qobj, remove: List[int]) -> qt.Qobj:
    keep = [i for i in range(len(state.dims[0])) if i not in remove]
    return state.ptrace(keep)


def _layer_channel(qnn_arch: List[int], unitaries: List[List[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), list(range(num_inputs)))


def feedforward(qnn_arch: List[int], unitaries: List[List[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    all_states = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            states.append(current)
        all_states.append(states)
    return all_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: List[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, ai in enumerate(states):
        for j, aj in enumerate(states[i + 1 :], start=i + 1):
            fid = state_fidelity(ai, aj)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  QCNN quantum circuit
# --------------------------------------------------------------------------- #

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Two‑qubit convolution unitary applied to each adjacent pair."""
    qc = QuantumCircuit(num_qubits)
    pairs = (num_qubits + 1) // 2
    params = ParameterVector(param_prefix, length=pairs * 3)
    idx = 0
    for q in range(0, num_qubits - 1, 2):
        qc.append(_conv_circuit(params[idx: idx + 3]), [q, q + 1])
        qc.barrier()
        idx += 3
    if num_qubits % 2 == 1:
        qc.append(_conv_circuit(params[idx: idx + 3]), [num_qubits - 1, 0])
        qc.barrier()
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """Pool unitary applied to each source‑sink pair."""
    qc = QuantumCircuit(len(sources) + len(sinks))
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for i, (src, snk) in enumerate(zip(sources, sinks)):
        qc.append(_pool_circuit(params[i * 3: i * 3 + 3]), [src, snk])
        qc.barrier()
    return qc


def build_qcnn_circuit() -> QuantumCircuit:
    """Construct the full QCNN circuit used in the original QML
    implementation but with a 8‑qubit feature map followed by
    three convolution–pooling blocks."""
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)
    ansatz.append(conv_layer(8, "c1"), range(8))
    ansatz.append(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8))
    ansatz.append(conv_layer(4, "c2"), range(4, 8))
    ansatz.append(pool_layer([0, 1], [2, 3], "p2"), range(4, 8))
    ansatz.append(conv_layer(2, "c3"), range(6, 8))
    ansatz.append(pool_layer([0], [1], "p3"), range(6, 8))
    circuit = QuantumCircuit(8)
    circuit.append(feature_map, range(8))
    circuit.append(ansatz, range(8))
    return circuit


# --------------------------------------------------------------------------- #
#  QCNNGraphHybrid quantum wrapper
# --------------------------------------------------------------------------- #

class QCNNGraphHybrid:
    """Quantum wrapper exposing the same API as the classical
    :class:`~ml_module.QCNNGraphHybrid`.  It builds a
    :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN` from the
    QCNN circuit and optionally seeds the variational parameters from
    a fidelity graph.
    """

    def __init__(self, init_from_graph: nx.Graph | None = None, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.circuit = build_qcnn_circuit()
        self.estimator = EstimatorQiskit()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" + "I" * 7, 1)]),
            input_params=ZFeatureMap(8).parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        if init_from_graph is not None:
            self._initialize_from_graph(init_from_graph)

    def _initialize_from_graph(self, graph: nx.Graph) -> None:
        """Map adjacency weights to initial parameter values."""
        param_values = []
        for idx, param in enumerate(self.circuit.parameters):
            if idx < len(list(graph.edges(data=True))):
                _, _, data = list(graph.edges(data=True))[idx]
                weight = data.get("weight", 0.0)
                if weight >= 1.0:
                    val = np.pi / 2
                elif weight >= 0.5:
                    val = np.pi / 4
                else:
                    val = np.random.uniform(0, 2 * np.pi)
            else:
                val = np.random.uniform(0, 2 * np.pi)
            param_values.append(val)
        for param, val in zip(self.circuit.parameters, param_values):
            param.assign_value(val)

    def forward(self, inputs: List[float]) -> List[float]:
        """Evaluate the EstimatorQNN on a single input vector."""
        return self.qnn.predict(inputs).tolist()

    @staticmethod
    def random_network_from_graph(
        graph: nx.Graph,
        samples: int = 100,
        seed: int | None = None,
    ) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """Return the same tuple as :func:`random_network`; the graph is
        ignored for the random generation but kept for API symmetry."""
        return random_network([8, 16, 16, 12, 8, 4, 4, 1], samples)

    @staticmethod
    def feedforward(
        qnn_arch: List[int],
        unitaries: List[List[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        """Return layer‑wise states for each sample."""
        return feedforward(qnn_arch, unitaries, samples)

    @staticmethod
    def fidelity_adjacency(
        states: List[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "QCNNGraphHybrid",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
]
