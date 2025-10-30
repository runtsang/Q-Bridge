"""Quantum Convolutional Neural Network (QCNN) with graph utilities for measurement results.

The module builds the QCNN variational circuit, supplies a measurement routine, and provides graph utilities that operate on measurement outcomes.  The graph functions are inspired by the classical GraphQNN utilities and allow clustering of measured states.

Typical usage::

    from hybrid_qcnn_graph import QCNN, measure_circuit, build_fidelity_graph
    qnn = QCNN()
    circuit = qnn.circuit
    state_vectors = measure_circuit(circuit, shots=1024)
    graph = build_fidelity_graph(state_vectors, threshold=0.9)
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import scipy as sc
import qutip as qt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

__all__ = [
    "QCNN",
    "measure_circuit",
    "build_fidelity_graph",
    "state_fidelity_q",
    "fidelity_adjacency_q",
    "random_network_q",
    "random_training_data_q",
    "feedforward_q",
]


# --------------------------------------------------------------------------- #
#   QCNN quantum circuit
# --------------------------------------------------------------------------- #
def QCNN() -> EstimatorQNN:
    """Return a QCNN EstimatorQNN object."""
    estimator = Estimator()

    # Convolutional two‑qubit unitary
    def conv_circuit(params):
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

    # Convolutional layer
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            start = i * 3
            qc.append(conv_circuit(params[start : start + 3]), [i, i + 1])
            qc.barrier()
        return qc

    # Pooling two‑qubit unitary
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Pooling layer
    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for src, snk in zip(sources, sinks):
            start = sources.index(src) * 3
            qc.append(pool_circuit(params[start : start + 3]), [src, snk])
            qc.barrier()
        return qc

    # Build the full ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Feature map
    feature_map = ZFeatureMap(8)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


# --------------------------------------------------------------------------- #
#   Measurement routine
# --------------------------------------------------------------------------- #
def measure_circuit(circuit: QuantumCircuit, shots: int = 1024) -> List[qt.Qobj]:
    """Return a list of state vectors measured from the circuit."""
    backend = Estimator()
    results = backend.run(circuit, shots=shots).result()
    state_vectors: List[qt.Qobj] = []
    for res in results:
        # Convert probability distribution to a state vector (approximate)
        probs = np.array(res)
        amplitudes = np.sqrt(probs)
        state = qt.Qobj(amplitudes, dims=[[2] * int(np.log2(len(amplitudes)))] * 2)
        state_vectors.append(state)
    return state_vectors


# --------------------------------------------------------------------------- #
#   Graph utilities for quantum outputs
# --------------------------------------------------------------------------- #
def state_fidelity_q(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between two pure states."""
    return float(abs((a.dag() * b)[0, 0]) ** 2)


def fidelity_adjacency_q(
    states: Iterable[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity_q(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def build_fidelity_graph(
    state_vectors: List[qt.Qobj],
    threshold: float = 0.9,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Convenience wrapper that builds a graph from a list of state vectors."""
    return fidelity_adjacency_q(
        state_vectors, threshold, secondary=secondary, secondary_weight=secondary_weight
    )


# --------------------------------------------------------------------------- #
#   Random data / network generation (quantum version)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data_q(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate input–target pairs for a target unitary."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network_q(qnn_arch: List[int], samples: int):
    """Generate a random unitary chain and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data_q(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: List[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: List[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: List[int],
    unitaries: List[List[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward_q(
    qnn_arch: List[int],
    unitaries: List[List[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Compute the state after each layer for every sample."""
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states
