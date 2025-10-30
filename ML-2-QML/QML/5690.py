"""Combined quantum graph neural network with QCNN‑style layers.

This module merges the GraphQNN quantum utilities with the QCNN circuit design,
providing a quantum variational circuit that mirrors the same convolution‑pooling
pattern and propagates states over a graph‑defined adjacency.  The class
is compatible with the classical counterpart and can be used in hybrid
training pipelines.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import qiskit as qk
import qiskit.quantum_info as qi
import scipy as sc
import torch

from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.machine_learning.optimizers import COBYLA
from qiskit.machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.machine_learning.neural_networks import EstimatorQNN

# --------------------------------------------------------------------------- #
# Quantum utilities – largely from the original GraphQNN QML seed
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qi.Qobj:
    return qi.Qobj(np.eye(2**num_qubits))

def _tensored_zero(num_qubits: int) -> qi.Qobj:
    proj = np.diag([1] + [0]*(2**num_qubits-1))
    return qi.Qobj(proj)

def _swap_registers(op: qi.Qobj, source: int, target: int) -> qi.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qi.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    return qi.Qobj(unitary)

def _random_qubit_state(num_qubits: int) -> qi.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    return qi.Qobj(amplitudes)

def random_training_data(unitary: qi.Qobj, samples: int) -> List[Tuple[qi.Qobj, qi.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qi.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qi.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qi.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qi.Qobj, keep: Sequence[int]) -> qi.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qi.Qobj, remove: Sequence[int]) -> qi.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qi.Qobj]], layer: int, input_state: qi.Qobj) -> qi.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qi.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qi.Qobj]], samples: Iterable[Tuple[qi.Qobj, qi.Qobj]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qi.Qobj, b: qi.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qi.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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
# QCNN‑style quantum circuit helpers
# --------------------------------------------------------------------------- #

def conv_circuit(params: ParameterVector) -> qk.QuantumCircuit:
    qc = qk.QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def pool_circuit(params: ParameterVector) -> qk.QuantumCircuit:
    qc = qk.QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> qk.QuantumCircuit:
    qc = qk.QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2], inplace=True)
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2], inplace=True)
        param_index += 3
    return qc

def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> qk.QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = qk.QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink], inplace=True)
        param_index += 3
    return qc

# --------------------------------------------------------------------------- #
# Hybrid Graph‑QCNN quantum model
# --------------------------------------------------------------------------- #

class HybridGraphQNN:
    """
    Quantum variational circuit that applies QCNN‑style convolution and pooling
    over a graph‑defined adjacency.  The circuit is built with Qiskit and
    evaluated with a StatevectorEstimator.  It mirrors the classical
    HybridGraphQNN but operates on quantum states.
    """

    def __init__(self, qnn_arch: Sequence[int], num_qubits: int = 8):
        self.qnn_arch = list(qnn_arch)
        self.num_qubits = num_qubits

        # Feature map and ansatz construction
        self.feature_map = ZFeatureMap(num_qubits)
        ansatz = qk.QuantumCircuit(num_qubits)

        # First convolution
        ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
        # First pooling
        ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), inplace=True)
        # Second convolution
        ansatz.compose(conv_layer(num_qubits // 2, "c2"), inplace=True)
        # Second pooling
        ansatz.compose(pool_layer(list(range(num_qubits // 4)), list(range(num_qubits // 4, num_qubits // 2)), "p2"), inplace=True)
        # Third convolution
        ansatz.compose(conv_layer(num_qubits // 4, "c3"), inplace=True)
        # Third pooling
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        observable = qi.SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        algorithm_globals.random_seed = 12345
        estimator = Estimator()

        self.qnn = EstimatorQNN(
            circuit=ansatz,
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum circuit on a batch of classical inputs.
        """
        return self.qnn.predict(inputs)

# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #

__all__ = [
    "HybridGraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
