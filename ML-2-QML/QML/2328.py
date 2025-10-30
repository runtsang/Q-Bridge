"""Quantum‑centric ConvGen module.

This module implements a pure quantum convolution filter and graph utilities
that rely on Qiskit state‑fidelity calculations.  It is intended for
experiments that require a strictly quantum pipeline.

The public API mirrors the classical module but all operations are executed
on a quantum backend (Aer simulator).  The filter uses a variational circuit
parameterized by the input kernel, and the graph utilities construct an
adjacency graph from the fidelity of quantum states encoded from node
features.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
# 1. Quantum convolution filter
# --------------------------------------------------------------------------- #

class ConvGen:
    """Quantum convolution filter using a variational circuit.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the 2‑D kernel.
    threshold : float, default 0.0
        Threshold used to encode the input into rotation angles.
    shots : int, default 100
        Number of shots for the quantum backend.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 100):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size ** 2
        self.circuit = self._build_circuit(self.n_qubits)
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, n_qubits: int) -> qiskit.QuantumCircuit:
        circ = qiskit.QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta_{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            circ.rx(theta[i], i)
        for i in range(0, n_qubits - 1, 2):
            circ.cx(i, i + 1)
        circ.measure_all()
        return circ

    def run(self, data: np.ndarray) -> float:
        """Evaluate the quantum filter on a 2‑D kernel.

        Parameters
        ----------
        data : array‑like of shape (kernel_size, kernel_size)

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {f"theta_{i}": np.pi if val > self.threshold else 0
                    for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        total = 0
        for bitstring, count in result.items():
            ones = sum(int(b) for b in bitstring)
            total += ones * count
        return total / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# 2. Quantum graph utilities
# --------------------------------------------------------------------------- #

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

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

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

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return state.ptrace(list(keep))

def _layer_channel(qnn_arch: Sequence[int],
                   unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int,
                   input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
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

# --------------------------------------------------------------------------- #
# 3. Quantum graph‑convolutional layer
# --------------------------------------------------------------------------- #

class GraphConvQuantum:
    """Graph‑convolutional layer that operates on quantum states.

    It builds an adjacency graph from state fidelities and applies a
    linear transformation to the aggregated state.
    """

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.8):
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.linear = qt.Qobj(np.random.randn(out_features, in_features))

    def forward(self, node_states: List[qt.Qobj]) -> List[qt.Qobj]:
        graph = fidelity_adjacency(node_states, self.threshold)
        adj = nx.to_numpy_array(graph, nodelist=range(len(node_states)))
        deg = np.diag(1 / (adj.sum(axis=1) + 1e-6))
        agg = [self.linear @ qt.tensor(state, qt.qeye(self.out_features)) @ self.linear.dag()
               for state in node_states]
        aggregated = []
        for i, state in enumerate(agg):
            weighted = sum(adj[i, j] * state for j, state in enumerate(agg))
            aggregated.append(weighted)
        return aggregated

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
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

__all__ = [
    "ConvGen",
    "GraphConvQuantum",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
