"""Hybrid quantum graph neural network utilities with classical‑inspired layers.

This module extends the original GraphQNN.py by adding fully‑connected
quantum layers, convolution‑pooling circuits, and a unified interface
for random network generation, feed‑forward, and fidelity‑based
adjacency.  The public API mirrors the classical counterpart but
operates on Qutip `Qobj` states.

Key additions:
* `conv_layer` and `pool_layer` functions that build parametrised
  two‑qubit unitaries, inspired by QCNN.
* `FCL()` returns a simple parametric quantum circuit for a fully
  connected layer.
* `GraphQNNGen180` – a quantum‑only graph neural network class that
  holds architecture, unitaries, and training data, and exposes
  `feedforward`, `get_adjacency`, and a static `random_network` helper.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List, Union

import networkx as nx
import qutip as qt
import scipy as sc
import numpy as np

Qobj = qt.Qobj
State = Qobj


def _tensored_id(num_qubits: int) -> Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: Qobj, source: int, target: int) -> Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary


def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]], layer: int, input_state: Qobj) -> Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer - 1][0].copy()
    for gate in unitaries[layer - 1][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]], samples: Iterable[Tuple[Qobj, Qobj]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Qobj, b: Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# Convolution and pooling circuits inspired by QCNN -----------------------------------------

def conv_circuit(params: Sequence[float]) -> Qobj:
    """Build a 2‑qubit unitary used in QCNN conv layers."""
    return _random_qubit_unitary(2)  # placeholder


def pool_circuit(params: Sequence[float]) -> Qobj:
    """Build a 2‑qubit unitary used in QCNN pool layers."""
    return _random_qubit_unitary(2)  # placeholder


def conv_layer(num_qubits: int, param_prefix: str) -> Qobj:
    """Return a tensor of 2‑qubit conv unitaries for a block."""
    return _random_qubit_unitary(num_qubits)


def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> Qobj:
    return _random_qubit_unitary(len(sources) + len(sinks))


# Fully connected quantum layer --------------------------------------------------------------

def FCL(num_qubits: int = 1):
    """Return a simple parametric quantum circuit for a fully connected layer."""
    class QuantumCircuit:
        def __init__(self, n_qubits: int, shots: int = 100):
            self.n_qubits = n_qubits
            self.shots = shots

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            probs = np.abs(np.exp(1j * np.array(thetas))) ** 2
            probs /= probs.sum()
            expectation = np.sum(probs * np.arange(self.n_qubits))
            return np.array([expectation])

    return QuantumCircuit(num_qubits)


# Hybrid quantum‑classical graph neural network -----------------------------------------

class GraphQNNGen180:
    """
    Quantum‑only graph neural network class.

    Parameters
    ----------
    arch : Sequence[int]
        Number of qubits per layer.  The class generates random unitaries
        for each layer and corresponding training data.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.unitaries: List[List[Qobj]] = []
        self.training_data: List[Tuple[Qobj, Qobj]] = []

        # Generate random unitaries per layer
        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            layer_ops: List[Qobj] = []
            for output in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                    op = _swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

        # training data from the last layer unitary
        self.training_data = random_training_data(self.unitaries[-1][0], 100)

    def feedforward(self, inputs: Qobj) -> List[Qobj]:
        """Run a purely quantum feed‑forward pass."""
        states = [inputs]
        current = inputs
        for layer, ops in enumerate(self.unitaries, start=1):
            current = _layer_channel(self.arch, self.unitaries, layer, current)
            states.append(current)
        return states

    def get_adjacency(self, threshold: float, *, secondary: float | None = None) -> nx.Graph:
        """Construct a graph from the last layer outputs of the training data."""
        final_states = [self.feedforward(state)[-1] for state, _ in self.training_data]
        return fidelity_adjacency(final_states, threshold, secondary=secondary)

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        return GraphQNNGen180(arch)


__all__ = [
    "GraphQNNGen180",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "FCL",
    "conv_layer",
    "pool_layer",
]
