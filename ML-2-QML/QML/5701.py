from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import numpy as np
import qutip as qt
import scipy as sc

Tensor = qt.Qobj

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

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

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

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

@dataclass
class GraphQLayerParameters:
    """Parameters for a quantum graph layer."""
    unitary: qt.Qobj

    @staticmethod
    def random(num_qubits: int) -> "GraphQLayerParameters":
        return GraphQLayerParameters(unitary=_random_qubit_unitary(num_qubits))

class GraphQNN:
    """Quantum graph neural network using qutip."""
    def __init__(self, arch: Sequence[int], clip: bool = True):
        self.arch = list(arch)
        self.clip = clip
        self.layers: List[GraphQLayerParameters] = [GraphQLayerParameters.random(n) for n in arch]

    def _layer_channel(self, layer_idx: int, input_state: Tensor) -> Tensor:
        num_inputs = self.arch[layer_idx - 1]
        num_outputs = self.arch[layer_idx]
        state = qt.tensor(input_state, _tensored_zero(num_outputs))
        unitary = self.layers[layer_idx].unitary
        return _partial_trace_remove(unitary * state * unitary.dag(), range(num_inputs))

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored_states: List[List[Tensor]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(self.arch)):
                current_state = self._layer_channel(layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        target_unitary = GraphQLayerParameters.random(qnn_arch[-1]).unitary
        training_data = GraphQNN.random_training_data(target_unitary, samples)
        return list(qnn_arch), [layer.unitary for layer in GraphQNN._build_random_layers(qnn_arch)], training_data, target_unitary

    @staticmethod
    def _build_random_layers(qnn_arch: Sequence[int]) -> List[GraphQLayerParameters]:
        return [GraphQLayerParameters.random(num_qubits) for num_qubits in qnn_arch]

    @staticmethod
    def random_training_data(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = _random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = [
    "GraphQNN",
    "GraphQLayerParameters",
]
