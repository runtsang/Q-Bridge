"""
Hybrid Graph Neural Network for quantum training with Qiskit back‑end.

The module provides:
* `GraphQNNHybrid` – a class that mirrors the classical version but uses Qiskit
  unitaries for quantum layers.
* Functions to generate random quantum networks, synthetic training data,
  forward propagation using unitary circuits and fidelity‑based adjacency graphs.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple, Sequence, Optional, Iterable

import numpy as np
import networkx as nx
import qiskit as qk
import qiskit.quantum_info as qi
import torch

# --------------------------------------------------------------------------- #
# Quantum utilities – random unitaries, state propagation
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qi.Qobj:
    """Identity operator with proper tensor dimensions."""
    identity = qi.Qobj(np.eye(2 ** num_qubits))
    identity.dims = [[2] * num_qubits, [2] * num_qubits]
    return identity


def _tensored_zero(num_qubits: int) -> qi.Qobj:
    """Zero projector with proper tensor dimensions."""
    zero = qi.Qobj(np.zeros((2 ** num_qubits, 1)))
    zero.dims = [[2] * num_qubits, [1] * num_qubits]
    return zero


def _swap_registers(op: qi.Qobj, source: int, target: int) -> qi.Qobj:
    """Swap two qubits in a tensor product operator."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qi.Qobj:
    """Generate a random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    qobj = qi.Qobj(matrix)
    qobj.dims = [[2] * num_qubits, [2] * num_qubits]
    return qobj


def _random_qubit_state(num_qubits: int) -> qi.Qobj:
    """Generate a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    vec /= np.linalg.norm(vec)
    qobj = qi.Qobj(vec)
    qobj.dims = [[2] * num_qubits, [1] * num_qubits]
    return qobj


def random_training_data(unitary: qi.Qobj, samples: int) -> List[Tuple[qi.Qobj, qi.Qobj]]:
    """Generate synthetic data for a regression task using a unitary."""
    dataset: List[Tuple[qi.Qobj, qi.Qobj]] = []
    for _ in range(samples):
        state = _random_qubit_state(len(unitary.dims[0]))
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qi.Qobj]], List[Tuple[qi.Qobj, qi.Qobj]], qi.Qobj]:
    """
    Return architecture, a list of lists of unitaries per layer, training data and
    the target unitary (last layer).
    """
    # Target unitary for the last layer
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # Build unitaries for intermediate layers
    unitaries: List[List[qi.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_q = qnn_arch[layer - 1]
        out_q = qnn_arch[layer]
        layer_ops: List[qi.Qobj] = []
        for out_idx in range(out_q):
            op = _random_qubit_unitary(in_q + 1)
            if out_q > 1:
                op = qi.tensor(_random_qubit_unitary(in_q + 1), _tensored_id(out_q - 1))
                op = _swap_registers(op, in_q, in_q + out_idx)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_remove(state: qi.Qobj, remove: Sequence[int]) -> qi.Qobj:
    """Partial trace over the specified qubits."""
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.ptrace(keep)


def _layer_channel(qnn_arch: Sequence[int],
                   unitaries: Sequence[Sequence[qi.Qobj]],
                   layer: int,
                   input_state: qi.Qobj) -> qi.Qobj:
    """Apply a layer’s unitary and trace out unused qubits."""
    in_q = qnn_arch[layer - 1]
    out_q = qnn_arch[layer]
    state = qi.tensor(input_state, _tensored_zero(out_q))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(in_q))


def feedforward(qnn_arch: Sequence[int],
                unitaries: Sequence[Sequence[qi.Qobj]],
                samples: Iterable[Tuple[qi.Qobj, qi.Qobj]]) -> List[List[qi.Qobj]]:
    """Forward propagation through the quantum network."""
    stored_states: List[List[qi.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: qi.Qobj, b: qi.Qobj) -> float:
    """Squared overlap of two pure quantum states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qi.Qobj],
                       threshold: float,
                       *,
                       secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid Graph Neural Network – Qiskit implementation
# --------------------------------------------------------------------------- #
class GraphQNNHybrid:
    """
    Hybrid graph neural network that can inject quantum‑derived features
    at any layer of the network.  The quantum side uses Qiskit unitaries
    to propagate states; classical layers use simple linear maps.
    """

    def __init__(self,
                 *qnn_arch: int,
                 quantum_layers: Optional[List[int]] = None,
                 n_qubits: int = 2) -> None:
        self.qnn_arch = list(qnn_arch)
        self.quantum_layers = quantum_layers or []
        self.n_qubits = n_qubits

        # Classical weights (as tensors for easy conversion to Qobj)
        self.weights = [
            qi.Qobj(np.random.randn(out, in_))
            for in_, out in zip(self.qnn_arch[:-1], self.qnn_arch[1:])
        ]

        # Quantum unitaries per layer
        self.unitaries, _, _, _ = random_network(self.qnn_arch, samples=10)

    def forward(self, state: qi.Qobj) -> qi.Qobj:
        """Forward propagation through the hybrid network."""
        current = state
        for idx, layer_ops in enumerate(self.unitaries[1:]):  # skip dummy layer 0
            # Classical layer: apply linear map
            if idx not in self.quantum_layers:
                # Convert current state to vector and apply linear map
                vec = current.data.reshape(-1, 1)
                out_vec = self.weights[idx] @ vec
                current = qi.Qobj(out_vec, dims=current.dims)
            else:
                # Quantum layer: apply unitary and trace out input qubits
                current = _layer_channel(self.qnn_arch, self.unitaries, idx + 1, current)
        return current

    def fidelity_graph(self,
                       states: List[qi.Qobj],
                       threshold: float,
                       *,
                       secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        """Construct a fidelity‑based adjacency graph from quantum states."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
