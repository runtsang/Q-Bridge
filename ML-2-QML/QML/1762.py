"""Hybrid Graph Neural Network (quantum side).

This module implements a parameterised variational circuit that learns a
unitary mapping between input and target states.  A fidelity‑based graph
regulariser is added to the loss to encourage smoothness between
neighbouring states in the batch.
"""

from __future__ import annotations

import itertools
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector, random_statevector, random_unitary, random_integer

Tensor = Statevector


def _random_unitary(num_qubits: int) -> Statevector:
    """Return a random unitary as a Statevector (operator)."""
    unitary_matrix = random_unitary(2 ** num_qubits).data
    return Statevector(unitary_matrix)


def random_training_data(
    unitary: Statevector, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs (|ψ>, U|ψ>) with random input states."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = random_statevector(2 ** unitary.num_qubits)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Return architecture, list of unitaries per layer, training data,
    and the target unitary for the last layer."""
    # target unitary is the last layer
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Statevector]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Statevector] = []
        for output in range(num_outputs):
            op = _random_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _apply_circuit(state: Statevector, circuit: QuantumCircuit) -> Statevector:
    """Apply a quantum circuit to a Statevector."""
    return circuit.apply(state)


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Statevector]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Return the state after each layer for each sample."""
    stored_states: List[List[Statevector]] = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for layer in range(1, len(qnn_arch)):
            # compose all gates of this layer
            combined = Statevector(np.identity(2 ** (qnn_arch[layer]), dtype=complex))
            for op in unitaries[layer]:
                combined = op @ combined
            current = combined @ current
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return the absolute squared overlap between pure states a and b."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNN__gen351:
    """Quantum Graph QNN with parameterised variational circuit and graph regulariser."""

    def __init__(
        self,
        arch: Sequence[int],
        learning_rate: float = 0.01,
        graph_threshold: float = 0.9,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
        seed: int | None = None,
    ):
        self.arch = list(arch)
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # Build a parameterised circuit (RealAmplitudes) for the whole network
        self.num_qubits = self.arch[-1]
        self.circuit = RealAmplitudes(self.num_qubits, reps=2)
        self.params = np.random.uniform(0, 2 * np.pi, self.circuit.num_parameters)

        # Target unitary for supervised learning (fixed random unitary)
        self.target = _random_unitary(self.num_qubits)

    def _apply(self, state: Statevector) -> Statevector:
        """Apply the parameterised circuit to a state."""
        circ = self.circuit.assign_parameters(self.params)
        return circ.apply(state)

    def _fidelity_loss(self, batch: List[Tuple[Statevector, Statevector]]) -> float:
        """Compute average negative fidelity (to be minimised)."""
        losses = []
        for inp, target in batch:
            out = self._apply(inp)
            fid = state_fidelity(out, target)
            losses.append(-fid)
        return np.mean(losses)

    def _graph_regulariser(self, outputs: List[Statevector]) -> float:
        """Graph regulariser based on fidelity adjacency."""
        graph = fidelity_adjacency(
            outputs,
            self.graph_threshold,
            secondary=self.secondary_threshold,
            secondary_weight=self.secondary_weight,
        )
        reg = 0.0
        for i, j, data in graph.edges(data=True):
            weight = data["weight"]
            reg += weight * np.linalg.norm(outputs[i] - outputs[j]) ** 2
        return reg / len(outputs)

    def train_step(self, batch: List[Tuple[Statevector, Statevector]]) -> float:
        """Perform one gradient step using parameter‑shift rule."""
        # compute current loss
        outputs = [self._apply(inp) for inp, _ in batch]
        loss = self._fidelity_loss(batch) + self._graph_regulariser(outputs)

        # parameter shift gradient
        grads = np.zeros_like(self.params)
        shift = np.pi / 2
        for idx in range(len(self.params)):
            params_plus = self.params.copy()
            params_plus[idx] += shift
            outputs_plus = [self._apply(inp) for inp, _ in batch]
            loss_plus = self._fidelity_loss(batch) + self._graph_regulariser(outputs_plus)

            params_minus = self.params.copy()
            params_minus[idx] -= shift
            outputs_minus = [self._apply(inp) for inp, _ in batch]
            loss_minus = self._fidelity_loss(batch) + self._graph_regulariser(outputs_minus)

            grads[idx] = (loss_plus - loss_minus) / (2 * np.sin(shift))

        # update parameters
        self.params -= self.learning_rate * grads
        return loss

    def fit(self, dataset: List[Tuple[Statevector, Statevector]], epochs: int = 50) -> List[float]:
        losses: List[float] = []
        for _ in range(epochs):
            loss = self.train_step(dataset)
            losses.append(loss)
        return losses

    def predict(self, inp: Statevector) -> Statevector:
        return self._apply(inp)


__all__ = [
    "GraphQNN__gen351",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
