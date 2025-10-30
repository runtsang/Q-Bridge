from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import qutip as qt

Tensor = np.ndarray


class GraphQNN:
    """A quantum‑graph‑neural‑network style model using PennyLane QNodes."""

    def __init__(self, architecture: Sequence[int], dev: qml.Device):
        self.arch = list(architecture)
        self.dev = dev
        # Random parameters for each layer (output × input)
        self.params: List[Tensor] = [
            np.random.randn(out, in_) for in_, out in zip(self.arch[:-1], self.arch[1:])
        ]

    def _circuit(self, x: Tensor, *params: Tensor) -> Tensor:
        """PennyLane circuit that mirrors the classical feedforward logic."""
        # Encode input as rotations
        for i, val in enumerate(x):
            qml.RX(val, wires=i)
        # Layerwise parameterised gates
        for layer_params in params:
            for out_idx in range(layer_params.shape[0]):
                for in_idx in range(layer_params.shape[1]):
                    qml.Rot(layer_params[out_idx, in_idx], wires=out_idx)
        # Measurement
        return qml.expval(qml.PauliZ(0))

    def qnode(self) -> qml.QNode:
        """Return a PennyLane QNode with autograd interface."""
        return qml.QNode(self._circuit, self.dev, interface="autograd")

    def forward(self, x: Tensor) -> List[Tensor]:
        """Return activations for each layer."""
        activations = [x]
        current = x
        for layer_params in self.params:
            qnode = self.qnode()
            current = qnode(current, *layer_params)
            activations.append(np.array(current))
        return activations

    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        """Squared overlap between two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from pairwise state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """Generate synthetic data by applying a target unitary to random states."""
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            amplitudes = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
            amplitudes /= np.linalg.norm(amplitudes)
            state = qt.Qobj(amplitudes.reshape(-1, 1), dims=[2] * num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Create a random unitary and training data for the QNN."""
        target_unitary = qt.random_unitary(2 ** arch[-1])
        training_data = GraphQNN.random_training_data(target_unitary, samples)
        # Layerwise random unitaries (output × input)
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            in_f = arch[layer - 1]
            out_f = arch[layer]
            layer_ops: List[qt.Qobj] = []
            for _ in range(out_f):
                op = qt.random_unitary(2 ** (in_f + 1))
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return list(arch), unitaries, training_data, target_unitary
