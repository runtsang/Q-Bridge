"""Hybrid quantum graph neural network utilities.

The quantum implementation uses Pennylane to create variational
circuits and to compute state‑fidelity kernels.  The API mirrors the
classical version: :class:`GraphQNN` stores architecture and a list of
parameter‑free QNodes per layer, and exposes the same utility functions
for random network generation, training data, state fidelity, and
fidelity‑based graph construction.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
import pennylane as qml

Qobj = np.ndarray  # type alias for clarity


class GraphQNN:
    """Quantum graph neural network based on variational unitaries.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes of the network.
    circuits : Optional[List[List[qml.QNode]]]
        Pre‑constructed QNodes for each layer.  If ``None`` random
        unitaries are generated.
    target_unitary : Optional[np.ndarray]
        Ground‑truth unitary for data generation.
    """

    def __init__(
        self,
        arch: Sequence[int],
        circuits: Optional[List[List[qml.QNode]]] = None,
        target_unitary: Optional[np.ndarray] = None,
    ):
        self.arch = list(arch)
        self.circuits = circuits or self._random_circuits()
        self.target_unitary = target_unitary or self._random_target()

    @staticmethod
    def _random_target() -> np.ndarray:
        """Return a random unitary for the output layer."""
        num_qubits = 1  # output layer uses a single qubit
        return qml.random_unitary(num_qubits)

    def _random_circuits(self) -> List[List[qml.QNode]]:
        """Generate a list of QNodes per layer."""
        circuits: List[List[qml.QNode]] = []
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[qml.QNode] = []
            for _ in range(num_outputs):
                num_qubits = num_inputs + 1  # one extra qubit for the output
                unitary = qml.random_unitary(num_qubits)
                dev = qml.device("default.qubit", wires=num_qubits)

                @qml.qnode(dev, interface="autograd")
                def qnode(state: np.ndarray, unitary=unitary):
                    qml.StateVector(state, wires=range(num_qubits))
                    qml.QubitUnitary(unitary, wires=range(num_qubits))
                    return qml.state()

                layer_ops.append(qnode)
            circuits.append(layer_ops)
        return circuits

    def feedforward(
        self, samples: Iterable[Tuple[Qobj, Qobj]]
    ) -> List[List[Qobj]]:
        """Propagate each sample through the quantum network.

        Parameters
        ----------
        samples : Iterable[Tuple[Qobj, Qobj]]
            Iterable yielding ``(state, target)`` pairs.  The target is
            ignored by the forward pass but kept for API compatibility.

        Returns
        -------
        List[List[Qobj]]
            List of state lists per sample.  Each inner list contains
            the input state followed by the output of every layer.
        """
        stored: List[List[Qobj]] = []
        for state, _ in samples:
            layerwise = [state]
            current = state
            for layer_ops in self.circuits:
                # For simplicity we apply the first QNode of the layer.
                # Multi‑output support is omitted for brevity.
                current = layer_ops[0](current)
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    @staticmethod
    def random_training_data(
        unitary: np.ndarray, samples: int
    ) -> List[Tuple[Qobj, Qobj]]:
        """Generate synthetic training data by applying a unitary."""
        dataset: List[Tuple[Qobj, Qobj]] = []
        dim = unitary.shape[0]
        for _ in range(samples):
            state = np.random.randn(dim, dtype=np.complex128)
            state /= np.linalg.norm(state)
            target = unitary @ state
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(
        arch: Sequence[int], samples: int
    ) -> Tuple[List[int], List[List[qml.QNode]], List[Tuple[Qobj, Qobj]], np.ndarray]:
        """Generate a random quantum network and training data.

        Returns
        -------
        arch : List[int]
            The network architecture.
        circuits : List[List[qml.QNode]]
            List of QNodes per layer.
        training_data : List[Tuple[Qobj, Qobj]]
            Synthetic dataset.
        target_unitary : np.ndarray
            The unitary used to generate the targets.
        """
        dummy = GraphQNN(arch)
        circuits = dummy.circuits
        target_unitary = dummy.target_unitary
        training_data = GraphQNN.random_training_data(target_unitary, samples)
        return list(arch), circuits, training_data, target_unitary

    @staticmethod
    def state_fidelity(a: Qobj, b: Qobj) -> float:
        """Return the absolute squared overlap between pure states."""
        return float(np.abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Qobj],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


# Backward‑compatible wrappers ------------------------------------------------

def feedforward(
    qnn_arch: Sequence[int],
    circuits: Sequence[Sequence[qml.QNode]],
    samples: Iterable[Tuple[Qobj, Qobj]],
) -> List[List[Qobj]]:
    """Wrapper that forwards to :class:`GraphQNN`."""
    return GraphQNN(qnn_arch, list(circuits)).feedforward(samples)


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[List[qml.QNode]], List[Tuple[Qobj, Qobj]], np.ndarray]:
    """Wrapper for :meth:`GraphQNN.random_network`."""
    return GraphQNN.random_network(qnn_arch, samples)


def random_training_data(
    unitary: np.ndarray, samples: int
) -> List[Tuple[Qobj, Qobj]]:
    """Wrapper for :meth:`GraphQNN.random_training_data`."""
    return GraphQNN.random_training_data(unitary, samples)


def state_fidelity(a: Qobj, b: Qobj) -> float:
    """Wrapper for :meth:`GraphQNN.state_fidelity`."""
    return GraphQNN.state_fidelity(a, b)


def fidelity_adjacency(
    states: Sequence[Qobj],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Wrapper for :meth:`GraphQNN.fidelity_adjacency`."""
    return GraphQNN.fidelity_adjacency(
        states, threshold, secondary=secondary, secondary_weight=secondary_weight
    )
