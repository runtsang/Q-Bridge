"""Quantum‑enhanced graph neural network.

This module implements the same :class:`GraphQNN` interface as the
pure‑classical version but replaces the linear layers with
variational quantum circuits.  The circuits are built with PennyLane
and are fully differentiable with PyTorch, allowing joint training
of classical and quantum parameters.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import networkx as nx
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
State = np.ndarray  # quantum state vector


class GraphQNN(nn.Module):
    """Quantum‑enhanced graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        List of feature dimensionalities for each layer.  The first
        element is the input dimensionality.
    n_qubits : int, default 0
        Number of quantum wires to allocate.  For the pure quantum
        construction ``n_qubits`` is ignored because each layer
        determines its own number of wires from ``arch``.
    """

    class QLayer(nn.Module):
        """Variational circuit that implements a layer of the QNN."""
        def __init__(self, wires: int) -> None:
            super().__init__()
            self.wires = wires
            self.device = qml.device("default.qubit", wires=wires)
            # Trainable parameters – one RX per qubit
            self.params = nn.Parameter(torch.randn(wires, requires_grad=True))

            @qml.qnode(self.device, interface="torch")
            def circuit(x: Tensor) -> List[Tensor]:
                # Encode classical features as rotation angles (RX)
                for i in range(self.wires):
                    qml.RX(x[i], wires=i)
                # Trainable rotation layer
                for i in range(self.wires):
                    qml.RX(self.params[i], wires=i)
                # Entangling CNOT chain
                for i in range(self.wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Return expectation values of Pauli‑Z for each qubit
                return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

            self.circuit = circuit

        def forward(self, x: Tensor) -> Tensor:
            return torch.stack(self.circuit(x))

    def __init__(self, arch: Sequence[int], n_qubits: int = 0) -> None:
        super().__init__()
        self.arch = list(arch)
        # Create one QLayer per hidden transition
        self.layers = nn.ModuleList(
            [self.QLayer(arch[i]) for i in range(len(arch) - 1)]
        )
        self.n_qubits = n_qubits

    def forward(self, x: Tensor) -> List[Tensor]:
        """Forward propagation through the hybrid network.

        Parameters
        ----------
        x : Tensor
            Node feature matrix of shape ``(num_nodes, in_features)``.

        Returns
        -------
        List[Tensor]
            List of activations per layer including the input.
        """
        activations: List[Tensor] = [x]
        current = x
        for layer in self.layers:
            # Apply the quantum layer to each node in the batch
            outputs = torch.stack([layer(node) for node in current])
            activations.append(outputs)
            current = outputs
        return activations

    # ------------------------------------------------------------------
    #  Static helpers – mirror the original seed implementation
    # ------------------------------------------------------------------
    @staticmethod
    def _random_unitary(dim: int) -> np.ndarray:
        # Generate a random unitary using the Haar measure
        random_matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        q, _ = np.linalg.qr(random_matrix)
        return q

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[State, State]]:
        dataset: List[Tuple[State, State]] = []
        dim = unitary.shape[0]
        for _ in range(samples):
            # Random pure state
            amp = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
            amp /= np.linalg.norm(amp)
            target = unitary @ amp
            dataset.append((amp, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Generate a random target unitary and a matching training set.

        Parameters
        ----------
        qnn_arch : Sequence[int]
            Architecture of the network.
        samples : int
            Number of training samples.

        Returns
        -------
        Tuple[List[int], List[np.ndarray], List[Tuple[State, State]], np.ndarray]
            Architecture, list of random unitary matrices per layer,
            training data, and the overall target unitary.
        """
        dim = qnn_arch[-1]
        target_unitary = GraphQNN._random_unitary(dim)
        training_data = GraphQNN.random_training_data(target_unitary, samples)
        unitaries: List[np.ndarray] = [GraphQNN._random_unitary(in_f) for in_f in qnn_arch]
        return list(qnn_arch), unitaries, training_data, target_unitary

    @staticmethod
    def state_fidelity(a: State, b: State) -> float:
        """Squared overlap between two pure states."""
        return float(abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[State],
        threshold: float,
        *,
        secondary: float | None = None,
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


__all__ = [
    "GraphQNN",
]
