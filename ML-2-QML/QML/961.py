"""GraphQNN__gen246: Quantum neural network with hybrid training interface.

This module implements a variational circuit that can be executed on Pennylane
backends.  It mirrors the classical interface: a `train` method that accepts a
classical module and runs a joint loss over classical and quantum predictions.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

Tensor = torch.Tensor
Graph = nx.Graph

# --------------------------------------------------------------------------- #
# 1.  Utilities copied from the seed
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Return a random unitary matrix of shape (2**num_qubits, 2**num_qubits)."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return torch.tensor(q, dtype=torch.cfloat)


def _random_qubit_state(num_qubits: int) -> Tensor:
    """Return a random pure state vector of length 2**num_qubits."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return torch.tensor(vec, dtype=torch.cfloat)


def random_training_data(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset for the target unitary."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        inp = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        out = unitary @ inp
        dataset.append((inp, out))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random unitary for the last layer and a training set."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), [target_unitary], training_data, target_unitary


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two pure states."""
    return float(abs((a.conj().t() @ b).item()) ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> Graph:
    """Build a graph where edges encode fidelity ≥ threshold."""
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
# 2.  Variational circuit
# --------------------------------------------------------------------------- #

class GraphQNN__gen246(nn.Module):
    """Variational quantum neural network that can be trained jointly with a classical GNN."""

    def __init__(
        self,
        arch: Sequence[int],
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.arch = list(arch)
        self.num_qubits = self.arch[-1]
        self.lr = lr
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.num_layers = len(self.arch) - 1

        # Parameters: (num_layers, num_qubits, 3) for RX, RY, RZ
        self.params = nn.Parameter(
            torch.randn(self.num_layers, self.num_qubits, 3, dtype=torch.float32, device=self.device)
        )

        # Pennylane device
        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=1, interface="torch")

        # Build the qnode
        def _circuit(params: Tensor, input_state: Tensor) -> List[Tensor]:
            qml.QubitStateVector(input_state, wires=range(self.num_qubits))
            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    qml.RX(params[l, q, 0], wires=q)
                    qml.RY(params[l, q, 1], wires=q)
                    qml.RZ(params[l, q, 2], wires=q)
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(wires=q)) for q in range(self.num_qubits)]

        self.circuit = qml.qnode(_circuit, dev=self.dev, interface="torch")

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #

    def forward(self, input_state: Tensor) -> Tensor:
        """Execute the circuit and return the expectation values of Z on each qubit."""
        return self.circuit(self.params, input_state)

    # --------------------------------------------------------------------- #
    # Helper: encode a graph into a quantum state
    # --------------------------------------------------------------------- #

    def encode_graph_to_state(self, graph: Graph) -> Tensor:
        """
        Simple amplitude encoding of the flattened node features.
        The vector is padded with zeros to length 2**num_qubits and normalised.
        """
        vec = []
        for _, data in graph.nodes(data=True):
            vec.extend(data["feature"].tolist())
        vec = np.array(vec, dtype=np.complex128)
        target_len = 2 ** self.num_qubits
        if len(vec) < target_len:
            vec = np.pad(vec, (0, target_len - len(vec)), "constant")
        else:
            vec = vec[:target_len]
        vec = vec / np.linalg.norm(vec)
        return torch.tensor(vec, dtype=torch.cfloat, device=self.device)

    # --------------------------------------------------------------------- #
    # Training utilities
    # --------------------------------------------------------------------- #

    def train(
        self,
        classical_module: "GraphQNN__gen246",
        data: List[Tuple[Graph, Tensor]],
        epochs: int = 10,
    ) -> None:
        """Jointly train this quantum circuit and a classical GNN."""
        self.train()
        classical_module.train()
        # Combine parameters
        combined_params = list(self.parameters()) + list(classical_module.parameters())
        optimizer = torch.optim.Adam(combined_params, lr=self.lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for graph, labels in data:
                # Classical predictions
                logits = classical_module.forward_from_graph(graph)
                # Quantum predictions
                input_state = self.encode_graph_to_state(graph)
                quantum_preds = self.forward(input_state)
                # Losses
                loss_cls = F.binary_cross_entropy_with_logits(logits, labels.to(self.device))
                loss_qml = F.binary_cross_entropy_with_logits(quantum_preds, labels.to(self.device))
                loss = loss_cls + loss_qml
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs} joint loss: {epoch_loss / len(data):.4f}")

    # --------------------------------------------------------------------- #
    # Static helper methods
    # --------------------------------------------------------------------- #

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Return a random unitary for the last layer and a training set."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: Tensor, samples: int):
        """Generate a random dataset for the target unitary."""
        return random_training_data(unitary, samples)

    @staticmethod
    def random_linear(in_features: int, out_features: int) -> Tensor:
        """Return a random unitary matrix."""
        return _random_qubit_unitary(out_features)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the absolute squared overlap between two pure states."""
        return state_fidelity(a, b)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[Tensor],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Placeholder for compatibility."""
        return []

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> Graph:
        """Build a graph where edges encode fidelity ≥ threshold."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "GraphQNN__gen246",
    "random_network",
    "random_training_data",
    "random_linear",
    "state_fidelity",
    "fidelity_adjacency",
]
