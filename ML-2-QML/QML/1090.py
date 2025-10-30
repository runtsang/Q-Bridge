"""Graph‑based quantum neural network utilities with hybrid loss and dataset generation.

This module extends the original QML seed by:
* Adding a variational quantum circuit layer that uses PennyLane.
* Providing a classical MLP head that maps the final quantum state vector to a real target.
* Exposing a `HybridLoss` that mixes MSE with a fidelity penalty.
* Adding a `generate_graph_dataset` helper that builds random graphs and feeds them through the circuit.
* Adding a `full_fidelity_matrix` helper that returns the full fidelity matrix for a set of quantum states.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Device for PennyLane (CPU simulator)
dev = qml.device("default.qubit", wires=8)  # maximum wires used in random circuits


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    mat, _ = np.linalg.qr(mat)  # orthogonalize
    return mat


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Return a random pure state vector of length 2**num_qubits."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(
    unitary: np.ndarray, samples: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate training pairs (state, U * state)."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational circuit and training data."""
    # Build a list of unitary matrices for each layer
    unitaries: List[np.ndarray] = []
    for layer_size in qnn_arch[1:]:
        unitaries.append(_random_qubit_unitary(layer_size))
    target_unitary = unitaries[-1]
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), unitaries, training_data, target_unitary


def _variational_circuit(params: np.ndarray, wires: Sequence[int]) -> None:
    """Simple variational circuit: rotation gates followed by CNOT entanglement."""
    for i, w in enumerate(wires):
        qml.RX(params[3 * i], wires=w)
        qml.RY(params[3 * i + 1], wires=w)
        qml.RZ(params[3 * i + 2], wires=w)
    # Entangle adjacent qubits
    for i in range(len(wires) - 1):
        qml.CNOT(wires[i], wires[i + 1])


def _qnode(unitary: np.ndarray, wires: Sequence[int]):
    """Return a PennyLane QNode that applies the variational circuit and then the target unitary."""
    @qml.qnode(dev, interface="torch")
    def circuit(state: torch.Tensor, params: torch.Tensor):
        # Prepare the state
        qml.QubitStateVector(state, wires=wires)
        # Variational layer
        _variational_circuit(params, wires)
        # Apply the fixed target unitary
        qml.QubitUnitary(unitary, wires=wires)
        return qml.state()
    return circuit


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[torch.Tensor]]:
    """Run a forward pass through the variational circuit and record state vectors."""
    activations: List[List[torch.Tensor]] = []
    for state_np, _ in samples:
        state = torch.tensor(state_np, dtype=torch.complex64)
        layer_states = [state]
        current = state
        for unitary in unitaries:
            # Random parameters for the variational layer
            params = torch.randn(3 * current.shape[0].bit_length() - 3, dtype=torch.float32)
            qnode = _qnode(unitary, wires=range(current.shape[0].bit_length() - 1))
            current = qnode(current, params)
            layer_states.append(current)
        activations.append(layer_states)
    return activations


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared overlap between two normalized quantum state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.abs(torch.dot(a_norm.conj(), b_norm)) ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def full_fidelity_matrix(states: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the full fidelity matrix for a list of quantum state vectors."""
    n = len(states)
    mat = torch.empty(n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(i, n):
            fid = state_fidelity(states[i], states[j])
            mat[i, j] = fid
            mat[j, i] = fid
    return mat


def generate_graph_dataset(
    num_graphs: int,
    qnn_arch: Sequence[int],
    samples_per_graph: int,
) -> List[Dict]:
    """Generate a list of random graphs with associated quantum data."""
    dataset: List[Dict] = []
    for _ in range(num_graphs):
        g = nx.gnp_random_graph(num_nodes=qnn_arch[0], p=0.3)
        arch, unitaries, training_data, _ = random_network(qnn_arch, samples_per_graph)
        activations = feedforward(arch, unitaries, training_data)
        states = [act[-1] for act in activations]
        dataset.append({"graph": g, "activations": activations, "states": states})
    return dataset


class HybridLoss(nn.Module):
    """Blend MSE loss with a fidelity penalty between predictions and true targets."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        # Fidelity penalty
        pred_norm = pred / (torch.norm(pred, dim=0, keepdim=True) + 1e-12)
        target_norm = target / (torch.norm(target, dim=0, keepdim=True) + 1e-12)
        fid = torch.clamp(torch.abs(torch.dot(pred_norm.conj(), target_norm)), 0.0, 1.0)
        fid_loss = 1.0 - fid
        return self.alpha * mse_loss + (1.0 - self.alpha) * fid_loss


class SimpleMLPHead(nn.Module):
    """Two‑layer MLP that maps the final quantum state vector to a real‑valued target."""

    def __init__(self, in_features: int, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class GraphQNNHybridModel(nn.Module):
    """Hybrid quantum‑classical model that runs a variational circuit followed by a classical head."""

    def __init__(self, qnn_arch: Sequence[int], head_hidden: int = 16):
        super().__init__()
        self.arch = list(qnn_arch)
        self.unitaries = [torch.tensor(_random_qubit_unitary(layer), dtype=torch.complex64)
                          for layer in qnn_arch[1:]]
        self.head = SimpleMLPHead(qnn_arch[-1], hidden=head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = x
        for unitary in self.unitaries:
            params = torch.randn(3 * current.shape[0].bit_length() - 3, dtype=torch.float32)
            qnode = _qnode(unitary.numpy(), wires=range(current.shape[0].bit_length() - 1))
            current = qnode(current, params)
        return self.head(current)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "full_fidelity_matrix",
    "generate_graph_dataset",
    "HybridLoss",
    "SimpleMLPHead",
    "GraphQNNHybridModel",
]
